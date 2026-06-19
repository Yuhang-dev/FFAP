from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ffap.json_utils import write_json
from stage1_smoke import collect_feature_stats, load_sae_compat, make_blocks
from stage2_gate_causal import (
    checkpoint_path,
    correlation_summary,
    feature_damage_for_selected,
    gate_decision,
    label_for_record,
    load_capability_losses,
    load_sweep_records,
    select_features,
)


@dataclass(frozen=True)
class MCExample:
    task: str
    example_id: str
    prompt: str
    choices: list[str]
    gold: int


@dataclass(frozen=True)
class ScoreRequest:
    example_key: str
    choice_index: int
    gold: int
    prompt: str
    continuation: str


def load_model(model_ref: str, device: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def normalize_continuation(text: str) -> str:
    text = text.strip()
    return text if text.startswith((" ", "\n", "\t")) else f" {text}"


def load_arc_easy(split: str, limit: int, seed: int) -> list[MCExample]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    examples: list[MCExample] = []
    for index in indices:
        row = ds[int(index)]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        answer = str(row["answerKey"]).strip()
        if answer not in labels:
            continue
        prompt = f"Question: {row['question'].strip()}\nAnswer:"
        examples.append(
            MCExample(
                task="arc_easy",
                example_id=f"arc_easy:{index}",
                prompt=prompt,
                choices=[normalize_continuation(choice) for choice in choices],
                gold=labels.index(answer),
            )
        )
        if len(examples) >= limit:
            break
    return examples


def load_hellaswag(split: str, limit: int, seed: int) -> list[MCExample]:
    ds = load_dataset("Rowan/hellaswag", split=split)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    examples: list[MCExample] = []
    for index in indices:
        row = ds[int(index)]
        ctx = str(row.get("ctx") or "").strip()
        if not ctx:
            ctx = f"{row.get('ctx_a', '').strip()} {row.get('ctx_b', '').strip()}".strip()
        endings = row["endings"]
        gold = int(row["label"])
        if gold < 0 or gold >= len(endings):
            continue
        examples.append(
            MCExample(
                task="hellaswag",
                example_id=f"hellaswag:{index}",
                prompt=ctx,
                choices=[normalize_continuation(choice) for choice in endings],
                gold=gold,
            )
        )
        if len(examples) >= limit:
            break
    return examples


def load_task_examples(tasks: str, split: str, limit_per_task: int, seed: int) -> list[MCExample]:
    examples: list[MCExample] = []
    for offset, task in enumerate(task.strip() for task in tasks.split(",") if task.strip()):
        task_seed = seed + offset * 997
        if task == "arc_easy":
            examples.extend(load_arc_easy(split, limit_per_task, task_seed))
        elif task == "hellaswag":
            examples.extend(load_hellaswag(split, limit_per_task, task_seed))
        else:
            raise ValueError(f"Unsupported task-causal dataset: {task}")
    if not examples:
        raise RuntimeError("No task examples loaded for task-causal gate.")
    return examples


def requests_from_examples(examples: list[MCExample]) -> list[ScoreRequest]:
    requests: list[ScoreRequest] = []
    for example in examples:
        for choice_index, continuation in enumerate(example.choices):
            requests.append(
                ScoreRequest(
                    example_key=example.example_id,
                    choice_index=choice_index,
                    gold=example.gold,
                    prompt=example.prompt,
                    continuation=continuation,
                )
            )
    return requests


def feature_texts_from_examples(examples: list[MCExample]) -> list[str]:
    texts = []
    for example in examples:
        texts.append(example.prompt + example.choices[example.gold])
    return texts


def encode_request(
    tokenizer: Any,
    request: ScoreRequest,
    max_length: int,
) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(request.prompt, add_special_tokens=False).input_ids
    continuation_ids = tokenizer(request.continuation, add_special_tokens=False).input_ids
    if not continuation_ids:
        continuation_ids = tokenizer(" ", add_special_tokens=False).input_ids
    if len(continuation_ids) >= max_length:
        continuation_ids = continuation_ids[: max_length - 1]
        prompt_ids = prompt_ids[-1:]
    else:
        prompt_budget = max_length - len(continuation_ids)
        prompt_ids = prompt_ids[-prompt_budget:]
    ids = prompt_ids + continuation_ids
    continuation_mask = [0] * len(prompt_ids) + [1] * len(continuation_ids)
    return ids, continuation_mask


def set_pad_token(tokenizer: Any) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad/eos token.")
    return int(tokenizer.pad_token_id)


def feature_ablation_hook(sae: Any, feature_id: int) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        flat = hidden.reshape(-1, hidden.shape[-1])
        with torch.no_grad():
            features = sae.encode(flat)
            feature_only = torch.zeros_like(features)
            feature_only[:, feature_id] = features[:, feature_id]
            zero_decoded = sae.decode(torch.zeros_like(features))
            feature_delta = sae.decode(feature_only) - zero_decoded
            patched = (flat.float() - feature_delta.float()).to(hidden.dtype)
            patched = patched.reshape_as(hidden)
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    return hook


@torch.no_grad()
def score_requests(
    model: Any,
    tokenizer: Any,
    requests: list[ScoreRequest],
    device: str,
    batch_size: int,
    max_length: int,
    sae: Any | None = None,
    layer: int | None = None,
    feature_id: int | None = None,
) -> dict[tuple[str, int], dict[str, float]]:
    pad_id = set_pad_token(tokenizer)
    encoded = [encode_request(tokenizer, request, max_length) for request in requests]
    layer_handle = None
    if feature_id is not None:
        if sae is None or layer is None:
            raise ValueError("Feature ablation scoring requires SAE and layer.")
        layer_handle = model.model.layers[layer].register_forward_hook(
            feature_ablation_hook(sae, feature_id)
        )

    out: dict[tuple[str, int], dict[str, float]] = {}
    try:
        for start in range(0, len(requests), batch_size):
            batch_requests = requests[start : start + batch_size]
            batch_encoded = encoded[start : start + batch_size]
            max_len = max(len(ids) for ids, _mask in batch_encoded)
            input_ids = torch.full(
                (len(batch_requests), max_len),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            continuation_mask = torch.zeros_like(input_ids)
            for row_index, (ids, cont_mask) in enumerate(batch_encoded):
                length = len(ids)
                input_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[row_index, :length] = 1
                continuation_mask[row_index, :length] = torch.tensor(
                    cont_mask, dtype=torch.long, device=device
                )

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            mask = continuation_mask[:, 1:] * attention_mask[:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            token_scores = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            token_scores = token_scores * mask
            sums = token_scores.sum(dim=-1)
            counts = mask.sum(dim=-1).clamp_min(1)
            means = sums / counts
            for row_index, request in enumerate(batch_requests):
                out[(request.example_key, request.choice_index)] = {
                    "sum_logprob": float(sums[row_index].detach().cpu()),
                    "mean_logprob": float(means[row_index].detach().cpu()),
                    "tokens": int(counts[row_index].detach().cpu()),
                    "gold": request.gold,
                }
    finally:
        if layer_handle is not None:
            layer_handle.remove()
    return out


def summarize_mc_scores(
    scores: dict[tuple[str, int], dict[str, float]],
    examples: list[MCExample],
    score_key: str,
) -> dict[str, Any]:
    margins = []
    gold_scores = []
    correct = 0
    by_task: dict[str, dict[str, float]] = {}
    for example in examples:
        choice_scores = [
            scores[(example.example_id, choice_index)][score_key]
            for choice_index in range(len(example.choices))
        ]
        gold_score = choice_scores[example.gold]
        other_scores = [
            score for choice_index, score in enumerate(choice_scores) if choice_index != example.gold
        ]
        margin = gold_score - max(other_scores)
        margins.append(margin)
        gold_scores.append(gold_score)
        prediction = max(range(len(choice_scores)), key=lambda index: choice_scores[index])
        correct += int(prediction == example.gold)
        task_summary = by_task.setdefault(
            example.task,
            {"count": 0.0, "correct": 0.0, "margin_sum": 0.0, "gold_score_sum": 0.0},
        )
        task_summary["count"] += 1
        task_summary["correct"] += int(prediction == example.gold)
        task_summary["margin_sum"] += margin
        task_summary["gold_score_sum"] += gold_score

    for summary in by_task.values():
        count = max(1.0, summary["count"])
        summary["accuracy"] = summary["correct"] / count
        summary["mean_margin"] = summary["margin_sum"] / count
        summary["mean_gold_score"] = summary["gold_score_sum"] / count

    return {
        "accuracy": correct / max(1, len(examples)),
        "mean_margin": sum(margins) / max(1, len(margins)),
        "mean_gold_score": sum(gold_scores) / max(1, len(gold_scores)),
        "examples": len(examples),
        "by_task": by_task,
    }


def compute_task_causal_scores(
    model: Any,
    tokenizer: Any,
    sae: Any,
    layer: int,
    examples: list[MCExample],
    feature_ids: list[int],
    batch_size: int,
    max_length: int,
    score_key: str,
    causal_score: str,
    device: str,
) -> tuple[dict[int, float], dict[str, Any]]:
    requests = requests_from_examples(examples)
    baseline_scores = score_requests(
        model,
        tokenizer,
        requests,
        device,
        batch_size,
        max_length,
    )
    baseline_summary = summarize_mc_scores(baseline_scores, examples, score_key)
    scores: dict[int, float] = {}
    feature_summaries = {}
    for feature_id in tqdm(feature_ids, desc="Task-causal feature ablations"):
        ablated_scores = score_requests(
            model,
            tokenizer,
            requests,
            device,
            batch_size,
            max_length,
            sae=sae,
            layer=layer,
            feature_id=feature_id,
        )
        ablated_summary = summarize_mc_scores(ablated_scores, examples, score_key)
        margin_delta = baseline_summary["mean_margin"] - ablated_summary["mean_margin"]
        gold_score_delta = (
            baseline_summary["mean_gold_score"] - ablated_summary["mean_gold_score"]
        )
        accuracy_delta = baseline_summary["accuracy"] - ablated_summary["accuracy"]
        if causal_score == "margin_delta":
            score = margin_delta
        elif causal_score == "gold_score_delta":
            score = gold_score_delta
        elif causal_score == "accuracy_delta":
            score = accuracy_delta
        else:
            raise ValueError(f"Unknown causal score: {causal_score}")
        scores[int(feature_id)] = float(score)
        feature_summaries[str(feature_id)] = {
            "margin_delta": float(margin_delta),
            "gold_score_delta": float(gold_score_delta),
            "accuracy_delta": float(accuracy_delta),
            "ablated_accuracy": float(ablated_summary["accuracy"]),
            "ablated_mean_margin": float(ablated_summary["mean_margin"]),
            "ablated_mean_gold_score": float(ablated_summary["mean_gold_score"]),
        }
    return scores, {"baseline": baseline_summary, "features": feature_summaries}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "target_sparsity",
        "checkpoint",
        "ppl",
        "ppl_relative_increase",
        "mean_acc_norm",
        "ability_loss",
        "ability_retention",
        "selected_mean_activation_l1",
        "selected_firing_rate_l1",
        "causal_weighted_mean_activation_l1",
        "abs_causal_weighted_mean_activation_l1",
        "causal_weighted_firing_rate_l1",
        "abs_causal_weighted_firing_rate_l1",
        "positive_causal_weight_sum",
        "abs_causal_weight_sum",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 task-conditioned causal gate")
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--tasks", default="arc_easy,hellaswag")
    parser.add_argument("--task-split", default="validation")
    parser.add_argument("--task-causal-limit", type=int, default=64)
    parser.add_argument("--feature-seq-len", type=int, default=128)
    parser.add_argument("--feature-blocks", type=int, default=16)
    parser.add_argument("--top-k-features", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--score-max-length", type=int, default=256)
    parser.add_argument("--score-key", choices=["mean_logprob", "sum_logprob"], default="mean_logprob")
    parser.add_argument(
        "--causal-score",
        choices=["margin_delta", "gold_score_delta", "accuracy_delta"],
        default="margin_delta",
    )
    parser.add_argument(
        "--feature-selection",
        choices=["firing_rate", "mean_activation", "activity_mass"],
        default="activity_mass",
    )
    parser.add_argument("--sweep-csv", action="append", default=None)
    parser.add_argument("--capability-csv", action="append", default=None)
    parser.add_argument("--max-sparsity", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="results/stage2_task_causal_gate.json")
    parser.add_argument("--out-csv", default="results/stage2_task_causal_gate.csv")
    args = parser.parse_args()
    if args.sweep_csv is None:
        args.sweep_csv = [
            "results/stage1_magnitude_sweep.csv",
            "results/stage1_wanda_sweep.csv",
        ]
    if args.capability_csv is None:
        args.capability_csv = [
            "results/stage1_capability_eval.csv",
            "results/stage1_wanda_capability_eval.csv",
        ]

    started = time.time()
    payload: dict[str, Any] = {
        "task": "stage2_task_causal_gate",
        "timestamp_unix": started,
        "config": vars(args),
        "status": "STARTED",
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Stage 2 task-causal gate.")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = "cuda:0"
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        examples = load_task_examples(
            args.tasks, args.task_split, args.task_causal_limit, args.seed
        )
        feature_texts = feature_texts_from_examples(examples)
        feature_blocks = make_blocks(
            tokenizer, feature_texts, args.feature_seq_len, args.feature_blocks, device
        )
        sae, sae_metadata = load_sae_compat(args.sae_release, args.sae_id, device)

        dense_model = load_model(args.model_id, device)
        dense_features = collect_feature_stats(dense_model, sae, feature_blocks, args.layer)
        feature_ids = select_features(
            dense_features, args.top_k_features, args.feature_selection
        )
        causal_scores, task_causal = compute_task_causal_scores(
            dense_model,
            tokenizer,
            sae,
            args.layer,
            examples,
            feature_ids,
            args.score_batch_size,
            args.score_max_length,
            args.score_key,
            args.causal_score,
            device,
        )
        del dense_model
        torch.cuda.empty_cache()

        records = load_sweep_records(args.sweep_csv, args.max_sparsity)
        if not records:
            raise RuntimeError("No sweep rows remained after applying filters.")
        ability_meta, ability_losses = load_capability_losses(args.capability_csv)
        rows = []
        for record in records:
            checkpoint = record["checkpoint"]
            label = label_for_record(record)
            ability = ability_losses.get(label) or ability_losses.get(checkpoint)
            if ability is None:
                raise KeyError(f"Missing capability eval for {label} / {checkpoint}")
            model = load_model(checkpoint, device)
            pruned_features = collect_feature_stats(model, sae, feature_blocks, args.layer)
            damage = feature_damage_for_selected(
                dense_features, pruned_features, feature_ids, causal_scores
            )
            row = {**record, **ability, **damage}
            rows.append(row)
            write_csv(Path(args.out_csv), rows)
            write_json(
                args.out_json,
                {
                    **payload,
                    "status": "RUNNING",
                    "completed_rows": len(rows),
                    "selected_features": feature_ids,
                },
            )
            del model
            torch.cuda.empty_cache()

        correlations = correlation_summary(rows)
        decision = gate_decision(correlations)
        causal_score_summary = {
            "min": min(causal_scores.values()) if causal_scores else None,
            "max": max(causal_scores.values()) if causal_scores else None,
            "mean": sum(causal_scores.values()) / len(causal_scores)
            if causal_scores
            else None,
            "positive_count": sum(1 for value in causal_scores.values() if value > 0),
        }
        payload.update(
            {
                "status": "PASS",
                "elapsed_sec": round(time.time() - started, 3),
                "torch": {
                    "version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0),
                },
                "sae": sae_metadata,
                "ability_meta": ability_meta,
                "task_examples": {
                    "count": len(examples),
                    "by_task": {
                        task: sum(1 for example in examples if example.task == task)
                        for task in sorted({example.task for example in examples})
                    },
                },
                "task_causal": task_causal,
                "selected_features": feature_ids,
                "causal_scores": {str(k): v for k, v in causal_scores.items()},
                "causal_score_summary": causal_score_summary,
                "rows": rows,
                "correlations": correlations,
                **decision,
                "outputs": {"json": args.out_json, "csv": args.out_csv},
                "conclusion": (
                    f"Stage 2 task-causal gate candidate: {decision['gate_status']}. "
                    f"{decision['reason']}"
                ),
            }
        )
        write_csv(Path(args.out_csv), rows)
        write_json(args.out_json, payload)
        print(f"wrote {args.out_json}")
        print(f"wrote {args.out_csv}")
        print(f"gate_status: {decision['gate_status']}")
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "FAIL",
                "elapsed_sec": round(time.time() - started, 3),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "conclusion": "Stage 2 task-causal gate failed; inspect logs.",
            }
        )
        write_json(args.out_json, payload)
        print(f"wrote {args.out_json}")
        print(f"status: FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
