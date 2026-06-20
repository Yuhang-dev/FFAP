from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ffap.json_utils import write_json
from stage1_smoke import collect_feature_stats, load_sae_compat
from stage2_gate_causal_v2 import (
    ability_batches,
    apply_protected_wanda,
    attribution_scores,
    collate_requests,
    collect_wanda_input_stats,
    evaluate_objective,
    evaluate_ppl,
    feature_reference,
    feature_texts_ability,
    get_wikitext_texts,
    load_model,
    load_task_examples,
    make_blocks,
    objective_summary,
    sharpen_weights,
)


PROTECTED_GROUPS = ("A_feature_grad", "A_loss_grad", "B_wanda", "C_random")
ALL_GROUPS = PROTECTED_GROUPS + ("D_wanda_no_protection",)


def _floats(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _config_fingerprint(args: argparse.Namespace) -> str:
    payload = vars(args).copy()
    payload.pop("step", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _gpu_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def _log(
    args: argparse.Namespace,
    name: str,
    started: float,
    status: str,
    key_numbers: dict[str, Any],
    conclusion: str,
) -> dict[str, Any]:
    payload = {
        "step": f"stage2_w1_{name}",
        "status": status,
        "elapsed_sec": round(time.time() - started, 3),
        "config": vars(args),
        "seeds": list(args.seeds),
        "torch": _gpu_summary(),
        "key_numbers": key_numbers,
        "conclusion": conclusion,
    }
    write_json(args.log_root / f"stage2_w1_{name}.json", payload)
    return payload


def split_ability_examples(args: argparse.Namespace) -> tuple[list[Any], list[Any]]:
    total = args.ability_calibration_per_task + args.ability_test_per_task
    examples = load_task_examples(args.tasks, args.task_split, total, args.split_seed)
    by_task: dict[str, list[Any]] = {}
    for example in examples:
        by_task.setdefault(example.task, []).append(example)
    calibration = []
    test = []
    for task, rows in sorted(by_task.items()):
        if len(rows) < total:
            raise RuntimeError(f"Task {task} yielded {len(rows)} examples, expected {total}.")
        calibration.extend(rows[: args.ability_calibration_per_task])
        test.extend(rows[args.ability_calibration_per_task : total])
    calibration_ids = {item.example_id for item in calibration}
    test_ids = {item.example_id for item in test}
    overlap = calibration_ids & test_ids
    if overlap:
        raise RuntimeError(f"Ability calibration/test split overlap: {sorted(overlap)[:5]}")
    return calibration, test


def writer_module_names(model: Any, target_layer: int, writer_scope: str) -> list[str]:
    modules = dict(model.named_modules())
    total_layers = len(getattr(model.model, "layers"))
    if writer_scope == "single":
        layers = range(target_layer, target_layer + 1)
    elif writer_scope == "upstream":
        layers = range(0, target_layer + 1)
    elif writer_scope == "all":
        layers = range(total_layers)
    else:
        raise ValueError(f"Unknown writer scope: {writer_scope}")
    names: list[str] = []
    for layer in layers:
        for suffix in ("self_attn.o_proj", "mlp.down_proj"):
            name = f"model.layers.{layer}.{suffix}"
            if isinstance(modules.get(name), torch.nn.Linear):
                names.append(name)
    if not names:
        raise RuntimeError(f"No writer modules found for scope={writer_scope}.")
    return names


def _enable_writer_grads(model: Any, writer_names: list[str]) -> dict[str, torch.nn.Linear]:
    model.requires_grad_(False)
    modules = dict(model.named_modules())
    writers = {}
    for name in writer_names:
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"Writer module is not linear: {name}")
        module.weight.requires_grad_(True)
        writers[name] = module
    return writers


def _zero_writer_grads(writers: dict[str, torch.nn.Linear]) -> None:
    for module in writers.values():
        module.weight.grad = None


def _init_score_maps(writers: dict[str, torch.nn.Linear]) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(module.weight.detach(), dtype=torch.float32, device="cpu")
        for name, module in writers.items()
    }


def _accumulate_weight_grad_scores(
    writers: dict[str, torch.nn.Linear],
    output: dict[str, torch.Tensor],
) -> None:
    for name, module in writers.items():
        grad = module.weight.grad
        if grad is None:
            continue
        score = grad.detach().float().abs() * module.weight.detach().float().abs()
        output[name] += score.cpu()


def _normalize_score_maps(score_maps: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    max_value = max(float(value.max()) for value in score_maps.values())
    if max_value <= 0:
        raise RuntimeError("Cross-layer gradient importance collapsed to zero.")
    return {name: value / max_value for name, value in score_maps.items()}


def crosslayer_feature_grad_scores(
    model: Any,
    tokenizer: Any,
    sae: Any,
    writer_names: list[str],
    target_layer: int,
    feature_weights: torch.Tensor,
    batches: list[Any],
    max_length: int,
    device: str,
) -> dict[str, torch.Tensor]:
    writers = _enable_writer_grads(model, writer_names)
    scores = _init_score_maps(writers)
    feature_weights = feature_weights.to(device=device, dtype=torch.float32)
    holder: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        flat = hidden.reshape(-1, hidden.shape[-1])
        holder["features"] = sae.encode(flat)
        return output

    handle = model.model.layers[target_layer].register_forward_hook(hook)
    try:
        for batch in tqdm(batches, desc="W1 feature-grad batches"):
            input_ids, attention_mask, _ = collate_requests(
                tokenizer, batch.requests, max_length, device
            )
            _zero_writer_grads(writers)
            holder.clear()
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            features = holder["features"]
            valid = attention_mask.reshape(-1).bool()
            selected = features[valid].float()
            objective = -(selected @ feature_weights).mean()
            objective.backward()
            _accumulate_weight_grad_scores(writers, scores)
            model.zero_grad(set_to_none=True)
    finally:
        handle.remove()
    return _normalize_score_maps(scores)


def crosslayer_loss_grad_scores(
    model: Any,
    tokenizer: Any,
    writer_names: list[str],
    batches: list[Any],
    max_length: int,
    device: str,
) -> dict[str, torch.Tensor]:
    import stage2_gate_causal_v2 as v2

    writers = _enable_writer_grads(model, writer_names)
    scores = _init_score_maps(writers)
    for batch in tqdm(batches, desc="W1 loss-grad batches"):
        input_ids, attention_mask, continuation_mask = v2.collate_requests(
            tokenizer, batch.requests, max_length, device
        )
        _zero_writer_grads(writers)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        choice_scores = v2.continuation_scores(
            outputs.logits, input_ids, attention_mask, continuation_mask
        )
        loss, _rows = v2.grouped_objective(choice_scores, batch)
        loss.backward()
        _accumulate_weight_grad_scores(writers, scores)
        model.zero_grad(set_to_none=True)
    return _normalize_score_maps(scores)


@torch.no_grad()
def protection_masks_from_scores(
    model: Any,
    input_stats: dict[str, dict[str, Any]],
    writer_names: list[str],
    score_maps: dict[str, torch.Tensor] | None,
    sparsity: float,
    protect_fraction: float,
    group: str,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    modules = dict(model.named_modules())
    total_writer_weights = sum(modules[name].weight.numel() for name in writer_names)
    budget = max(1, int(total_writer_weights * protect_fraction))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidates: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    at_risk_masks: dict[str, torch.Tensor] = {}
    for name in writer_names:
        module = modules[name]
        weight = module.weight.detach()
        cols = weight.shape[1]
        prune_per_row = int(cols * sparsity)
        keep_per_row = cols - prune_per_row
        if prune_per_row <= 0 or keep_per_row <= 0:
            continue
        cap = max(1, int(min(prune_per_row, keep_per_row) * 0.8))
        rms = input_stats[f"{name}.weight"]["rms"].to(weight.device)
        base = weight.abs().float() * rms.unsqueeze(0)
        at_risk = torch.topk(base, prune_per_row, dim=1, largest=False).indices
        at_risk_mask = torch.zeros_like(weight, dtype=torch.bool)
        at_risk_mask.scatter_(1, at_risk, True)
        at_risk_masks[name] = at_risk_mask.cpu()
        if group == "C_random":
            score = torch.rand(at_risk.shape, generator=generator, device="cpu").to(weight.device)
        elif group == "B_wanda":
            score = base.gather(1, at_risk)
        else:
            if score_maps is None or name not in score_maps:
                raise RuntimeError(f"Missing score map for {group}:{name}")
            score = score_maps[name].to(weight.device).gather(1, at_risk)
        cap = min(cap, prune_per_row)
        values, local_indices = torch.topk(score, cap, dim=1)
        indices = at_risk.gather(1, local_indices)
        row_ids = torch.arange(weight.shape[0], device=weight.device).unsqueeze(1).expand_as(indices)
        candidates.append((name, values.cpu(), (row_ids * cols + indices).cpu()))
        del base, at_risk, at_risk_mask, score, values, local_indices, indices, row_ids

    if not candidates:
        raise RuntimeError("No protection candidates were generated.")
    flat_values = torch.cat([item[1].flatten() for item in candidates])
    budget = min(budget, flat_values.numel())
    selected = torch.topk(flat_values, budget).indices
    selected_set = torch.zeros(flat_values.numel(), dtype=torch.bool)
    selected_set[selected] = True
    masks = {
        name: torch.zeros_like(modules[name].weight, dtype=torch.bool, device="cpu")
        for name in writer_names
    }
    offset = 0
    for name, values, flat_indices in candidates:
        count = values.numel()
        chosen = selected_set[offset : offset + count]
        masks[name].view(-1)[flat_indices.flatten()[chosen]] = True
        offset += count
    if any(bool((masks[name] & ~at_risk_masks[name]).any()) for name in writer_names):
        raise RuntimeError(f"{group} protection mask escaped the baseline Wanda at-risk set.")
    return masks, {
        "group": group,
        "writer_modules": writer_names,
        "protect_fraction": protect_fraction,
        "protected_weights": sum(int(mask.sum()) for mask in masks.values()),
        "total_writer_weights": total_writer_weights,
        "at_risk_subset_verified": True,
    }


def mask_diagnostics(masks_by_group: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    flattened = {
        group: torch.cat([masks_by_group[group][name].flatten() for name in sorted(masks_by_group[group])])
        for group in PROTECTED_GROUPS
    }
    counts = {group: int(mask.sum()) for group, mask in flattened.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Protection budgets differ: {counts}")
    pairwise = {}
    groups = list(PROTECTED_GROUPS)
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            intersection = int((flattened[left] & flattened[right]).sum())
            union = int((flattened[left] | flattened[right]).sum())
            pairwise[f"{left}_vs_{right}"] = {
                "identical": bool(torch.equal(flattened[left], flattened[right])),
                "jaccard": intersection / max(1, union),
            }
    for group in ("A_feature_grad", "A_loss_grad", "B_wanda"):
        if pairwise[f"{group}_vs_C_random"]["identical"]:
            raise RuntimeError(f"{group} protection mask is identical to random.")
    return {"counts": counts, "pairwise": pairwise, "contrast_verified": True}


def paired_bootstrap(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    keyed: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in rows:
        if row["group"] not in {left, right}:
            continue
        key = (row["seed"], row["sparsity"], row["unit_id"])
        keyed.setdefault(key, {})[row["group"]] = float(row[metric])
    diffs = np.asarray(
        [value[left] - value[right] for value in keyed.values() if left in value and right in value],
        dtype=float,
    )
    if diffs.size == 0:
        return {"n": 0, "mean_difference": None, "ci95": None, "p_two_sided": None}
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        boot[index] = float(rng.choice(diffs, diffs.size, replace=True).mean())
    low, high = np.quantile(boot, [0.025, 0.975])
    p_left = float(np.mean(boot <= 0))
    p_right = float(np.mean(boot >= 0))
    return {
        "n": int(diffs.size),
        "mean_difference": float(diffs.mean()),
        "ci95": [float(low), float(high)],
        "p_two_sided": min(1.0, 2.0 * min(p_left, p_right)),
    }


def summarize_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups = sorted(set(row["group"] for row in rows))
    for group in groups:
        subset = [row for row in rows if row["group"] == group]
        output.append(
            {
                "group": group,
                "n": len(subset),
                "mean_correct": float(np.mean([float(row["correct"]) for row in subset])),
                "mean_margin": float(np.mean([float(row["margin"]) for row in subset])),
            }
        )
    return output


def score_summary(score_maps: dict[str, torch.Tensor]) -> dict[str, Any]:
    values = torch.cat([value.flatten() for value in score_maps.values()]).float()
    return {
        "weights": int(values.numel()),
        "nonzero": int((values > 0).sum()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "q50": float(torch.quantile(values, 0.50)),
        "q90": float(torch.quantile(values, 0.90)),
        "q99": float(torch.quantile(values, 0.99)),
    }


def run_seed(args: argparse.Namespace, seed: int, device: str) -> dict[str, Any]:
    import stage2_gate_causal_v2 as v2

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    v2.set_pad_token(tokenizer)
    calibration_examples, test_examples = split_ability_examples(args)
    calibration_batches = ability_batches(
        tokenizer, calibration_examples, args.batch_examples, args.use_chat_template
    )
    test_batches = ability_batches(
        tokenizer, test_examples, args.batch_examples, args.use_chat_template
    )
    calibration_texts = feature_texts_ability(tokenizer, calibration_examples, args.use_chat_template)
    calibration_blocks = make_blocks(
        tokenizer, calibration_texts, args.calib_seq_len, args.calib_blocks, device
    )
    feature_blocks = make_blocks(
        tokenizer,
        feature_texts_ability(tokenizer, test_examples, args.use_chat_template),
        args.feature_seq_len,
        args.feature_blocks,
        device,
    )
    ppl_blocks = []
    if not args.skip_ppl:
        ppl_texts = get_wikitext_texts(args.ppl_split, args.ppl_num_texts, seed)
        ppl_blocks = make_blocks(tokenizer, ppl_texts, args.ppl_seq_len, args.ppl_blocks, device)

    dense = load_model(args.model_id, device)
    sae, sae_metadata = load_sae_compat(args.sae_release, args.sae_id, device)
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    input_stats = collect_wanda_input_stats(dense, calibration_blocks)
    writer_names = writer_module_names(dense, args.layer, args.writer_scope)
    reference = feature_reference(
        dense,
        tokenizer,
        sae,
        args.layer,
        calibration_batches,
        args.max_length,
        device,
        args.resample_pool_tokens,
        seed,
    )
    ability_causal, ability_meta = attribution_scores(
        dense,
        tokenizer,
        sae,
        args.layer,
        calibration_batches,
        reference,
        args.max_length,
        device,
        "mean",
        seed,
    )
    feature_weights = sharpen_weights(
        ability_causal,
        args.causal_top_fraction,
        args.causal_sharpen_power,
        True,
    )
    feature_scores = crosslayer_feature_grad_scores(
        dense,
        tokenizer,
        sae,
        writer_names,
        args.layer,
        feature_weights,
        calibration_batches,
        args.max_length,
        device,
    )
    loss_scores = crosslayer_loss_grad_scores(
        dense,
        tokenizer,
        writer_names,
        calibration_batches,
        args.max_length,
        device,
    )
    _dense_loss, dense_rows = evaluate_objective(
        dense, tokenizer, test_batches, args.max_length, device
    )
    dense_summary = objective_summary(dense_rows)
    dense_ppl = None if args.skip_ppl else evaluate_ppl(dense, ppl_blocks)["ppl"]
    dense_features = collect_feature_stats(dense, sae, feature_blocks, args.layer)
    del dense
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_rows: list[dict[str, Any]] = []
    ability_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row in dense_rows:
        ability_rows.append(
            {
                "seed": seed,
                "sparsity": 0.0,
                "group": "dense",
                "unit_id": row["unit_id"],
                "correct": row["correct"],
                "margin": row["margin"],
            }
        )
    model_rows.append(
        {
            "seed": seed,
            "sparsity": 0.0,
            "group": "dense",
            "actual_sparsity": 0.0,
            "ability_accuracy": dense_summary["accuracy"],
            "ability_margin": dense_summary["mean_margin"],
            "ppl": dense_ppl,
            "feature_l0": dense_features["l0"],
            "feature_decoded_activation_cosine": dense_features["decoded_activation_cosine"],
        }
    )

    score_by_group = {
        "A_feature_grad": feature_scores,
        "A_loss_grad": loss_scores,
        "B_wanda": None,
        "C_random": None,
    }
    for sparsity in args.sparsities:
        masks_by_group = {}
        protection_by_group = {}
        probe = load_model(args.model_id, device)
        for group in PROTECTED_GROUPS:
            masks, protection = protection_masks_from_scores(
                probe,
                input_stats,
                writer_names,
                score_by_group[group],
                sparsity,
                args.protect_fraction,
                group,
                seed * 1000003 + int(sparsity * 1000) * 17 + list(PROTECTED_GROUPS).index(group),
            )
            masks_by_group[group] = {name: mask.clone() for name, mask in masks.items()}
            protection_by_group[group] = protection
        del probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        diagnostics.append({"seed": seed, "sparsity": sparsity, **mask_diagnostics(masks_by_group)})

        for group in ALL_GROUPS:
            model = load_model(args.model_id, device)
            if group == "D_wanda_no_protection":
                masks = {}
                protection = {
                    "group": group,
                    "protected_weights": 0,
                    "total_writer_weights": sum(
                        dict(model.named_modules())[name].weight.numel() for name in writer_names
                    ),
                    "at_risk_subset_verified": True,
                }
            else:
                masks = masks_by_group[group]
                protection = protection_by_group[group]
            pruning = apply_protected_wanda(
                model,
                input_stats,
                sparsity,
                masks,
                "whole",
                writer_names,
            )
            loss, rows = evaluate_objective(model, tokenizer, test_batches, args.max_length, device)
            summary = objective_summary(rows)
            ppl = None if args.skip_ppl else evaluate_ppl(model, ppl_blocks)["ppl"]
            features = collect_feature_stats(model, sae, feature_blocks, args.layer)
            for row in rows:
                ability_rows.append(
                    {
                        "seed": seed,
                        "sparsity": sparsity,
                        "group": group,
                        "unit_id": row["unit_id"],
                        "correct": row["correct"],
                        "margin": row["margin"],
                    }
                )
            model_rows.append(
                {
                    "seed": seed,
                    "sparsity": sparsity,
                    "group": group,
                    "actual_sparsity": pruning["actual_sparsity"],
                    "protected_weights": protection["protected_weights"],
                    "rescued_weights": pruning["rescued_weights"],
                    "protected_pruned_overlap": pruning["protected_pruned_overlap"],
                    "ability_loss": loss,
                    "ability_accuracy": summary["accuracy"],
                    "ability_margin": summary["mean_margin"],
                    "ability_accuracy_delta": summary["accuracy"] - dense_summary["accuracy"],
                    "ability_margin_delta": summary["mean_margin"] - dense_summary["mean_margin"],
                    "ppl": ppl,
                    "ppl_relative_increase": None if ppl is None else ppl / dense_ppl - 1.0,
                    "feature_l0": features["l0"],
                    "feature_l0_delta": features["l0"] - dense_features["l0"],
                    "feature_decoded_activation_cosine": features["decoded_activation_cosine"],
                    "feature_decoded_cosine_delta": (
                        features["decoded_activation_cosine"]
                        - dense_features["decoded_activation_cosine"]
                    ),
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "seed": seed,
        "sae_metadata": sae_metadata,
        "writer_names": writer_names,
        "ability_meta": ability_meta,
        "feature_score_summary": score_summary(feature_scores),
        "loss_score_summary": score_summary(loss_scores),
        "model_rows": model_rows,
        "ability_rows": ability_rows,
        "diagnostics": diagnostics,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    ability_rows = _read_csv(args.output_root / "ability_rows.csv")
    model_rows = _read_csv(args.output_root / "models.csv")
    tests = {
        "feature_vs_wanda_correct": paired_bootstrap(
            ability_rows,
            "A_feature_grad",
            "B_wanda",
            "correct",
            args.bootstrap_samples,
            args.split_seed + 1,
        ),
        "feature_vs_random_correct": paired_bootstrap(
            ability_rows,
            "A_feature_grad",
            "C_random",
            "correct",
            args.bootstrap_samples,
            args.split_seed + 2,
        ),
        "feature_vs_no_protection_correct": paired_bootstrap(
            ability_rows,
            "A_feature_grad",
            "D_wanda_no_protection",
            "correct",
            args.bootstrap_samples,
            args.split_seed + 3,
        ),
        "loss_vs_wanda_correct": paired_bootstrap(
            ability_rows,
            "A_loss_grad",
            "B_wanda",
            "correct",
            args.bootstrap_samples,
            args.split_seed + 4,
        ),
    }
    group_summary = summarize_group_rows(ability_rows)
    feature_beats_wanda = (
        tests["feature_vs_wanda_correct"]["mean_difference"] is not None
        and tests["feature_vs_wanda_correct"]["mean_difference"] > 0
    )
    feature_beats_random = (
        tests["feature_vs_random_correct"]["mean_difference"] is not None
        and tests["feature_vs_random_correct"]["mean_difference"] > 0
    )
    gate_status = "W1_PASS_CANDIDATE" if feature_beats_wanda and feature_beats_random else "W1_INCONCLUSIVE"
    result = {
        "step": "stage2_w1_analyze",
        "status": "PASS",
        "gate_status": gate_status,
        "config": vars(args),
        "group_summary": group_summary,
        "paired_tests": tests,
        "model_rows": model_rows,
        "conclusion": (
            "Feature-fidelity cross-layer protection improved held-out ability over Wanda-geometry and random controls."
            if gate_status == "W1_PASS_CANDIDATE"
            else "W1 did not show a clear feature-fidelity advantage over both Wanda-geometry and random controls."
        ),
    }
    write_json(args.final_json, result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    if args.step == "analyze":
        analysis = analyze(args)
        _log(
            args,
            "analyze",
            started,
            analysis["gate_status"],
            {
                "final_json": str(args.final_json),
                "paired_tests": analysis["paired_tests"],
                "group_summary": analysis["group_summary"],
            },
            analysis["conclusion"],
        )
        return analysis
    fingerprint_path = args.output_root / "config_fingerprint.txt"
    fingerprint = _config_fingerprint(args)
    if fingerprint_path.exists() and fingerprint_path.read_text(encoding="utf-8") != fingerprint:
        raise RuntimeError(
            f"Output root {args.output_root} has a different config fingerprint. Use a new output root."
        )
    fingerprint_path.write_text(fingerprint, encoding="utf-8")
    all_model_rows: list[dict[str, Any]] = []
    all_ability_rows: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    seed_summaries = []
    for seed in args.seeds:
        seed_model_path = args.output_root / f"seed{seed}_models.csv"
        seed_ability_path = args.output_root / f"seed{seed}_ability_rows.csv"
        seed_diag_path = args.output_root / f"seed{seed}_diagnostics.json"
        if seed_model_path.exists() and seed_ability_path.exists() and seed_diag_path.exists():
            all_model_rows.extend(_read_csv(seed_model_path))
            all_ability_rows.extend(_read_csv(seed_ability_path))
            diag = json.loads(seed_diag_path.read_text(encoding="utf-8"))
            all_diagnostics.extend(diag["diagnostics"])
            seed_summaries.append({**diag["summary"], "reused": True})
            continue
        seed_result = run_seed(args, seed, args.device)
        _write_csv(seed_model_path, seed_result["model_rows"])
        _write_csv(seed_ability_path, seed_result["ability_rows"])
        seed_diag = {
            "seed": seed,
            "summary": {
                "seed": seed,
                "writer_modules": len(seed_result["writer_names"]),
                "ability_examples": seed_result["ability_meta"]["examples"],
                "feature_score_summary": seed_result["feature_score_summary"],
                "loss_score_summary": seed_result["loss_score_summary"],
            },
            "diagnostics": seed_result["diagnostics"],
        }
        write_json(seed_diag_path, seed_diag)
        all_model_rows.extend(seed_result["model_rows"])
        all_ability_rows.extend(seed_result["ability_rows"])
        all_diagnostics.extend(seed_result["diagnostics"])
        seed_summaries.append(seed_diag["summary"])
    _write_csv(args.output_root / "models.csv", all_model_rows)
    _write_csv(args.output_root / "ability_rows.csv", all_ability_rows)
    write_json(args.output_root / "mask_diagnostics.json", {"diagnostics": all_diagnostics})
    log = _log(
        args,
        "run",
        started,
        "PASS",
        {
            "output_root": str(args.output_root),
            "models_csv": str(args.output_root / "models.csv"),
            "ability_rows_csv": str(args.output_root / "ability_rows.csv"),
            "seeds": seed_summaries,
        },
        "W1 cross-layer ability intervention completed.",
    )
    if args.step in {"all", "analyze"}:
        analysis = analyze(args)
        _log(
            args,
            "analyze",
            started,
            analysis["gate_status"],
            {
                "final_json": str(args.final_json),
                "paired_tests": analysis["paired_tests"],
                "group_summary": analysis["group_summary"],
            },
            analysis["conclusion"],
        )
        return analysis
    return log


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Stage 2 W1 ability cross-layer importance gate")
    result.add_argument("--step", choices=("run", "analyze", "all"), default="all")
    result.add_argument("--model-id", default="google/gemma-2-2b")
    result.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    result.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    result.add_argument("--layer", type=int, default=12)
    result.add_argument("--writer-scope", choices=("single", "upstream", "all"), default="upstream")
    result.add_argument("--tasks", default="arc_easy,hellaswag")
    result.add_argument("--task-split", default="validation")
    result.add_argument("--ability-calibration-per-task", type=int, default=128)
    result.add_argument("--ability-test-per-task", type=int, default=128)
    result.add_argument("--split-seed", type=int, default=20260621)
    result.add_argument("--seeds", default="0,1,2")
    result.add_argument("--sparsities", default="0.40,0.50,0.60")
    result.add_argument("--protect-fraction", type=float, default=0.02)
    result.add_argument("--batch-examples", type=int, default=2)
    result.add_argument("--max-length", type=int, default=256)
    result.add_argument("--calib-seq-len", type=int, default=128)
    result.add_argument("--calib-blocks", type=int, default=32)
    result.add_argument("--feature-seq-len", type=int, default=128)
    result.add_argument("--feature-blocks", type=int, default=16)
    result.add_argument("--resample-pool-tokens", type=int, default=4096)
    result.add_argument("--causal-top-fraction", type=float, default=0.05)
    result.add_argument("--causal-sharpen-power", type=float, default=2.0)
    result.add_argument("--ppl-split", default="test")
    result.add_argument("--ppl-num-texts", type=int, default=96)
    result.add_argument("--ppl-seq-len", type=int, default=256)
    result.add_argument("--ppl-blocks", type=int, default=16)
    result.add_argument("--skip-ppl", action="store_true")
    result.add_argument("--bootstrap-samples", type=int, default=5000)
    result.add_argument("--use-chat-template", action="store_true")
    result.add_argument("--output-root", type=Path, default=Path("results/stage2_w1_ability"))
    result.add_argument("--log-root", type=Path, default=Path("logs"))
    result.add_argument("--final-json", type=Path, default=Path("results/stage2_w1_ability.json"))
    result.add_argument("--device", default="cuda")
    result.add_argument("--smoke", action="store_true")
    return result


def config_from_args(args: argparse.Namespace) -> argparse.Namespace:
    args.seeds = _ints(args.seeds)
    args.sparsities = _floats(args.sparsities)
    if args.smoke:
        args.ability_calibration_per_task = min(args.ability_calibration_per_task, 8)
        args.ability_test_per_task = min(args.ability_test_per_task, 8)
        args.seeds = (0,)
        args.sparsities = (0.5,)
        args.calib_blocks = 2
        args.feature_blocks = 2
        args.ppl_blocks = 2
        args.bootstrap_samples = 500
        args.output_root = Path("results/stage2_w1_ability_smoke")
        args.final_json = Path("results/stage2_w1_ability_smoke.json")
        args.skip_ppl = True
    if not 0 < args.protect_fraction < 1:
        raise ValueError("--protect-fraction must be in (0, 1).")
    if any(not 0 < value < 1 for value in args.sparsities):
        raise ValueError("--sparsities must all be in (0, 1).")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("W1 requires CUDA.")
    return args


def main() -> int:
    args = config_from_args(parser().parse_args())
    result = run(args)
    print(f"status: {result.get('gate_status', result.get('status'))}")
    print(result.get("conclusion", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
