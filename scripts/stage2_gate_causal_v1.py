from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from ffap.json_utils import write_json
from stage1_smoke import (
    collect_feature_stats,
    evaluate_ppl,
    get_wikitext_texts,
    load_sae_compat,
    make_blocks,
    strip_feature_vectors,
)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


def load_model(model_ref: str, device: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def checkpoint_path(method: str, sparsity: float) -> str:
    label = f"{method}_s{sparsity:.2f}"
    if method == "local_magnitude_unstructured":
        return f"outputs/stage1_magnitude_sweep/{label}"
    if method == "wanda_unstructured":
        return f"outputs/stage1_wanda_sweep/{label}"
    raise ValueError(f"Unsupported method for checkpoint inference: {method}")


def load_sweep_records(paths: list[str], max_sparsity: float | None = None) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        for row in read_rows(path):
            method = row.get("method", "")
            if method == "dense":
                continue
            target_sparsity = as_float(row.get("target_sparsity"))
            if target_sparsity is None:
                target_sparsity = as_float(row.get("sparsity"), 0.0)
            assert target_sparsity is not None
            if max_sparsity is not None and target_sparsity > max_sparsity:
                continue
            records.append(
                {
                    "method": method,
                    "target_sparsity": target_sparsity,
                    "actual_sparsity": as_float(row.get("sparsity")),
                    "ppl": as_float(row.get("ppl")),
                    "ppl_relative_increase": as_float(
                        row.get("ppl_relative_increase"), 0.0
                    ),
                    "checkpoint": checkpoint_path(method, target_sparsity),
                }
            )
    return records


def load_capability_losses(paths: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = []
    for path in paths:
        rows.extend(read_rows(path))
    dense_values = [
        as_float(row.get("acc_norm"))
        for row in rows
        if row.get("method") == "dense" and row.get("acc_norm") != ""
    ]
    dense_mean = sum(v for v in dense_values if v is not None) / max(1, len(dense_values))
    by_label: dict[str, list[float]] = {}
    by_ref: dict[str, list[float]] = {}
    for row in rows:
        if row.get("method") == "dense":
            continue
        value = as_float(row.get("acc_norm"))
        if value is None:
            continue
        by_label.setdefault(row["label"], []).append(value)
        by_ref.setdefault(row["model_ref"], []).append(value)

    losses = {}
    for key, values in {**by_label, **by_ref}.items():
        mean_acc_norm = sum(values) / len(values)
        losses[key] = {
            "mean_acc_norm": mean_acc_norm,
            "ability_loss": dense_mean - mean_acc_norm,
            "ability_retention": mean_acc_norm / dense_mean if dense_mean else None,
            "dense_mean_acc_norm": dense_mean,
            "n_task_values": len(values),
        }
    return {"dense_mean_acc_norm": dense_mean, "n_dense_values": len(dense_values)}, losses


def label_for_record(record: dict[str, Any]) -> str:
    return f"{record['method']}_s{record['target_sparsity']:.2f}"


@torch.no_grad()
def nll_for_block(model: Any, block: torch.Tensor) -> tuple[float, int]:
    outputs = model(input_ids=block, labels=block)
    tokens = max(1, block.numel() - 1)
    return float(outputs.loss.detach().cpu()) * tokens, tokens


def ablated_block_nll(
    model: Any,
    sae: Any,
    layer_index: int,
    block: torch.Tensor,
    feature_id: int,
) -> tuple[float, int]:
    layer = model.model.layers[layer_index]

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

    handle = layer.register_forward_hook(hook)
    try:
        return nll_for_block(model, block)
    finally:
        handle.remove()


def compute_causal_scores(
    model: Any,
    sae: Any,
    layer: int,
    blocks: list[torch.Tensor],
    feature_ids: list[int],
) -> dict[int, float]:
    baseline_loss = 0.0
    baseline_tokens = 0
    with torch.no_grad():
        for block in blocks:
            loss, tokens = nll_for_block(model, block)
            baseline_loss += loss
            baseline_tokens += tokens
    baseline_nll = baseline_loss / max(1, baseline_tokens)

    scores: dict[int, float] = {}
    for feature_id in feature_ids:
        total_loss = 0.0
        total_tokens = 0
        for block in blocks:
            loss, tokens = ablated_block_nll(model, sae, layer, block, feature_id)
            total_loss += loss
            total_tokens += tokens
        ablated_nll = total_loss / max(1, total_tokens)
        scores[feature_id] = ablated_nll - baseline_nll
    return scores


def select_features(
    dense_features: dict[str, Any],
    top_k: int,
    strategy: str,
) -> list[int]:
    if strategy == "firing_rate":
        scores = dense_features["firing_rate"]
    elif strategy == "mean_activation":
        scores = dense_features["mean_activation"]
    elif strategy == "activity_mass":
        scores = dense_features["firing_rate"] * dense_features["mean_activation"]
    else:
        raise ValueError(f"Unknown feature selection strategy: {strategy}")
    k = min(top_k, int(scores.numel()))
    return [int(index) for index in torch.topk(scores, k=k).indices.tolist()]


def feature_damage_for_selected(
    dense_features: dict[str, Any],
    pruned_features: dict[str, Any],
    feature_ids: list[int],
    causal_scores: dict[int, float],
) -> dict[str, float]:
    ids = torch.tensor(feature_ids, dtype=torch.long)
    dense_mean = dense_features["mean_activation"][ids]
    pruned_mean = pruned_features["mean_activation"][ids]
    dense_rate = dense_features["firing_rate"][ids]
    pruned_rate = pruned_features["firing_rate"][ids]
    mean_shift = torch.abs(dense_mean - pruned_mean)
    rate_shift = torch.abs(dense_rate - pruned_rate)

    weights = torch.tensor(
        [max(0.0, causal_scores[int(feature_id)]) for feature_id in feature_ids],
        dtype=torch.float32,
    )
    abs_weights = torch.tensor(
        [abs(causal_scores[int(feature_id)]) for feature_id in feature_ids],
        dtype=torch.float32,
    )
    eps = 1e-12
    return {
        "selected_mean_activation_l1": float(mean_shift.mean().item()),
        "selected_firing_rate_l1": float(rate_shift.mean().item()),
        "causal_weighted_mean_activation_l1": float(
            (mean_shift * weights).sum().item() / (weights.sum().item() + eps)
        ),
        "abs_causal_weighted_mean_activation_l1": float(
            (mean_shift * abs_weights).sum().item() / (abs_weights.sum().item() + eps)
        ),
        "causal_weighted_firing_rate_l1": float(
            (rate_shift * weights).sum().item() / (weights.sum().item() + eps)
        ),
        "abs_causal_weighted_firing_rate_l1": float(
            (rate_shift * abs_weights).sum().item() / (abs_weights.sum().item() + eps)
        ),
        "positive_causal_weight_sum": float(weights.sum().item()),
        "abs_causal_weight_sum": float(abs_weights.sum().item()),
    }


def correlation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = [row["ability_loss"] for row in rows]
    predictors = [
        "ppl_relative_increase",
        "selected_mean_activation_l1",
        "selected_firing_rate_l1",
        "causal_weighted_mean_activation_l1",
        "abs_causal_weighted_mean_activation_l1",
        "causal_weighted_firing_rate_l1",
        "abs_causal_weighted_firing_rate_l1",
    ]
    out = {}
    for predictor in predictors:
        x = [row[predictor] for row in rows]
        if len(set(x)) <= 1 or len(set(y)) <= 1:
            out[predictor] = {"spearman_r": None, "p_value": None}
            continue
        result = spearmanr(x, y)
        rho = float(result.statistic) if not math.isnan(result.statistic) else None
        p_value = float(result.pvalue) if not math.isnan(result.pvalue) else None
        out[predictor] = {"spearman_r": rho, "p_value": p_value}
    return out


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


def gate_decision(correlations: dict[str, Any]) -> dict[str, str]:
    causal_predictors = [
        "causal_weighted_mean_activation_l1",
        "abs_causal_weighted_mean_activation_l1",
        "causal_weighted_firing_rate_l1",
        "abs_causal_weighted_firing_rate_l1",
    ]
    geometry_predictors = [
        "selected_mean_activation_l1",
        "selected_firing_rate_l1",
    ]

    def best_of(predictors: list[str]) -> tuple[str | None, float | None]:
        valid = [
            (name, correlations[name]["spearman_r"])
            for name in predictors
            if correlations[name]["spearman_r"] is not None
        ]
        if not valid:
            return None, None
        return max(valid, key=lambda item: item[1])

    best_causal_name, best_causal = best_of(causal_predictors)
    best_geometry_name, best_geometry = best_of(geometry_predictors)
    ppl = correlations["ppl_relative_increase"]["spearman_r"]

    if best_causal is None or best_geometry is None or ppl is None:
        status = "INCONCLUSIVE"
        reason = "Insufficient variation for Spearman comparison."
    elif best_causal > best_geometry and best_causal > ppl:
        status = "PASS_CANDIDATE"
        reason = (
            f"Best causal-weighted metric ({best_causal_name}, rho={best_causal:.3f}) "
            f"exceeds best geometry metric ({best_geometry_name}, rho={best_geometry:.3f}) "
            f"and PPL relative increase (rho={ppl:.3f})."
        )
    else:
        status = "FAIL_OR_REVISE_CANDIDATE"
        reason = (
            f"Best causal-weighted metric ({best_causal_name}, rho={best_causal:.3f}) "
            f"does not exceed both best geometry metric ({best_geometry_name}, "
            f"rho={best_geometry:.3f}) and PPL relative increase (rho={ppl:.3f})."
        )
    return {
        "gate_status": status,
        "reason": reason,
        "best_causal_metric": best_causal_name,
        "best_causal_spearman_r": best_causal,
        "best_geometry_metric": best_geometry_name,
        "best_geometry_spearman_r": best_geometry,
        "ppl_spearman_r": ppl,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 causal feature gate")
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--num-texts", type=int, default=96)
    parser.add_argument("--feature-seq-len", type=int, default=128)
    parser.add_argument("--feature-blocks", type=int, default=16)
    parser.add_argument("--causal-seq-len", type=int, default=64)
    parser.add_argument("--causal-blocks", type=int, default=4)
    parser.add_argument("--top-k-features", type=int, default=32)
    parser.add_argument(
        "--feature-selection",
        choices=["firing_rate", "mean_activation", "activity_mass"],
        default="activity_mass",
    )
    parser.add_argument(
        "--sweep-csv",
        action="append",
        default=None,
        help="Repeatable. Defaults to the Stage 1 magnitude and Wanda sweep CSVs.",
    )
    parser.add_argument(
        "--capability-csv",
        action="append",
        default=None,
        help="Repeatable. Defaults to the Stage 1 magnitude and Wanda capability CSVs.",
    )
    parser.add_argument(
        "--max-sparsity",
        type=float,
        default=None,
        help="Optional robustness filter; excludes sweep rows above this target sparsity.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="results/stage2_gate.json")
    parser.add_argument("--out-csv", default="results/stage2_gate.csv")
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
        "task": "stage2_gate_causal",
        "timestamp_unix": started,
        "config": vars(args),
        "status": "STARTED",
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Stage 2 gate.")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = "cuda:0"
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        texts = get_wikitext_texts(args.dataset_split, args.num_texts, args.seed)
        feature_blocks = make_blocks(
            tokenizer, texts, args.feature_seq_len, args.feature_blocks, device
        )
        causal_blocks = make_blocks(
            tokenizer, texts, args.causal_seq_len, args.causal_blocks, device
        )
        sae, sae_metadata = load_sae_compat(args.sae_release, args.sae_id, device)

        dense_model = load_model(args.model_id, device)
        dense_features = collect_feature_stats(dense_model, sae, feature_blocks, args.layer)
        feature_ids = select_features(
            dense_features, args.top_k_features, args.feature_selection
        )
        causal_scores = compute_causal_scores(
            dense_model, sae, args.layer, causal_blocks, feature_ids
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
            row = {
                **record,
                **ability,
                **damage,
            }
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
                "selected_features": feature_ids,
                "causal_scores": {str(k): v for k, v in causal_scores.items()},
                "causal_score_summary": causal_score_summary,
                "rows": rows,
                "correlations": correlations,
                **decision,
                "outputs": {"json": args.out_json, "csv": args.out_csv},
                "conclusion": (
                    f"Stage 2 gate candidate: {decision['gate_status']}. "
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
                "conclusion": "Stage 2 causal gate failed; inspect logs.",
            }
        )
        write_json(args.out_json, payload)
        print(f"wrote {args.out_json}")
        print(f"status: FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
