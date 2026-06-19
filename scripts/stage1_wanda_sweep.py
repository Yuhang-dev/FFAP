from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ffap.json_utils import write_json
from stage1_magnitude_sweep import (
    load_model,
    parse_sparsities,
    save_pruned_checkpoint,
)
from stage1_smoke import (
    collect_feature_stats,
    compare_feature_stats,
    evaluate_ppl,
    get_wikitext_texts,
    iter_prunable_linear_weights,
    load_sae_compat,
    make_blocks,
    strip_feature_vectors,
    write_stage1_csv,
)


def prunable_linear_modules(model: Any) -> dict[str, torch.nn.Linear]:
    modules = {}
    for name, module in model.named_modules():
        if name == "lm_head":
            continue
        if isinstance(module, torch.nn.Linear):
            modules[name] = module
    return modules


@torch.no_grad()
def collect_wanda_input_stats(
    model: Any,
    blocks: list[torch.Tensor],
) -> dict[str, dict[str, Any]]:
    modules = prunable_linear_modules(model)
    stats: dict[str, dict[str, Any]] = {
        name: {
            "sum_sq": torch.zeros(module.in_features, dtype=torch.float64),
            "count": 0,
        }
        for name, module in modules.items()
    }
    handles = []

    def make_hook(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs:
                return
            x = inputs[0].detach()
            flat = x.reshape(-1, x.shape[-1]).float().cpu()
            stats[name]["sum_sq"] += (flat * flat).sum(dim=0, dtype=torch.float64)
            stats[name]["count"] += flat.shape[0]

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        for block in tqdm(blocks, desc="Wanda calibration blocks"):
            _ = model(input_ids=block)
    finally:
        for handle in handles:
            handle.remove()

    output = {}
    for name, item in stats.items():
        count = max(1, int(item["count"]))
        rms = torch.sqrt(item["sum_sq"] / count).to(torch.float32)
        output[f"{name}.weight"] = {
            "rms": rms,
            "count": count,
            "mean_rms": float(rms.mean().item()),
            "max_rms": float(rms.max().item()),
        }
    return output


@torch.no_grad()
def apply_wanda_pruning(
    model: Any,
    input_stats: dict[str, dict[str, Any]],
    sparsity: float,
) -> dict[str, Any]:
    total = 0
    pruned = 0
    layer_summaries = []
    for name, weight in tqdm(iter_prunable_linear_weights(model), desc="Wanda prune"):
        if name not in input_stats:
            raise KeyError(f"Missing Wanda input stats for {name}")
        numel = weight.numel()
        cols = weight.shape[1]
        k = int(cols * sparsity)
        if k <= 0:
            continue
        rms = input_stats[name]["rms"].to(weight.device)
        metric = weight.detach().abs().float() * rms.unsqueeze(0)
        threshold = torch.kthvalue(metric, k, dim=1).values.unsqueeze(1)
        mask = metric <= threshold
        pruned_this = int(mask.sum().item())
        weight[mask] = 0
        total += numel
        pruned += pruned_this
        layer_summaries.append(
            {
                "name": name,
                "numel": numel,
                "pruned": pruned_this,
                "sparsity": pruned_this / numel,
                "mean_threshold": float(threshold.mean().detach().cpu()),
                "mean_input_rms": input_stats[name]["mean_rms"],
                "max_input_rms": input_stats[name]["max_rms"],
            }
        )
        del metric, threshold, mask
    return {
        "method": "wanda_unstructured",
        "target_sparsity": sparsity,
        "total_considered": total,
        "total_pruned": pruned,
        "actual_sparsity": pruned / total if total else 0.0,
        "layers": layer_summaries,
    }


def summarize_pruning(pruning: dict[str, Any], keep_layer_details: bool) -> dict[str, Any]:
    if keep_layer_details:
        return pruning
    return {key: value for key, value in pruning.items() if key != "layers"}


def summarize_input_stats(input_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mean_rms_values = [item["mean_rms"] for item in input_stats.values()]
    counts = [item["count"] for item in input_stats.values()]
    return {
        "layers": len(input_stats),
        "min_count": min(counts) if counts else None,
        "max_count": max(counts) if counts else None,
        "mean_input_rms_min": min(mean_rms_values) if mean_rms_values else None,
        "mean_input_rms_max": max(mean_rms_values) if mean_rms_values else None,
        "mean_input_rms_avg": (
            sum(mean_rms_values) / len(mean_rms_values) if mean_rms_values else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 Wanda pruning sweep")
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--num-eval-texts", type=int, default=96)
    parser.add_argument("--num-calib-texts", type=int, default=128)
    parser.add_argument("--ppl-seq-len", type=int, default=256)
    parser.add_argument("--ppl-blocks", type=int, default=32)
    parser.add_argument("--feature-seq-len", type=int, default=128)
    parser.add_argument("--feature-blocks", type=int, default=16)
    parser.add_argument("--calib-seq-len", type=int, default=128)
    parser.add_argument("--calib-blocks", type=int, default=16)
    parser.add_argument("--sparsities", default="0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-root", default="outputs/stage1_wanda_sweep")
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-layer-details", action="store_true")
    parser.add_argument("--out-json", default="logs/stage1_wanda_sweep.json")
    parser.add_argument("--out-csv", default="results/stage1_wanda_sweep.csv")
    args = parser.parse_args()

    started = time.time()
    sparsities = parse_sparsities(args.sparsities)
    payload: dict[str, Any] = {
        "task": "stage1_wanda_sweep",
        "timestamp_unix": started,
        "config": {**vars(args), "sparsities": sparsities},
        "status": "STARTED",
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Stage 1 Wanda sweep.")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = "cuda:0"
        torch.cuda.reset_peak_memory_stats()

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        eval_texts = get_wikitext_texts(args.eval_split, args.num_eval_texts, args.seed)
        calib_texts = get_wikitext_texts(
            args.calib_split, args.num_calib_texts, args.seed
        )
        ppl_blocks = make_blocks(
            tokenizer, eval_texts, args.ppl_seq_len, args.ppl_blocks, device
        )
        feature_blocks = make_blocks(
            tokenizer, eval_texts, args.feature_seq_len, args.feature_blocks, device
        )
        calib_blocks = make_blocks(
            tokenizer, calib_texts, args.calib_seq_len, args.calib_blocks, device
        )
        sae, sae_metadata = load_sae_compat(args.sae_release, args.sae_id, device)

        model = load_model(args.model_id, device)
        input_stats = collect_wanda_input_stats(model, calib_blocks)
        dense_ppl = evaluate_ppl(model, ppl_blocks)
        dense_features = collect_feature_stats(model, sae, feature_blocks, args.layer)
        del model
        torch.cuda.empty_cache()

        rows: list[dict[str, Any]] = [
            {
                "model_id": args.model_id,
                "method": "dense",
                "sparsity": 0.0,
                "ppl": dense_ppl["ppl"],
                "nll": dense_ppl["nll"],
                "ppl_tokens": dense_ppl["tokens"],
                "feature_l0": dense_features["l0"],
                "feature_reconstruction_mse": dense_features["reconstruction_mse"],
                "feature_decoded_activation_cosine": dense_features[
                    "decoded_activation_cosine"
                ],
                "active_features_count": dense_features["active_features_count"],
            }
        ]
        runs = []

        for sparsity in sparsities:
            run_started = time.time()
            model = load_model(args.model_id, device)
            pruning = apply_wanda_pruning(model, input_stats, sparsity)
            saved = save_pruned_checkpoint(
                model,
                tokenizer,
                args.checkpoint_root,
                pruning["method"],
                sparsity,
                args.save_checkpoints,
            )
            pruned_ppl = evaluate_ppl(model, ppl_blocks)
            pruned_features = collect_feature_stats(model, sae, feature_blocks, args.layer)
            feature_damage = compare_feature_stats(dense_features, pruned_features)
            row = {
                "model_id": args.model_id,
                "method": pruning["method"],
                "sparsity": pruning["actual_sparsity"],
                "target_sparsity": sparsity,
                "ppl": pruned_ppl["ppl"],
                "nll": pruned_ppl["nll"],
                "ppl_tokens": pruned_ppl["tokens"],
                "ppl_relative_increase": pruned_ppl["ppl"] / dense_ppl["ppl"] - 1.0,
                "feature_l0": pruned_features["l0"],
                "feature_reconstruction_mse": pruned_features["reconstruction_mse"],
                "feature_decoded_activation_cosine": pruned_features[
                    "decoded_activation_cosine"
                ],
                "active_features_count": pruned_features["active_features_count"],
                **feature_damage,
            }
            rows.append(row)
            runs.append(
                {
                    "target_sparsity": sparsity,
                    "elapsed_sec": round(time.time() - run_started, 3),
                    "pruning": summarize_pruning(pruning, args.keep_layer_details),
                    "saved_pruned": saved,
                    "ppl": pruned_ppl,
                    "features": strip_feature_vectors(pruned_features),
                    "feature_damage": feature_damage,
                    "ppl_relative_increase": row["ppl_relative_increase"],
                }
            )
            del model
            torch.cuda.empty_cache()
            write_stage1_csv(Path(args.out_csv), rows)
            write_json(args.out_json, {**payload, "status": "RUNNING", "runs": runs})

        payload.update(
            {
                "status": "PASS",
                "elapsed_sec": round(time.time() - started, 3),
                "torch": {
                    "version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0),
                    "peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated() / 1024**3, 3
                    ),
                    "peak_reserved_gib": round(
                        torch.cuda.max_memory_reserved() / 1024**3, 3
                    ),
                },
                "sae": sae_metadata,
                "wanda_input_stats": summarize_input_stats(input_stats),
                "dense": {
                    "ppl": dense_ppl,
                    "features": strip_feature_vectors(dense_features),
                },
                "runs": runs,
                "outputs": {"json": args.out_json, "csv": args.out_csv},
                "conclusion": "Stage 1 Wanda sweep completed.",
            }
        )
        write_stage1_csv(Path(args.out_csv), rows)
        write_json(args.out_json, payload)
        print(f"wrote {args.out_json}")
        print(f"wrote {args.out_csv}")
        print("status: PASS")
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "FAIL",
                "elapsed_sec": round(time.time() - started, 3),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "conclusion": "Stage 1 Wanda sweep failed; inspect error and logs.",
            }
        )
        write_json(args.out_json, payload)
        print(f"wrote {args.out_json}")
        print(f"status: FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

