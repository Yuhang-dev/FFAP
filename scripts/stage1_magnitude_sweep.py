from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ffap.json_utils import write_json
from stage1_smoke import (
    apply_local_magnitude_pruning,
    collect_feature_stats,
    compare_feature_stats,
    evaluate_ppl,
    get_wikitext_texts,
    load_sae_compat,
    make_blocks,
    strip_feature_vectors,
    write_stage1_csv,
)


def parse_sparsities(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0 or value >= 1:
            raise ValueError(f"Sparsity must be in (0, 1), got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one sparsity is required.")
    return values


def load_model(model_id: str, device: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def checkpoint_dir(root: str, method: str, sparsity: float) -> Path:
    return Path(root) / f"{method}_s{sparsity:.2f}"


def save_pruned_checkpoint(
    model: Any,
    tokenizer: Any,
    root: str,
    method: str,
    sparsity: float,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    save_dir = checkpoint_dir(root, method, sparsity)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    return {
        "enabled": True,
        "path": str(save_dir),
        "files": [
            str(path.relative_to(save_dir))
            for path in sorted(save_dir.rglob("*"))
            if path.is_file()
        ],
    }


def summarize_pruning(pruning: dict[str, Any], keep_layer_details: bool) -> dict[str, Any]:
    if keep_layer_details:
        return pruning
    return {key: value for key, value in pruning.items() if key != "layers"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 magnitude pruning sweep")
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--num-texts", type=int, default=96)
    parser.add_argument("--ppl-seq-len", type=int, default=256)
    parser.add_argument("--ppl-blocks", type=int, default=32)
    parser.add_argument("--feature-seq-len", type=int, default=128)
    parser.add_argument("--feature-blocks", type=int, default=16)
    parser.add_argument("--sparsities", default="0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-root", default="outputs/stage1_magnitude_sweep")
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-layer-details", action="store_true")
    parser.add_argument("--out-json", default="logs/stage1_magnitude_sweep.json")
    parser.add_argument("--out-csv", default="results/stage1_magnitude_sweep.csv")
    args = parser.parse_args()

    started = time.time()
    sparsities = parse_sparsities(args.sparsities)
    payload: dict[str, Any] = {
        "task": "stage1_magnitude_sweep",
        "timestamp_unix": started,
        "config": {**vars(args), "sparsities": sparsities},
        "status": "STARTED",
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Stage 1 magnitude sweep.")

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = "cuda:0"
        torch.cuda.reset_peak_memory_stats()

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        texts = get_wikitext_texts(args.dataset_split, args.num_texts, args.seed)
        ppl_blocks = make_blocks(
            tokenizer, texts, args.ppl_seq_len, args.ppl_blocks, device
        )
        feature_blocks = make_blocks(
            tokenizer, texts, args.feature_seq_len, args.feature_blocks, device
        )
        sae, sae_metadata = load_sae_compat(args.sae_release, args.sae_id, device)

        model = load_model(args.model_id, device)
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
            pruning = apply_local_magnitude_pruning(model, sparsity)
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
                "dense": {
                    "ppl": dense_ppl,
                    "features": strip_feature_vectors(dense_features),
                },
                "runs": runs,
                "outputs": {"json": args.out_json, "csv": args.out_csv},
                "conclusion": (
                    "Stage 1 magnitude sweep completed for dense plus configured "
                    "local unstructured magnitude sparsities."
                ),
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
                "conclusion": "Stage 1 magnitude sweep failed; inspect error and logs.",
            }
        )
        write_json(args.out_json, payload)
        print(f"wrote {args.out_json}")
        print(f"status: FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

