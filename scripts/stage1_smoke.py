from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ffap.json_utils import write_json


def load_sae_compat(release: str, sae_id: str, device: str) -> tuple[Any, dict[str, Any]]:
    from sae_lens import SAE

    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    metadata: dict[str, Any] = {"release": release, "sae_id": sae_id}
    if isinstance(loaded, tuple):
        sae = loaded[0]
        if len(loaded) > 1:
            metadata["cfg"] = loaded[1]
        if len(loaded) > 2 and loaded[2] is not None:
            metadata["sparsity_loaded"] = True
    else:
        sae = loaded
    sae.eval()
    sae.to(device)
    return sae, metadata


def get_wikitext_texts(split: str, num_texts: int, seed: int) -> list[str]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = [row["text"].strip() for row in ds if row.get("text", "").strip()]
    texts = [text for text in texts if len(text.split()) >= 16]
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:num_texts]


def make_blocks(
    tokenizer: Any,
    texts: list[str],
    seq_len: int,
    max_blocks: int,
    device: str,
) -> list[torch.Tensor]:
    joined = "\n\n".join(texts)
    tokenized = tokenizer(joined, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized.input_ids[0]
    blocks = []
    for start in range(0, max(0, input_ids.numel() - 1), seq_len):
        block = input_ids[start : start + seq_len]
        if block.numel() < 8:
            break
        blocks.append(block.unsqueeze(0).to(device))
        if len(blocks) >= max_blocks:
            break
    if not blocks:
        raise RuntimeError("No token blocks created for WikiText evaluation.")
    return blocks


@torch.no_grad()
def evaluate_ppl(model: Any, blocks: list[torch.Tensor]) -> dict[str, float]:
    losses = []
    token_count = 0
    for block in tqdm(blocks, desc="PPL blocks"):
        outputs = model(input_ids=block, labels=block)
        tokens = max(1, block.numel() - 1)
        losses.append(float(outputs.loss.detach().cpu()) * tokens)
        token_count += tokens
    mean_nll = sum(losses) / max(1, token_count)
    return {
        "nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "tokens": token_count,
        "blocks": len(blocks),
    }


def capture_layer_activation(model: Any, layer_index: int, input_ids: torch.Tensor) -> torch.Tensor:
    activations: list[torch.Tensor] = []
    layer = model.model.layers[layer_index]

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        activations.append(hidden.detach())

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids)
    finally:
        handle.remove()
    if not activations:
        raise RuntimeError(f"No activation captured for layer {layer_index}")
    return activations[-1]


@torch.no_grad()
def collect_feature_stats(
    model: Any,
    sae: Any,
    blocks: list[torch.Tensor],
    layer: int,
) -> dict[str, Any]:
    active_counts = None
    activation_sums = None
    total_tokens = 0
    l0_sum = 0.0
    mse_sum = 0.0
    cosine_sum = 0.0

    for block in tqdm(blocks, desc="SAE feature blocks"):
        activation = capture_layer_activation(model, layer, block)
        flat = activation.reshape(-1, activation.shape[-1])
        features = sae.encode(flat)
        reconstruction = sae.decode(features)

        active = features > 0
        if active_counts is None:
            active_counts = active.sum(dim=0).detach().cpu().to(torch.float32)
            activation_sums = features.detach().cpu().to(torch.float32).sum(dim=0)
        else:
            active_counts += active.sum(dim=0).detach().cpu().to(torch.float32)
            activation_sums += features.detach().cpu().to(torch.float32).sum(dim=0)

        tokens = flat.shape[0]
        total_tokens += tokens
        l0_sum += float(active.sum(dim=-1).float().mean().detach().cpu()) * tokens
        mse_sum += float(
            torch.mean((reconstruction.float() - flat.float()) ** 2).detach().cpu()
        ) * tokens
        cosine_sum += float(
            torch.nn.functional.cosine_similarity(
                reconstruction.float(), flat.float(), dim=-1
            )
            .mean()
            .detach()
            .cpu()
        ) * tokens

    assert active_counts is not None
    assert activation_sums is not None
    firing_rate = active_counts / max(1, total_tokens)
    mean_activation = activation_sums / max(1, total_tokens)
    active_features = firing_rate > 0
    return {
        "tokens": total_tokens,
        "active_features_count": int(active_features.sum().item()),
        "active_features": active_features,
        "firing_rate": firing_rate,
        "mean_activation": mean_activation,
        "l0": l0_sum / max(1, total_tokens),
        "reconstruction_mse": mse_sum / max(1, total_tokens),
        "decoded_activation_cosine": cosine_sum / max(1, total_tokens),
    }


def compare_feature_stats(dense: dict[str, Any], pruned: dict[str, Any]) -> dict[str, float]:
    dense_active = dense["active_features"]
    pruned_active = pruned["active_features"]
    intersection = (dense_active & pruned_active).sum().item()
    union = (dense_active | pruned_active).sum().item()
    return {
        "active_jaccard": float(intersection / union) if union else 1.0,
        "firing_rate_l1": float(
            torch.mean(torch.abs(dense["firing_rate"] - pruned["firing_rate"])).item()
        ),
        "mean_activation_l1": float(
            torch.mean(
                torch.abs(dense["mean_activation"] - pruned["mean_activation"])
            ).item()
        ),
        "l0_delta": float(pruned["l0"] - dense["l0"]),
        "reconstruction_mse_delta": float(
            pruned["reconstruction_mse"] - dense["reconstruction_mse"]
        ),
        "decoded_activation_cosine_delta": float(
            pruned["decoded_activation_cosine"] - dense["decoded_activation_cosine"]
        ),
    }


def iter_prunable_linear_weights(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    weights = []
    for name, module in model.named_modules():
        if name == "lm_head":
            continue
        if isinstance(module, torch.nn.Linear):
            weights.append((f"{name}.weight", module.weight))
    return weights


@torch.no_grad()
def apply_local_magnitude_pruning(model: Any, sparsity: float) -> dict[str, Any]:
    total = 0
    pruned = 0
    layer_summaries = []
    for name, weight in tqdm(iter_prunable_linear_weights(model), desc="Magnitude prune"):
        numel = weight.numel()
        k = int(numel * sparsity)
        if k <= 0:
            continue
        flat_abs = weight.detach().abs().float().flatten()
        threshold = torch.kthvalue(flat_abs, k).values
        mask = weight.detach().abs() <= threshold.to(weight.device)
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
                "threshold": float(threshold.detach().cpu()),
            }
        )
    return {
        "method": "local_magnitude_unstructured",
        "target_sparsity": sparsity,
        "total_considered": total,
        "total_pruned": pruned,
        "actual_sparsity": pruned / total if total else 0.0,
        "layers": layer_summaries,
    }


def strip_feature_vectors(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stats.items()
        if key not in {"active_features", "firing_rate", "mean_activation"}
    }


def resolve_save_dir(save_pruned_dir: str, method: str, sparsity: float) -> Path | None:
    if save_pruned_dir.lower() in {"", "none", "false", "no"}:
        return None
    if save_pruned_dir == "auto":
        return Path("outputs/stage1_smoke") / f"{method}_s{sparsity:.2f}"
    return Path(save_pruned_dir)


def write_stage1_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 smoke experiment")
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--num-texts", type=int, default=64)
    parser.add_argument("--ppl-seq-len", type=int, default=256)
    parser.add_argument("--ppl-blocks", type=int, default=16)
    parser.add_argument("--feature-seq-len", type=int, default=128)
    parser.add_argument("--feature-blocks", type=int, default=8)
    parser.add_argument("--sparsity", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-pruned-dir",
        default="auto",
        help="Directory for saving the pruned model, or 'none' to disable.",
    )
    parser.add_argument("--out-json", default="logs/stage1_smoke.json")
    parser.add_argument("--out-csv", default="results/stage1_smoke.csv")
    args = parser.parse_args()

    started = time.time()
    payload: dict[str, Any] = {
        "task": "stage1_smoke",
        "timestamp_unix": started,
        "config": vars(args),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        },
        "status": "STARTED",
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Stage 1 smoke.")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        device = "cuda:0"
        torch.cuda.reset_peak_memory_stats()

        texts = get_wikitext_texts(args.dataset_split, args.num_texts, args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        model.eval()
        sae, sae_metadata = load_sae_compat(args.sae_release, args.sae_id, device)

        ppl_blocks = make_blocks(
            tokenizer, texts, args.ppl_seq_len, args.ppl_blocks, device
        )
        feature_blocks = make_blocks(
            tokenizer, texts, args.feature_seq_len, args.feature_blocks, device
        )

        dense_ppl = evaluate_ppl(model, ppl_blocks)
        dense_features = collect_feature_stats(model, sae, feature_blocks, args.layer)

        prune_summary = apply_local_magnitude_pruning(model, args.sparsity)
        save_dir = resolve_save_dir(
            args.save_pruned_dir, prune_summary["method"], args.sparsity
        )
        saved_pruned: dict[str, Any] = {"enabled": save_dir is not None}
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir, safe_serialization=True)
            tokenizer.save_pretrained(save_dir)
            saved_pruned = {
                "enabled": True,
                "path": str(save_dir),
                "files": [
                    str(path.relative_to(save_dir))
                    for path in sorted(save_dir.rglob("*"))
                    if path.is_file()
                ],
            }
        pruned_ppl = evaluate_ppl(model, ppl_blocks)
        pruned_features = collect_feature_stats(model, sae, feature_blocks, args.layer)
        feature_damage = compare_feature_stats(dense_features, pruned_features)

        rows = [
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
            },
            {
                "model_id": args.model_id,
                "method": "local_magnitude_unstructured",
                "sparsity": prune_summary["actual_sparsity"],
                "target_sparsity": args.sparsity,
                "ppl": pruned_ppl["ppl"],
                "nll": pruned_ppl["nll"],
                "ppl_tokens": pruned_ppl["tokens"],
                "feature_l0": pruned_features["l0"],
                "feature_reconstruction_mse": pruned_features["reconstruction_mse"],
                "feature_decoded_activation_cosine": pruned_features[
                    "decoded_activation_cosine"
                ],
                "active_features_count": pruned_features["active_features_count"],
                **feature_damage,
            },
        ]
        write_stage1_csv(Path(args.out_csv), rows)

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
                "pruning": prune_summary,
                "saved_pruned": saved_pruned,
                "pruned": {
                    "ppl": pruned_ppl,
                    "features": strip_feature_vectors(pruned_features),
                },
                "feature_damage": feature_damage,
                "outputs": {
                    "json": args.out_json,
                    "csv": args.out_csv,
                },
                "conclusion": (
                    "Stage 1 smoke completed for dense vs 20% local magnitude "
                    "pruning on a small WikiText/SAE sample."
                ),
            }
        )
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
                "conclusion": "Stage 1 smoke failed; inspect error and logs.",
            }
        )
        write_json(args.out_json, payload)
        print(f"wrote {args.out_json}")
        print(f"status: FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
