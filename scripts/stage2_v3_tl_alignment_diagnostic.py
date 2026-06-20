from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from ffap.json_utils import write_json
from ffap.stage2_v3.causal import _compatibility_texts, feature_metrics, layer_from_sae_id
from ffap.stage2_v3.config import DEFAULT_SAE_IDS, Stage2V3Config
from ffap.stage2_v3.data import PromptExample
from ffap.stage2_v3.legacy import v2
from ffap.stage2_v3.sae_runtime import sae_runtime_summary


def _strings(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key
        in {
            "metric_scope",
            "tokens",
            "explained_variance",
            "decoded_cosine",
            "reconstruction_mse",
            "l0",
            "dead_feature_rate",
            "hidden_norm_mean",
            "reconstruction_norm_mean",
            "reconstruction_to_hidden_norm",
            "optimal_reconstruction_scale",
            "optimal_scaled_explained_variance",
        }
    }


def _gpu_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def _load_tl_model(model_name: str, device: str) -> Any:
    from transformer_lens import HookedTransformer

    load = getattr(HookedTransformer, "from_pretrained_no_processing", None)
    if load is None:
        load = HookedTransformer.from_pretrained
    attempts = (
        {"device": device, "dtype": torch.bfloat16},
        {"device": device, "dtype": "bfloat16"},
        {"device": device},
    )
    errors = []
    for kwargs in attempts:
        try:
            model = load(model_name, **kwargs)
            model.eval()
            model.requires_grad_(False)
            return model
        except TypeError as error:
            errors.append(f"{kwargs}: {error}")
    raise RuntimeError("Could not load TransformerLens model: " + " | ".join(errors))


@torch.no_grad()
def collect_tl_token_hidden(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    hook_name: str,
    max_length: int,
    device: str,
    token_limit: int,
) -> torch.Tensor:
    rows = []
    total = 0
    original_side = getattr(tokenizer, "padding_side", None)
    if original_side is not None:
        tokenizer.padding_side = "right"
    try:
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = encoded.input_ids.to(device)
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name == hook_name,
                return_type=None,
                remove_batch_dim=False,
            )
            hidden = cache[hook_name].detach().float().cpu().reshape(-1, cache[hook_name].shape[-1])
            if token_limit > 0:
                remaining = token_limit - total
                if remaining <= 0:
                    break
                hidden = hidden[:remaining]
            rows.append(hidden)
            total += hidden.shape[0]
            if token_limit > 0 and total >= token_limit:
                break
    finally:
        if original_side is not None:
            tokenizer.padding_side = original_side
    if not rows:
        raise RuntimeError("No TransformerLens activations were captured.")
    return torch.cat(rows, dim=0)


def _tl_cfg_summary(model: Any) -> dict[str, Any]:
    cfg = getattr(model, "cfg", None)
    keys = (
        "model_name",
        "n_layers",
        "d_model",
        "original_architecture",
        "normalization_type",
        "dtype",
        "device",
        "tokenizer_name",
        "default_prepend_bos",
    )
    return {key: str(getattr(cfg, key, None)) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 v3 TransformerLens SAE alignment diagnostic")
    parser.add_argument("--tl-model-name", default="gemma-2-9b-it")
    parser.add_argument("--sae-release", default="gemma-scope-9b-it-res-canonical")
    parser.add_argument("--sae-ids", default=",".join(DEFAULT_SAE_IDS))
    parser.add_argument("--manifest", type=Path, default=Path("results/stage2_v3/split_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("logs/stage2_v3_tl_alignment_diagnostic.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--compatibility-text-limit", type=int, default=128)
    parser.add_argument("--compatibility-token-limit", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TransformerLens alignment diagnostic.")
    if args.smoke:
        args.compatibility_text_limit = min(args.compatibility_text_limit, 16)
        args.compatibility_token_limit = min(args.compatibility_token_limit, 1024)

    started = time.time()
    manifest = _read_json(args.manifest)
    model = _load_tl_model(args.tl_model_name, args.device)
    tokenizer = model.tokenizer
    v2.set_pad_token(tokenizer)
    harmful = [PromptExample(**item) for item in manifest["harmful"]["calibration"]]
    benign = [PromptExample(**item) for item in manifest["benign"]["calibration"]]
    config = Stage2V3Config(
        compatibility_text_limit=args.compatibility_text_limit,
        compatibility_max_length=args.max_length,
        compatibility_token_limit=args.compatibility_token_limit,
        extra={"tl_alignment_diagnostic": True, "smoke": bool(args.smoke)},
    )
    texts = _compatibility_texts(tokenizer, manifest, harmful, benign, config)

    rows = []
    for sae_id in _strings(args.sae_ids):
        layer = layer_from_sae_id(sae_id)
        hook_name = f"blocks.{layer}.hook_resid_post"
        sae, metadata = v2.load_sae_compat(args.sae_release, sae_id, args.device)
        v2.freeze_sae(sae)
        try:
            hidden = collect_tl_token_hidden(
                model,
                tokenizer,
                texts,
                hook_name,
                args.max_length,
                args.device,
                args.compatibility_token_limit,
            )
            metrics = feature_metrics(sae, hidden, args.device, "tl_hook_resid_post_token_level")
            rows.append(
                {
                    "layer": layer,
                    "sae_id": sae_id,
                    "hook_name": hook_name,
                    "sae_metadata": {
                        "release": metadata.get("release", args.sae_release),
                        "sae_id": metadata.get("sae_id", sae_id),
                        "runtime_normalization": sae_runtime_summary(sae),
                    },
                    "token_level": _metric_subset(metrics),
                    "status": "PASS",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "layer": layer,
                    "sae_id": sae_id,
                    "hook_name": hook_name,
                    "status": "FAIL",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
        finally:
            del sae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = {
        "step": "stage2_v3_tl_alignment_diagnostic",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "INCONCLUSIVE",
        "elapsed_sec": round(time.time() - started, 3),
        "config": vars(args),
        "transformer_lens": _tl_cfg_summary(model),
        "torch": _gpu_summary(),
        "rows": rows,
        "conclusion": (
            "If TransformerLens hook_resid_post reconstructs well, replace HF hooks with TL-aligned "
            "activation collection. If it does not, Stage 2 v3 R0 is blocked by matched-SAE transfer."
        ),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(args.out, result)
    print(f"wrote {args.out}")
    print(f"status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
