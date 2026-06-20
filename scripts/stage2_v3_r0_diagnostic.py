from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from ffap.json_utils import write_json
from ffap.stage2_v3.causal import (
    _compatibility_texts,
    collect_prompt_hidden,
    collect_token_hidden,
    feature_scale_scan,
    feature_metrics,
    layer_from_sae_id,
)
from ffap.stage2_v3.config import DEFAULT_SAE_IDS, Stage2V3Config
from ffap.stage2_v3.data import PromptExample
from ffap.stage2_v3.legacy import v2
from ffap.stage2_v3.sae_runtime import ensure_sae_runtime_normalization, sae_runtime_summary


def _strings(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


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


def _load_sae(release: str, sae_id: str, device: str, wrapped: bool) -> tuple[Any, dict[str, Any]]:
    sae, metadata = v2.load_sae_compat(release, sae_id, device)
    runtime = ensure_sae_runtime_normalization(sae) if wrapped else sae_runtime_summary(sae)
    v2.freeze_sae(sae)
    return sae, {
        "release": metadata.get("release", release),
        "sae_id": metadata.get("sae_id", sae_id),
        "cfg_type": type(metadata.get("cfg")).__name__ if metadata.get("cfg") is not None else None,
        "sparsity_loaded": bool(metadata.get("sparsity_loaded", False)),
        "runtime_normalization": {**runtime, "wrapped": wrapped},
    }


def _floats(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 v3 R0 SAE diagnostic")
    parser.add_argument("--model-id", default="google/gemma-2-9b-it")
    parser.add_argument("--sae-release", default="gemma-scope-9b-it-res-canonical")
    parser.add_argument("--sae-ids", default=",".join(DEFAULT_SAE_IDS))
    parser.add_argument("--manifest", type=Path, default=Path("results/stage2_v3/split_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("logs/stage2_v3_r0_diagnostic.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-examples", type=int, default=2)
    parser.add_argument("--compatibility-text-limit", type=int, default=256)
    parser.add_argument("--compatibility-token-limit", type=int, default=8192)
    parser.add_argument("--hook-points", default="post,pre")
    parser.add_argument("--scale-scan", default="0.125,0.25,0.333333,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R0 diagnostic.")
    started = time.time()
    if args.smoke:
        args.compatibility_text_limit = min(args.compatibility_text_limit, 16)
        args.compatibility_token_limit = min(args.compatibility_token_limit, 1024)

    manifest = _read_json(args.manifest)
    config = Stage2V3Config(
        model_id=args.model_id,
        sae_release=args.sae_release,
        sae_ids=_strings(args.sae_ids),
        max_length=args.max_length,
        batch_examples=args.batch_examples,
        compatibility_text_limit=args.compatibility_text_limit,
        compatibility_max_length=args.max_length,
        compatibility_token_limit=args.compatibility_token_limit,
        extra={"diagnostic": True, "smoke": bool(args.smoke)},
    )
    tokenizer = v2.AutoTokenizer.from_pretrained(args.model_id)
    v2.set_pad_token(tokenizer)
    model = v2.load_model(args.model_id, args.device)
    harmful = [PromptExample(**item) for item in manifest["harmful"]["calibration"]]
    benign = [PromptExample(**item) for item in manifest["benign"]["calibration"]]

    rows = []
    for sae_id in _strings(args.sae_ids):
        layer = layer_from_sae_id(sae_id)
        for hook_point in _strings(args.hook_points):
            prompt_hidden = collect_prompt_hidden(
                model,
                tokenizer,
                [item.prompt for item in harmful + benign],
                layer,
                args.max_length,
                args.batch_examples,
                args.device,
                True,
                hook_point=hook_point,
            )
            token_hidden = collect_token_hidden(
                model,
                tokenizer,
                _compatibility_texts(tokenizer, manifest, harmful, benign, config),
                layer,
                args.max_length,
                args.batch_examples,
                args.device,
                args.compatibility_token_limit,
                hook_point=hook_point,
            )
            for mode, wrapped in (("raw", False), ("wrapped", True)):
                sae, metadata = _load_sae(args.sae_release, sae_id, args.device, wrapped)
                try:
                    token_metrics = feature_metrics(sae, token_hidden, args.device, "token_level")
                    prompt_metrics = feature_metrics(sae, prompt_hidden, args.device, "prompt_final")
                    rows.append(
                        {
                            "layer": layer,
                            "sae_id": sae_id,
                            "hook_point": hook_point,
                            "mode": mode,
                            "sae_metadata": metadata,
                            "token_level": _metric_subset(token_metrics),
                            "prompt_final": _metric_subset(prompt_metrics),
                            "token_level_scale_scan": [
                                _metric_subset(row) | {"input_scale": row["input_scale"]}
                                for row in feature_scale_scan(
                                    sae,
                                    token_hidden,
                                    args.device,
                                    _floats(args.scale_scan),
                                    f"{hook_point}_token_level",
                                )
                            ],
                            "status": "PASS",
                        }
                    )
                except Exception as error:
                    rows.append(
                        {
                            "layer": layer,
                            "sae_id": sae_id,
                            "hook_point": hook_point,
                            "mode": mode,
                            "sae_metadata": metadata,
                            "status": "FAIL",
                            "error": {"type": type(error).__name__, "message": str(error)},
                        }
                    )
                finally:
                    del sae
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    result = {
        "step": "stage2_v3_r0_diagnostic",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "INCONCLUSIVE",
        "elapsed_sec": round(time.time() - started, 3),
        "config": vars(args),
        "torch": _gpu_summary(),
        "rows": rows,
        "conclusion": (
            "Compare raw vs wrapped, pre vs post hook points, and input-scale scans "
            "before deciding whether the failure is normalization, hook alignment, or SAE transfer."
        ),
    }
    write_json(args.out, result)
    print(f"wrote {args.out}")
    print(f"status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
