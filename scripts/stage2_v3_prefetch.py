from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import dataset_info, model_info, snapshot_download
from sae_lens import SAE

from ffap.json_utils import write_json


MODEL_PATTERNS = (
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
    "tokenizer*",
    "generation_config.json",
)
SAE_IDS = (
    "layer_9/width_16k/canonical",
    "layer_20/width_16k/canonical",
    "layer_31/width_16k/canonical",
)


def _hf_cache_dir(repo_id: str, repo_type: str = "model") -> str:
    cache_root = Path(os.getenv("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
    prefix = "datasets" if repo_type == "dataset" else "models"
    return str(cache_root / f"{prefix}--{repo_id.replace('/', '--')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch Stage 2 v3 assets over HTTP")
    parser.add_argument("--model-id", default="google/gemma-2-9b-it")
    parser.add_argument("--sae-release", default="gemma-scope-9b-it-res-canonical")
    parser.add_argument("--advbench-dataset", default="walledai/AdvBench")
    parser.add_argument("--xstest-dataset", default="walledai/XSTest")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--out", default="logs/stage2_v3_prefetch.json")
    args = parser.parse_args()
    started = time.time()
    payload: dict[str, Any] = {
        "step": "stage2_v3_prefetch",
        "status": "STARTED",
        "config": vars(args),
        "env": {
            "HF_HOME": os.getenv("HF_HOME"),
            "HF_HUB_CACHE": os.getenv("HF_HUB_CACHE"),
            "HF_DATASETS_CACHE": os.getenv("HF_DATASETS_CACHE"),
            "HF_HUB_DISABLE_XET": os.getenv("HF_HUB_DISABLE_XET"),
        },
    }
    try:
        if os.getenv("HF_HUB_DISABLE_XET") != "1":
            raise RuntimeError("Stage 2 v3 prefetch requires HF_HUB_DISABLE_XET=1 for HTTP downloads.")
        info = model_info(args.model_id)
        model_path = snapshot_download(
            repo_id=args.model_id,
            allow_patterns=list(MODEL_PATTERNS),
            max_workers=args.max_workers,
        )
        payload["model"] = {
            "repo_id": args.model_id,
            "revision": info.sha,
            "local_path": model_path,
            "cache_dir": _hf_cache_dir(args.model_id),
        }
        payload["saes"] = []
        sae_revision = model_info("google/gemma-scope-9b-it-res").sha
        for sae_id in SAE_IDS:
            loaded = SAE.from_pretrained(
                release=args.sae_release, sae_id=sae_id, device="cpu"
            )
            sae = loaded[0] if isinstance(loaded, tuple) else loaded
            payload["saes"].append(
                {
                    "release": args.sae_release,
                    "sae_id": sae_id,
                    "repo_id": "google/gemma-scope-9b-it-res",
                    "revision": sae_revision,
                    "cache_dir": _hf_cache_dir("google/gemma-scope-9b-it-res"),
                    "d_in": getattr(sae.cfg, "d_in", None),
                    "d_sae": getattr(sae.cfg, "d_sae", None),
                }
            )
            del sae, loaded
        advbench = load_dataset(args.advbench_dataset, split="train")
        xstest = load_dataset(args.xstest_dataset, split="test")
        payload["datasets"] = {
            "advbench": {
                "id": args.advbench_dataset,
                "revision": dataset_info(args.advbench_dataset).sha,
                "rows": len(advbench),
                "cache_dir": _hf_cache_dir(args.advbench_dataset, repo_type="dataset"),
                "cache_files": advbench.cache_files,
            },
            "xstest": {
                "id": args.xstest_dataset,
                "revision": dataset_info(args.xstest_dataset).sha,
                "rows": len(xstest),
                "cache_dir": _hf_cache_dir(args.xstest_dataset, repo_type="dataset"),
                "cache_files": xstest.cache_files,
            },
        }
        payload["status"] = "PASS"
        payload["elapsed_sec"] = round(time.time() - started, 3)
        payload["conclusion"] = "9B-IT, three matched IT SAEs, AdvBench, and XSTest are cached."
        write_json(args.out, payload)
        print(f"wrote {args.out}")
        print("status: PASS")
        return 0
    except Exception as error:
        payload["status"] = "FAIL"
        payload["elapsed_sec"] = round(time.time() - started, 3)
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(args.out, payload)
        print(f"wrote {args.out}")
        print(f"status: FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
