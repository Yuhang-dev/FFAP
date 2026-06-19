from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from ffap.json_utils import write_json


MODEL_ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
    "tokenizer*",
    "generation_config.json",
]


def summarize_path(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if not root.exists():
        return {"exists": False, "path": str(root)}
    total = 0
    files = 0
    for item in root.rglob("*"):
        if item.is_file():
            files += 1
            total += item.stat().st_size
    return {
        "exists": True,
        "path": str(root),
        "files": files,
        "size_gib": round(total / 1024**3, 3),
    }


def prefetch_model(model_id: str) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    started = time.time()
    path = snapshot_download(
        repo_id=model_id,
        allow_patterns=MODEL_ALLOW_PATTERNS,
        resume_download=True,
    )
    return {
        "repo_id": model_id,
        "local_path": path,
        "elapsed_sec": round(time.time() - started, 3),
        "summary": summarize_path(path),
    }


def prefetch_sae(release: str, sae_id: str) -> dict[str, Any]:
    from sae_lens import SAE

    started = time.time()
    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device="cpu")
    if isinstance(loaded, tuple):
        sae = loaded[0]
    else:
        sae = loaded
    feature_dim = getattr(sae, "cfg", None)
    return {
        "release": release,
        "sae_id": sae_id,
        "elapsed_sec": round(time.time() - started, 3),
        "d_sae": getattr(feature_dim, "d_sae", None),
        "d_in": getattr(feature_dim, "d_in", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch FFAP Task 0 assets")
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-sae", action="store_true")
    parser.add_argument("--out", default="logs/task0_prefetch.json")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "task": "task0_prefetch",
        "timestamp_unix": time.time(),
        "config": vars(args),
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        },
        "status": "STARTED",
    }

    try:
        if args.skip_model:
            payload["model"] = {"skipped": True}
        else:
            payload["model"] = prefetch_model(args.model_id)

        if args.skip_sae:
            payload["sae"] = {"skipped": True}
        else:
            payload["sae"] = prefetch_sae(args.sae_release, args.sae_id)

        payload["status"] = "PASS"
        write_json(args.out, payload)
        print(f"wrote {args.out}")
        print("status: PASS")
        return 0
    except Exception as exc:
        payload["status"] = "FAIL"
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(args.out, payload)
        print(f"wrote {args.out}")
        print(f"status: FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

