from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ffap.json_utils import write_json


PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "huggingface_hub",
    "sae_lens",
    "lm_eval",
    "numpy",
    "scipy",
    "sklearn",
    "einops",
]

DIST_NAMES = {
    "huggingface_hub": "huggingface-hub",
    "sae_lens": "sae-lens",
    "lm_eval": "lm-eval",
    "sklearn": "scikit-learn",
}


def package_version(import_name: str) -> dict[str, Any]:
    dist_name = DIST_NAMES.get(import_name, import_name)
    try:
        return {"ok": True, "version": importlib.metadata.version(dist_name)}
    except importlib.metadata.PackageNotFoundError as exc:
        return {"ok": False, "error": repr(exc)}


def run_command(command: list[str], timeout: int = 30) -> dict[str, Any]:
    started = time.time()
    if shutil.which(command[0]) is None:
        return {"ok": False, "error": f"{command[0]} not found"}
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def torch_probe() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"import_ok": False, "error": repr(exc)}

    payload: dict[str, Any] = {
        "import_ok": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count(),
    }
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_gib": round(props.total_memory / 1024**3, 2),
                "major": props.major,
                "minor": props.minor,
            }
        )
    payload["devices"] = devices
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="FFAP Task 0 remote preflight")
    parser.add_argument("--out", default="logs/task0_preflight.json")
    args = parser.parse_args()

    payload = {
        "task": "task0_preflight",
        "timestamp_unix": time.time(),
        "host": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "env": {
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        },
        "packages": {name: package_version(name) for name in PACKAGES},
        "torch": torch_probe(),
        "nvidia_smi": run_command(["nvidia-smi"], timeout=30),
        "disk": run_command(["df", "-h"], timeout=30),
    }
    payload["status"] = (
        "PASS" if payload["torch"].get("cuda_available") else "FAIL_NO_CUDA"
    )
    write_json(args.out, payload)

    print(f"wrote {args.out}")
    print(f"status: {payload['status']}")
    if payload["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

