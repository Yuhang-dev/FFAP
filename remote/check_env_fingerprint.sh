#!/usr/bin/env bash
set -euo pipefail

FFAP_ROOT="${FFAP_ROOT:-${DATA_DISK:-/root/autodl-tmp}/ffap}"
source "$FFAP_ROOT/remote/common.sh"
resolve_ffap_root
cd "$FFAP_ROOT"
activate_pbp_if_needed
configure_ffap_env

python - <<'PY'
import importlib.metadata as m
import json
import sys
from pathlib import Path

payload = {
    "python": sys.executable,
    "python_version": sys.version,
    "packages": {},
}

for pkg in [
    "torch",
    "datasets",
    "transformers",
    "accelerate",
    "lm-eval",
    "numpy",
    "scipy",
    "safetensors",
    "tokenizers",
    "huggingface-hub",
    "nltk",
    "langdetect",
    "immutabledict",
    "triton",
    "sae-lens",
]:
    try:
        payload["packages"][pkg] = m.version(pkg)
    except Exception as exc:
        payload["packages"][pkg] = f"MISSING: {exc}"

try:
    import torch

    payload["torch_runtime"] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "arch_list": torch.cuda.get_arch_list(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload["torch_runtime"].update(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "memory_gib": round(props.total_memory / 1024**3, 2),
            }
        )
        x = torch.ones(1, device="cuda")
        payload["torch_runtime"]["kernel_ok"] = float((x + 1).item())
except Exception as exc:
    payload["torch_runtime"] = {"error": repr(exc)}

Path("logs").mkdir(exist_ok=True)
Path("logs/env_fingerprint.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
