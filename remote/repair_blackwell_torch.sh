#!/usr/bin/env bash
set -euo pipefail

FFAP_ROOT="${FFAP_ROOT:-${DATA_DISK:-/root/autodl-tmp}/ffap}"
source "$FFAP_ROOT/remote/common.sh"
resolve_ffap_root
cd "$FFAP_ROOT"
activate_pbp_if_needed
configure_ffap_env

python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --pre torch \
  --index-url https://download.pytorch.org/whl/nightly/cu128
python -m pip uninstall -y torchvision torchaudio

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("arch_list", torch.cuda.get_arch_list())
print("available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
x = torch.ones(1, device="cuda")
print("kernel_ok", (x + 1).item())
PY
