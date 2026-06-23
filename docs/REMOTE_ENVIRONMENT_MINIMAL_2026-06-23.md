# Remote Environment Minimal Notes

Date: 2026-06-23

This file is a generic remote-environment note for future work on the same AutoDL machine. It intentionally does not include any experiment handoff, next-stage plan, or project-specific scientific conclusion.

## Remote Machine

```text
Platform: AutoDL container
Data disk: /root/autodl-tmp
Project root: /root/autodl-tmp/ffap
Conda env: pbp
Python: /root/miniconda3/envs/pbp/bin/python
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
Working PyTorch target: torch 2.12.0+cu130
Torch CUDA: 13.0
```

Do not create a new conda environment unless the existing `pbp` environment is broken. The known-good setup uses the existing `pbp` environment.

## Existing Cache Locations

Preserve these across cleanups:

```bash
/root/autodl-tmp/hf_cache
/root/autodl-tmp/torch_cache
/root/autodl-tmp/pip_cache
/root/miniconda3/envs/pbp
```

Useful environment variables:

```bash
export DATA_DISK=/root/autodl-tmp
export FFAP_ROOT=$DATA_DISK/ffap
export HF_HOME=$DATA_DISK/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$DATA_DISK/torch_cache
export HF_XET_CACHE=$HF_HOME/xet
export PIP_CACHE_DIR=$DATA_DISK/pip_cache
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

If gated Hugging Face models are needed, use the `HF_TOKEN` already configured on the remote machine or set it privately in the shell. Do not paste the token into logs or repository files.

## Standard Session Bootstrap

Use this at the start of a remote session:

```bash
cd /root/autodl-tmp/ffap
git pull

source remote/common.sh
resolve_ffap_root
activate_pbp_if_needed
configure_ffap_env
install_ffap_no_deps
```

Equivalent manual form:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pbp

export DATA_DISK=/root/autodl-tmp
export FFAP_ROOT=$DATA_DISK/ffap
export HF_HOME=$DATA_DISK/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$DATA_DISK/torch_cache
export HF_XET_CACHE=$HF_HOME/xet
export PIP_CACHE_DIR=$DATA_DISK/pip_cache
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$FFAP_ROOT"
python -m pip install -e . --no-build-isolation --no-deps
```

## Package Install Rules

Use the official PyPI index when installing missing Python packages:

```bash
python -m pip install --index-url https://pypi.org/simple --cache-dir /root/autodl-tmp/pip_cache <package>
```

Reason: the AutoDL default mirror previously caused package resolution failures for some packages.

Do not reinstall PyTorch unless necessary. If PyTorch must be repaired for the Blackwell GPU, target the known-good family:

```text
torch==2.12.0+cu130
```

Avoid CUDA 12 PyTorch wheels on this Blackwell machine.

## Environment Verification Commands

```bash
which python
python - <<'PY'
import torch
print("python ok")
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

df -h /root/autodl-tmp
du -sh /root/autodl-tmp/hf_cache /root/autodl-tmp/torch_cache /root/autodl-tmp/pip_cache 2>/dev/null || true
```

## Cleanup: Remove Project Outputs, Preserve Caches

This removes generated project artifacts but keeps reusable downloaded models/datasets and package caches.

Preview:

```bash
cd /root/autodl-tmp/ffap

du -sh outputs results logs 2>/dev/null || true
du -sh /root/autodl-tmp/hf_cache /root/autodl-tmp/torch_cache /root/autodl-tmp/pip_cache 2>/dev/null || true
find /root/autodl-tmp -maxdepth 1 -type f \
  \( -name 'stage*_outputs.tgz' -o -name 'stage2_*outputs.tgz' -o -name '*_bundle.tgz' -o -name '*_bundle.tar.gz' \) \
  -ls 2>/dev/null || true
```

Clean:

```bash
cd /root/autodl-tmp/ffap

rm -rf outputs/*
rm -rf results/*
rm -rf logs/*

find /root/autodl-tmp -maxdepth 1 -type f \
  \( -name 'stage*_outputs.tgz' -o -name 'stage2_*outputs.tgz' -o -name '*_bundle.tgz' -o -name '*_bundle.tar.gz' \) \
  -print -delete

find /root/autodl-tmp/hf_cache -type f \
  \( -name '*.incomplete' -o -name '*.tmp' \) \
  -print -delete 2>/dev/null || true

mkdir -p outputs results logs
```

Verify:

```bash
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/ffap /root/autodl-tmp/hf_cache /root/autodl-tmp/torch_cache /root/autodl-tmp/pip_cache 2>/dev/null || true
git -C /root/autodl-tmp/ffap status --short
```

## More Aggressive Cleanup: Reclone Code, Preserve Caches

Use this only if the checkout itself can be discarded. This preserves `/root/autodl-tmp/hf_cache`, `/root/autodl-tmp/torch_cache`, `/root/autodl-tmp/pip_cache`, and the `pbp` conda environment.

```bash
cd /root/autodl-tmp

mv ffap ffap_old_delete_me_$(date +%Y%m%d_%H%M%S)
git clone https://github.com/Yuhang-dev/FFAP.git ffap
cd ffap

source remote/common.sh
resolve_ffap_root
activate_pbp_if_needed
configure_ffap_env
install_ffap_no_deps

du -sh ../ffap_old_delete_me_* 2>/dev/null || true
```

After confirming the new checkout works:

```bash
cd /root/autodl-tmp
rm -rf ffap_old_delete_me_*
```
