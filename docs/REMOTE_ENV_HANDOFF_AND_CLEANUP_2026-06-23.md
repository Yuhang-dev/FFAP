# FFAP Remote Environment Handoff and Cleanup Notes

Date: 2026-06-23

This document is intended for a fresh Codex/agent conversation that will continue FFAP experiments on the existing AutoDL remote machine.

## Remote Host Assumptions

- Platform: AutoDL container
- Data disk: `/root/autodl-tmp`
- Project root: `/root/autodl-tmp/ffap`
- Conda environment: `pbp`
- Python path: `/root/miniconda3/envs/pbp/bin/python`
- GPU used in successful runs: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Working PyTorch target for Blackwell: `torch==2.12.0+cu130`
- Torch CUDA version in successful environment: CUDA `13.0`
- Driver observed in prior successful environment: `590.44.01`
- Git remote: `https://github.com/Yuhang-dev/FFAP.git`
- Branch: `main`

## Important Environment Variables

The user's remote `.bashrc` already contained the following relevant variables:

```bash
export DATA_DISK=/root/autodl-tmp
export HF_HOME=$DATA_DISK/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$DATA_DISK/torch_cache
export TOKENIZERS_PARALLELISM=false
export AAP_ROOT=$DATA_DISK/aap
```

The FFAP remote scripts additionally set:

```bash
export FFAP_ROOT=${FFAP_ROOT:-$DATA_DISK/ffap}
export HF_XET_CACHE=${HF_XET_CACHE:-$HF_HOME/xet}
export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$DATA_DISK/pip_cache}
export FFAP_PIP_INDEX_URL=${FFAP_PIP_INDEX_URL:-https://pypi.org/simple}
export OMP_NUM_THREADS=1
```

Rationale:

- Use the official PyPI index for missing packages; AutoDL's default mirror caused package resolution failures earlier.
- Disable Hugging Face Xet for these model downloads; HTTP mode was more reliable.
- Keep all reusable model/dataset/package caches under `/root/autodl-tmp`.

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

Or use the task-specific wrappers, which already do the same setup:

```bash
bash remote/run_task0_preflight.sh
bash remote/run_stage2_w1_ability.sh --smoke
```

## Known Good Package State

The successful Blackwell environment used:

```text
Python: /root/miniconda3/envs/pbp/bin/python
Python: 3.11.x, observed 3.11.15
PyTorch: 2.12.0+cu130
torch CUDA: 13.0
CUDA available: True
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
datasets: 5.0.0
transformers: 5.12.1
accelerate: 1.14.0
lm-eval: 0.4.12
numpy: 2.4.4
scipy: 1.17.1
safetensors: 0.8.0
tokenizers: 0.22.2
huggingface_hub: 1.19.0
nltk: 3.9.4
langdetect: 1.0.9
immutabledict: 4.3.1
triton: 3.7.0 after torch 2.12 reinstall
```

Do not reinstall PyTorch unless it is actually broken. If PyTorch is broken on Blackwell, the target is `2.12.0+cu130`, not older CUDA 12 wheels.

## Reusable Assets to Preserve

Preserve these directories:

```text
/root/autodl-tmp/hf_cache
/root/autodl-tmp/torch_cache
/root/autodl-tmp/pip_cache
/root/miniconda3/envs/pbp
/root/autodl-tmp/ffap/.git
```

Previously downloaded reusable Hugging Face assets included:

```text
google/gemma-2-2b
google/gemma-2-9b-it
google/gemma-scope-2b-pt-res-canonical
google/gemma-scope-9b-it-res-canonical
ARC-Easy / HellaSwag datasets in HF datasets cache
AdvBench / XSTest assets if cached by datasets or HF hub
```

Do not delete `/root/autodl-tmp/hf_cache` if you want to avoid re-downloading models and datasets.

## Project Cleanup: Preserve Caches, Remove Experiment Outputs

This removes generated FFAP outputs/logs/results and local tar bundles, while preserving model/dataset/package caches and the git checkout.

Review disk usage first:

```bash
cd /root/autodl-tmp/ffap

du -sh outputs results logs 2>/dev/null || true
du -sh /root/autodl-tmp/hf_cache /root/autodl-tmp/torch_cache /root/autodl-tmp/pip_cache 2>/dev/null || true
find /root/autodl-tmp -maxdepth 1 -type f -name 'stage*_outputs.tgz' -ls 2>/dev/null || true
find /root/autodl-tmp -maxdepth 1 -type f -name 'stage2_*outputs.tgz' -ls 2>/dev/null || true
```

Clean generated artifacts:

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

Verify after cleanup:

```bash
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/ffap /root/autodl-tmp/hf_cache /root/autodl-tmp/torch_cache /root/autodl-tmp/pip_cache 2>/dev/null || true
git -C /root/autodl-tmp/ffap status --short
```

## More Aggressive Cleanup: Reclone Code, Preserve Caches

Use only if the project checkout itself can be discarded. This deletes `/root/autodl-tmp/ffap` and reclones from GitHub, but keeps HF/Torch/Pip caches.

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

After confirming the new checkout works, delete the old checkout:

```bash
cd /root/autodl-tmp
rm -rf ffap_old_delete_me_*
```

## Current Experiment State Summary

The latest completed branch was Stage 2 W1 ability cross-layer testing.

Summary:

- W1 pipeline works.
- Mask budgets are equal and masks are non-degenerate.
- `A_feature_grad` consistently beats random at-risk protection in high-sparsity settings.
- `A_feature_grad` does not robustly beat the stronger `B_wanda` at-risk geometry rescue control.
- Current interpretation: `W1_DIRECTIONAL_CANDIDATE`, not strict PASS.

Detailed report:

```text
docs/STAGE2_W1_ABILITY_EXPERIMENT_REPORT_2026-06-21.md
```

Most recent code commits relevant to W1:

```text
4438023 Tighten W1 ability gate diagnostics
a7857ed Fix W1 score summary for large tensors
001479d Fix W1 score summary sampling bounds
f05d08b Add Stage 2 W1 ability experiment report
```

## Recommended Next Technical Direction

Do not keep rerunning the same W1 arm unchanged. The current blocker is not sample size alone; it is that `A_feature_grad` remains strongly correlated with Wanda geometry.

Reasonable next experiment:

- Add a new causal arm that reduces geometry coupling, e.g.:
  - `A_feature_grad_invfreq`: reweight causal feature scores by inverse firing rate.
  - residualized feature-gradient score against Wanda score.
- Keep original `A_feature_grad` as a registered baseline.
- Compare new arm against `B_wanda`, `C_random`, and `D_wanda_no_protection` on `0.60/0.65`.
- If the new arm passes, validate with a standard lm-eval / Stage 1 capability metric.
