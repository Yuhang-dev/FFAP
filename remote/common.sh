#!/usr/bin/env bash

activate_pbp_if_needed() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "pbp" ]]; then
    return 0
  fi

  if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/root/miniconda3/etc/profile.d/conda.sh"
  elif [[ -x "/root/miniconda3/bin/conda" ]]; then
    eval "$(/root/miniconda3/bin/conda shell.bash hook)"
  else
    echo "Could not find conda initialization under /root/miniconda3" >&2
    return 1
  fi

  conda activate pbp
}

resolve_ffap_root() {
  export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
  export FFAP_ROOT="${FFAP_ROOT:-$DATA_DISK/ffap}"
}

configure_ffap_env() {
  export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
  export FFAP_ROOT="${FFAP_ROOT:-$DATA_DISK/ffap}"
  export HF_HOME="${HF_HOME:-$DATA_DISK/hf_cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export TORCH_HOME="${TORCH_HOME:-$DATA_DISK/torch_cache}"
  export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
  export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_DISK/pip_cache}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}
