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
  export FFAP_PIP_INDEX_URL="${FFAP_PIP_INDEX_URL:-https://pypi.org/simple}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
    export OMP_NUM_THREADS=1
  fi
}

install_ffap_no_deps() {
  python -m pip install -e . --no-build-isolation --no-deps
}

install_task0_deps() {
  python -m pip install \
    --index-url "$FFAP_PIP_INDEX_URL" \
    --cache-dir "$PIP_CACHE_DIR" \
    --no-deps \
    einops scipy scikit-learn sae-lens lm-eval matplotlib
}
