#!/usr/bin/env bash
set -euo pipefail

FFAP_ROOT="${FFAP_ROOT:-${DATA_DISK:-/root/autodl-tmp}/ffap}"
source "$FFAP_ROOT/remote/common.sh"
resolve_ffap_root
cd "$FFAP_ROOT"
activate_pbp_if_needed
configure_ffap_env

export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE
export PYTHONPATH="$FFAP_ROOT:$FFAP_ROOT/scripts:${PYTHONPATH:-}"
mkdir -p logs results/stage2_v3

install_ffap_no_deps
python scripts/stage2_v3_prefetch.py "$@"
