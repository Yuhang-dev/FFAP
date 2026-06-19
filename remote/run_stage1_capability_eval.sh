#!/usr/bin/env bash
set -euo pipefail

FFAP_ROOT="${FFAP_ROOT:-${DATA_DISK:-/root/autodl-tmp}/ffap}"
source "$FFAP_ROOT/remote/common.sh"
resolve_ffap_root
cd "$FFAP_ROOT"
activate_pbp_if_needed
configure_ffap_env

export PYTHONPATH="$FFAP_ROOT:${PYTHONPATH:-}"
mkdir -p logs results
python scripts/stage1_capability_eval.py "$@"

