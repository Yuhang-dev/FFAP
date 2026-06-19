#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ffap
source /root/autodl-tmp/ffap/remote/common.sh
activate_pbp_if_needed
configure_ffap_env

export PYTHONPATH=/root/autodl-tmp/ffap:${PYTHONPATH:-}
mkdir -p logs results figures
python scripts/task0_smoke.py --out logs/task0_smoke.json "$@"

