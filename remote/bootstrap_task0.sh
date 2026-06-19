#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ffap
source /root/autodl-tmp/ffap/remote/common.sh
activate_pbp_if_needed
configure_ffap_env

python -m pip install -e . --no-build-isolation

