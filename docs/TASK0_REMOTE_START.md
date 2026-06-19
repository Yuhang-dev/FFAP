# Task 0 Remote Start

Task 0 checks the first mandatory environment gate from `FFAP_experiment_spec.md`.

## Expected Remote

The provided handoff says the active environment is:

```text
host: autodl-container-fu642y22eg-c8ab8624
user: root
env: pbp
repo: /root/autodl-tmp/preference-boundary-pruning
HF_HOME: /root/autodl-tmp/hf_cache
```

For FFAP, use a separate remote working directory:

```bash
/root/autodl-tmp/ffap
```

Optional `.bashrc` helper matching the existing AAP/PBP setup:

```bash
export FFAP_ROOT=$DATA_DISK/ffap

ffapenv() {
  cd "$FFAP_ROOT"
  conda activate pbp
  export PYTHONPATH="$FFAP_ROOT:${PYTHONPATH:-}"
}
```

## Current Blocker

The handoff recorded `torch.cuda.is_available: False` and empty GPU sections.
Before running the smoke test, run:

```bash
cd /root/autodl-tmp/ffap
bash remote/run_task0_preflight.sh
```

Do not run the full smoke test until `logs/task0_preflight.json` reports
`status: PASS`.

## Smoke Command

Install Task 0 Python dependencies from official PyPI. This intentionally uses
`https://pypi.org/simple` instead of the AutoDL default mirror:

```bash
cd /root/autodl-tmp/ffap
bash remote/install_task0_deps.sh
```

Prefetch required model/SAE assets first:

```bash
cd /root/autodl-tmp/ffap
bash remote/prefetch_task0_assets.sh
```

```bash
cd /root/autodl-tmp/ffap
bash remote/run_task0_smoke.sh
```

Optional faster debugging without ARC-Easy:

```bash
bash remote/run_task0_smoke.sh --skip-lm-eval
```

## Success Artifact

Task 0 is complete only when `logs/task0_smoke.json` contains:

- model load elapsed time and peak GPU memory
- dense forward elapsed time
- SAE reconstruction MSE and L0
- ARC-Easy small-subset baseline from `lm-eval`
