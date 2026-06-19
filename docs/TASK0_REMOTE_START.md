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

