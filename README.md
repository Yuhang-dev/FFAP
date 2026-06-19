# FFAP Experiments

Feature-Fidelity-Aware Pruning (FFAP) experiment scaffold.

This repository starts with Task 0 from `FFAP_experiment_spec.md`: verify that
the remote machine can load `google/gemma-2-2b`, run a Gemma Scope SAE
reconstruction pass, and run a small ARC-Easy baseline through `lm-eval`.

Do not proceed past the Stage 2 gate without human confirmation.

## Remote Quick Start

Expected remote working directory:

```bash
${FFAP_ROOT:-/root/autodl-tmp/ffap}
```

Run the environment preflight first:

```bash
cd /root/autodl-tmp/ffap
bash remote/run_task0_preflight.sh
```

If CUDA is available, run the Task 0 smoke test:

```bash
cd /root/autodl-tmp/ffap
bash remote/run_task0_smoke.sh
```

Artifacts:

```text
logs/task0_preflight.json
logs/task0_smoke.json
results/task0_lm_eval/
```

## Local Policy

Local execution is limited to syntax checks and lightweight inspection.
Do not load large models or download datasets locally.

Allowed local check:

```bash
python -m compileall ffap scripts
```
