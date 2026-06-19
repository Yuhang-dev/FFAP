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

Install Task 0 Python dependencies from official PyPI:

```bash
cd /root/autodl-tmp/ffap
bash remote/install_task0_deps.sh
```

If PyTorch was downgraded on a Blackwell GPU, repair it to the known-good
Phase 1 target (`torch==2.12.0+cu130`) with:

```bash
cd /root/autodl-tmp/ffap
bash remote/repair_blackwell_torch.sh
```

Record the active environment fingerprint with:

```bash
cd /root/autodl-tmp/ffap
bash remote/check_env_fingerprint.sh
```

Prefetch Task 0 assets:

```bash
cd /root/autodl-tmp/ffap
bash remote/prefetch_task0_assets.sh
```

If CUDA is available, run the Task 0 smoke test:

```bash
cd /root/autodl-tmp/ffap
bash remote/run_task0_smoke.sh
```

Run the first Stage 1 smoke experiment:

```bash
cd /root/autodl-tmp/ffap
bash remote/run_stage1_smoke.sh
```

Run the Stage 1 magnitude sparsity sweep:

```bash
cd /root/autodl-tmp/ffap
bash remote/run_stage1_magnitude_sweep.sh
```

Run small-subset capability eval for dense and saved magnitude checkpoints:

```bash
cd /root/autodl-tmp/ffap
bash remote/run_stage1_capability_eval.sh
```

Artifacts:

```text
logs/task0_preflight.json
logs/task0_prefetch.json
logs/task0_smoke.json
logs/stage1_smoke.json
logs/stage1_magnitude_sweep.json
logs/stage1_capability_eval.json
results/task0_lm_eval/
results/stage1_smoke.csv
results/stage1_magnitude_sweep.csv
results/stage1_capability_eval.csv
```

## Local Policy

Local execution is limited to syntax checks and lightweight inspection.
Do not load large models or download datasets locally.

Allowed local check:

```bash
python -m compileall ffap scripts
```
