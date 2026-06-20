# Stage 2 W1 Ability Cross-Layer Gate

## Purpose

W1 tests whether feature-fidelity-derived cross-layer writer importance can make whole-model Wanda pruning preserve held-out ability better than geometry or random protection controls.

This branch intentionally does not use the blocked Stage 2 v3 refusal measurement route.

## Default Setup

- Model: `google/gemma-2-2b`
- SAE: `gemma-scope-2b-pt-res-canonical`
- SAE ID: `layer_12/width_16k/canonical`
- Target layer: 12
- Writer scope: `upstream`
  - `model.layers.0..12.self_attn.o_proj`
  - `model.layers.0..12.mlp.down_proj`
- Pruning: whole-model Wanda
- Protection budget: 2% of selected writer weights
- Tasks: ARC-Easy and HellaSwag
- Groups:
  - `A_feature_grad`: protect weights with high gradient saliency from the ability-causal SAE feature-fidelity objective
  - `A_loss_grad`: protect weights with high direct ability-loss gradient saliency
  - `B_wanda`: protect high Wanda-score weights among the baseline at-risk set
  - `C_random`: protect random at-risk weights
  - `D_wanda_no_protection`: plain whole-model Wanda, no protection

## Smoke

```bash
cd /root/autodl-tmp/ffap
git pull
bash remote/run_stage2_w1_ability.sh --smoke
```

Expected output files:

```text
logs/stage2_w1_run.json
logs/stage2_w1_analyze.json
results/stage2_w1_ability_smoke.json
results/stage2_w1_ability_smoke/models.csv
results/stage2_w1_ability_smoke/ability_rows.csv
results/stage2_w1_ability_smoke/mask_diagnostics.json
```

Smoke only checks plumbing and mask sanity. Do not use smoke numbers for scientific conclusions.

## Full Candidate Run

```bash
cd /root/autodl-tmp/ffap
git pull
bash remote/run_stage2_w1_ability.sh \
  --seeds 0,1,2 \
  --sparsities 0.40,0.50,0.60 \
  --ability-calibration-per-task 128 \
  --ability-test-per-task 128 \
  --writer-scope upstream
```

Expected output files:

```text
logs/stage2_w1_run.json
logs/stage2_w1_analyze.json
results/stage2_w1_ability.json
results/stage2_w1_ability/models.csv
results/stage2_w1_ability/ability_rows.csv
results/stage2_w1_ability/mask_diagnostics.json
```

## Gate Reading

The first W1 gate is a candidate gate, not a final paper claim.

Useful status values:

- `W1_PASS_CANDIDATE`: `A_feature_grad` has positive paired accuracy difference over both `B_wanda` and `C_random`.
- `W1_INCONCLUSIVE`: no clear advantage over both controls.

Primary fields:

- `group_summary`: held-out ability by group.
- `paired_tests.feature_vs_wanda_correct`
- `paired_tests.feature_vs_random_correct`
- `paired_tests.feature_vs_no_protection_correct`
- `paired_tests.loss_vs_wanda_correct`

## Disk Behavior

This script does not save pruned model checkpoints. It writes only CSV/JSON diagnostics.
