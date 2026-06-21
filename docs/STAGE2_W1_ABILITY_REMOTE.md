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

`B_wanda` is an at-risk rescue control: within the weights plain Wanda would
prune, it rescues the highest Wanda-score weights. It is not the same as the
plain Wanda baseline (`D_wanda_no_protection`).

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
- `W1_DIRECTIONAL_CANDIDATE`: `A_feature_grad` has positive mean paired accuracy difference over both controls, but the strict bootstrap CI gate did not pass.
- `W1_INCONCLUSIVE`: no clear advantage over both controls.

`W1_PASS_CANDIDATE` is strict: the 95% bootstrap CI lower bound must be positive
for `A_feature_grad` versus both `B_wanda` and `C_random`. Direction-only signals
are retained separately for triage.

Primary fields:

- `group_summary`: held-out ability by group.
- `paired_tests.feature_vs_wanda_correct`
- `paired_tests.feature_vs_random_correct`
- `paired_tests.feature_vs_no_protection_correct`
- `paired_tests.loss_vs_wanda_correct`
- `directional_signal`
- `strict_signal`
- `mask_overlap_summary`

Seed-level diagnostics are written in `seed*_diagnostics.json` under
`summary.saliency_diagnostics`:

- causal feature score versus firing/activity-mass Spearman correlations
- sampled `A_feature_grad` versus Wanda-score Spearman
- top score overlap for `A_feature_grad` versus Wanda score
- matched 2B SAE sanity metrics (`L0`, decoded cosine, reconstruction MSE,
  dead-feature rate)

`A_loss_grad` is a direct ability-loss baseline. It is reported for comparison
but does not drive the FFAP W1 gate.

## Disk Behavior

This script does not save pruned model checkpoints. It writes only CSV/JSON diagnostics.
