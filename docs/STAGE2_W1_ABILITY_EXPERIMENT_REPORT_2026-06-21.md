# Stage 2 W1 Ability Cross-Layer Experiments Report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-06-21
- Verification Status: ANALYZED
- Version Label: validation_v1
- Project: FFAP
- Scope: Stage 2 W1 ability cross-layer protection experiments only
- Code Path: `scripts/stage2_w1_ability_crosslayer.py`
- Remote Wrapper: `remote/run_stage2_w1_ability.sh`
- Bundles Analyzed:
  - `stage2_w1_smoke_outputs.tgz`
  - `stage2_w1_full_outputs.tgz`
  - `stage2_w1_highsparse_outputs.tgz`
  - `stage2_w1_confirm_060_065_outputs.tgz`

## Executive Summary

The W1 ability cross-layer pipeline is operational and the masks/interventions are not degenerate. Across repeated runs, `A_feature_grad` shows a directional advantage over `C_random`, and in the high-sparsity run it also strictly beats `D_wanda_no_protection`. However, the key comparison against `B_wanda` does not pass the strict bootstrap gate.

Current status:

```text
W1_DIRECTIONAL_CANDIDATE
```

Interpretation:

- Not a strict PASS.
- Not a pipeline/mask failure.
- Evidence supports a weak-to-moderate high-sparsity directional signal.
- The remaining blocker is that `A_feature_grad` does not robustly beat the `B_wanda` at-risk geometry rescue control.
- A likely mechanism is partial coupling between feature-gradient saliency and Wanda geometry: `A_feature_grad` vs Wanda score Spearman is consistently about `0.82`.

## Experimental Design

Model and SAE:

```text
Model: google/gemma-2-2b
SAE release: gemma-scope-2b-pt-res-canonical
SAE ID: layer_12/width_16k/canonical
Target SAE layer: 12
Writer scope: upstream
Writer modules: model.layers.0..12.self_attn.o_proj and model.layers.0..12.mlp.down_proj
Pruning method: whole-model Wanda with controlled writer-weight protection
```

Arms:

| Arm | Meaning | Gate Role |
|---|---|---|
| `A_feature_grad` | Protect weights with high gradient saliency from ability-causal SAE feature-fidelity objective | Main FFAP arm |
| `A_loss_grad` | Protect weights with high direct ability-loss gradient saliency | Direct-loss baseline; not gate-driving |
| `B_wanda` | Protect high Wanda-score weights inside the plain-Wanda at-risk set | Main geometry rescue control |
| `C_random` | Protect random weights inside the same at-risk budget | Random rescue control |
| `D_wanda_no_protection` | Plain whole-model Wanda without writer protection | No-protection baseline |

Gate rule implemented after code revision:

```text
W1_PASS_CANDIDATE:
  A_feature_grad 95% bootstrap CI lower bound > 0 versus both B_wanda and C_random.

W1_DIRECTIONAL_CANDIDATE:
  A_feature_grad mean paired accuracy difference > 0 versus both B_wanda and C_random,
  but strict CI gate does not pass.
```

Note: W1 internal ability metric is an in-script continuation log-prob multiple-choice objective. It is not identical to Stage 1 lm-eval / capability evaluation. Relative within-run comparisons are interpretable; absolute dense accuracy should not be compared directly to Stage 1 scores.

## Run Inventory

| Run | Purpose | Sparsities | Seeds | Test per Task | Gate |
|---|---|---:|---:|---:|---|
| `smoke` | Pipeline and mask sanity only | `[0.50]` | `[0]` | 8 | `W1_INCONCLUSIVE` |
| `full_040_050_060` | First full W1 candidate | `[0.40, 0.50, 0.60]` | `[0,1,2]` | 128 | `W1_DIRECTIONAL_CANDIDATE` |
| `highsparse_055_060_065` | High-sparsity follow-up | `[0.55, 0.60, 0.65]` | `[0,1,2]` | 128 | `W1_DIRECTIONAL_CANDIDATE` |
| `confirm_060_065` | Focused confirmation with larger held-out set | `[0.60, 0.65]` | `[0,1,2]` | 256 | `W1_DIRECTIONAL_CANDIDATE` |

## Primary Statistical Results

Effect units are percentage points of paired accuracy difference. Positive means `A_feature_grad` is better than the comparator.

| Run | A > B_wanda Mean pp (95% CI, p) | A > C_random Mean pp (95% CI, p) | A > D_plain Mean pp (95% CI, p) |
|---|---:|---:|---:|
| smoke | 0.00 `[-18.75, 18.75]`, p=1.0000 | -6.25 `[-18.75, 0.00]`, p=0.7200 | -6.25 `[-18.75, 0.00]`, p=0.7200 |
| full_040_050_060 | +0.48 `[-1.00, 2.00]`, p=0.5708 | +1.52 `[-0.09, 3.08]`, p=0.0664 | +0.74 `[-0.74, 2.26]`, p=0.3664 |
| highsparse_055_060_065 | +1.43 `[0.00, 2.91]`, p=0.0536 | +2.60 `[1.13, 4.12]`, p=0.0004 | +1.95 `[0.48, 3.52]`, p=0.0112 |
| confirm_060_065 | +0.81 `[-0.52, 2.15]`, p=0.2320 | +2.02 `[0.72, 3.35]`, p=0.0040 | +1.01 `[-0.33, 2.41]`, p=0.1492 |

Key reading:

- `A_feature_grad` consistently beats `C_random` in non-smoke runs, and strictly does so in the high-sparsity and confirm runs.
- `A_feature_grad` does not strictly beat `B_wanda`; the strongest result was the high-sparsity run, where the CI lower bound was exactly `0.00` and p was `0.0536`.
- The focused confirm run did not strengthen the `A > B_wanda` result; it reduced the mean difference to `+0.81 pp` with CI crossing zero.

## Accuracy by Sparsity and Arm

### Smoke

Smoke is not used for scientific interpretation.

| Sparsity | A_feature | A_loss | B_wanda | C_random | D_plain |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.2500 | 0.3750 | 0.2500 | 0.3125 | 0.3125 |

### Full Candidate: 0.40 / 0.50 / 0.60

| Sparsity | A_feature | A_loss | B_wanda | C_random | D_plain |
|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.3594 | 0.3372 | 0.3594 | 0.3411 | 0.3516 |
| 0.50 | 0.3073 | 0.2969 | 0.3164 | 0.3151 | 0.3398 |
| 0.60 | 0.3203 | 0.2865 | 0.2969 | 0.2852 | 0.2734 |

Reading:

- The directional signal is concentrated at `0.60`.
- `0.50` works against `A_feature_grad`.
- Pooling all three sparsities dilutes the high-sparsity signal.

### High-Sparse: 0.55 / 0.60 / 0.65

| Sparsity | A_feature | A_loss | B_wanda | C_random | D_plain |
|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.2982 | 0.3177 | 0.2852 | 0.3151 | 0.3125 |
| 0.60 | 0.3177 | 0.2878 | 0.2969 | 0.2852 | 0.2734 |
| 0.65 | 0.3099 | 0.2982 | 0.3008 | 0.2474 | 0.2812 |

Reading:

- `0.60` and `0.65` support A over random/no-protection.
- `0.55` is mixed and hurts the aggregate comparison to random/no-protection.
- `A > B_wanda` is positive but borderline.

### Confirmation: 0.60 / 0.65, Larger Held-Out Set

| Sparsity | A_feature | A_loss | B_wanda | C_random | D_plain |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.3262 | 0.2917 | 0.3027 | 0.3079 | 0.3105 |
| 0.65 | 0.3092 | 0.2969 | 0.3164 | 0.2871 | 0.3047 |

Reading:

- `0.60` supports A over B/C/D.
- `0.65` does not support A over B_wanda.
- Larger test size confirms A over random but not A over Wanda.

## Diagnostic Summary

| Run | Dense Acc | Dense L0 | Dense Decoded Cosine | Dead Rate | A-Wanda Score Spearman | A-Wanda Mask Jaccard | A-C Mask Jaccard | Positive Causal vs Firing Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 0.5000 | 88.69 | 0.848 | 0.725 | 0.797 | 0.241 | 0.020 | 0.541 |
| full_040_050_060 | 0.3398 | 90.11 | 0.851 | 0.412 | 0.819 | 0.248 | 0.021 | 0.338 |
| highsparse_055_060_065 | 0.3398 | 90.11 | 0.851 | 0.412 | 0.819 | 0.221 | 0.017 | 0.338 |
| confirm_060_065 | 0.3633 | 90.11 | 0.851 | 0.412 | 0.819 | 0.217 | 0.016 | 0.338 |

Diagnostic interpretation:

- SAE sanity is acceptable for the matched 2B setup: L0 near `90`, decoded cosine near `0.851`.
- Smoke dead rate is high because the feature-stat sample is only 256 tokens; non-smoke runs use 2048 feature tokens and show lower dead rate.
- Mask construction is valid: all protected arms have equal budget; pairwise masks are non-identical; protected-pruned overlap is zero.
- `A_feature_grad` is not the same mask as `B_wanda`, but the score-level Spearman with Wanda is high at about `0.819`. This indicates the feature-gradient arm still contains strong geometry/Wanda structure.

## Task-Level Notes From Confirmation Run

Confirmation run dense scores:

| Task | Dense Accuracy | Dense Mean Margin |
|---|---:|---:|
| ARC-Easy | 0.3711 | -0.792 |
| HellaSwag | 0.3555 | -0.405 |

At `0.60`:

| Task | A_feature | B_wanda | C_random | D_plain |
|---|---:|---:|---:|---:|
| ARC-Easy | 0.3529 | 0.3320 | 0.3438 | 0.3359 |
| HellaSwag | 0.2995 | 0.2734 | 0.2721 | 0.2852 |

At `0.65`:

| Task | A_feature | B_wanda | C_random | D_plain |
|---|---:|---:|---:|---:|
| ARC-Easy | 0.3164 | 0.3125 | 0.3047 | 0.3398 |
| HellaSwag | 0.3021 | 0.3203 | 0.2695 | 0.2695 |

Task-level reading:

- At `0.60`, A improves over B/C/D on both ARC-Easy and HellaSwag.
- At `0.65`, A loses to B_wanda on HellaSwag and loses to D_plain on ARC-Easy.
- This explains why the focused confirmation run remains directional rather than strict.

## Statistical Interpretation

Overall confidence: CAUTION.

| Finding | Test | Value | Effect Size | Confidence |
|---|---|---:|---|---|
| A_feature vs C_random in high-sparse run | Paired bootstrap | +2.60 pp, CI `[1.13, 4.12]`, p=0.0004 | Small but consistent | SOLID for this internal metric |
| A_feature vs D_plain in high-sparse run | Paired bootstrap | +1.95 pp, CI `[0.48, 3.52]`, p=0.0112 | Small | CAUTION |
| A_feature vs B_wanda in high-sparse run | Paired bootstrap | +1.43 pp, CI `[0.00, 2.91]`, p=0.0536 | Borderline small | CAUTION |
| A_feature vs C_random in confirm run | Paired bootstrap | +2.02 pp, CI `[0.72, 3.35]`, p=0.0040 | Small but stable | SOLID for this internal metric |
| A_feature vs B_wanda in confirm run | Paired bootstrap | +0.81 pp, CI `[-0.52, 2.15]`, p=0.2320 | Small and uncertain | CAUTION |
| A_loss vs B_wanda in confirm run | Paired bootstrap | -1.53 pp, CI `[-2.80, -0.26]`, p=0.0160 | Small negative | NOTE: direct loss gradient is not a better baseline here |

Important statistical caveats:

- The W1 script reports bootstrap p-values and confidence intervals, but this report does not apply an additional multiple-comparison correction across all exploratory runs.
- The runs are sequential/adaptive: high-sparse and confirm were chosen after observing earlier results. Treat them as follow-up diagnostics, not a preregistered final test.
- Absolute dense accuracy is low under this W1 objective. Claims should be framed as relative within-run comparisons unless validated by lm-eval or Stage 1 capability metrics.

## Fallacy Scan

Coverage: 11/11 fallacy types checked.

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Simpson's Paradox | CAUTION | Aggregate W1 signal differs by sparsity. `0.60` is supportive; `0.50/0.55/0.65` are mixed. | Report per-sparsity results, not only pooled means. |
| Ecological Fallacy | NOTE | No individual-human inference is made; unit is model/example condition. | Not a main risk. |
| Berkson's Paradox | NOTE | No selected clinical/admission-style population. | Not applicable. |
| Collider Bias | NOTE | No regression controls introduced. | Not applicable. |
| Base Rate Neglect | NOTE | No diagnostic sensitivity/specificity framing. | Not applicable. |
| Regression to the Mean | NOTE | No extreme-group pre/post selection design. | Not applicable. |
| Survivorship Bias | NOTE | No dropout/attrition process. Failed smoke bugs were fixed before full runs and not selectively included as results. | Keep bug-fix history documented. |
| Look-Elsewhere Effect | CAUTION | Multiple sparsity grids were tried after observing earlier outputs. | Treat high-sparse/confirm as exploratory follow-ups. |
| Garden of Forking Paths | CAUTION | Experimental choices evolved during debugging and diagnosis. | Do not present this as preregistered confirmatory evidence. |
| Correlation != Causation | CAUTION | The intervention is causal at the pruning-mask level, but broad claims about feature fidelity causing downstream ability retention need stronger external validation. | Limit claims to observed intervention arms and internal metric. |
| Reverse Causality | NOTE | Not a correlational temporal claim. | Not applicable. |

## Reviewer-Facing Bottom Line

The W1 ability cross-layer experiment provides evidence that `A_feature_grad` is better than random at-risk protection under high sparsity. It does not yet prove that feature-fidelity-derived cross-layer saliency is better than a strong Wanda geometry rescue control.

Recommended wording:

```text
Stage 2 W1 shows a directional high-sparsity signal: causal SAE feature-gradient protection
outperforms random at-risk protection and sometimes no-protection Wanda, but it does not
robustly outperform the Wanda geometry rescue control. The current evidence supports
continued method refinement rather than a strict PASS.
```

Recommended next technical step:

- Add a new arm that explicitly reduces geometry coupling, e.g. `A_feature_grad_invfreq` or residualized score against Wanda.
- Keep the existing `A_feature_grad` arm unchanged as the registered baseline.
- Evaluate on the same `0.60/0.65` high-sparsity setting and then, if positive, validate with a standard lm-eval / Stage 1 capability metric.

## Artifact Index

Smoke:

```text
logs/stage2_w1_run.json
logs/stage2_w1_analyze.json
results/stage2_w1_ability_smoke.json
results/stage2_w1_ability_smoke/models.csv
results/stage2_w1_ability_smoke/ability_rows.csv
results/stage2_w1_ability_smoke/mask_diagnostics.json
results/stage2_w1_ability_smoke/seed0_diagnostics.json
```

Full candidate:

```text
logs/stage2_w1_run.json
logs/stage2_w1_analyze.json
results/stage2_w1_ability.json
results/stage2_w1_ability/models.csv
results/stage2_w1_ability/ability_rows.csv
results/stage2_w1_ability/mask_diagnostics.json
results/stage2_w1_ability/seed0_diagnostics.json
results/stage2_w1_ability/seed1_diagnostics.json
results/stage2_w1_ability/seed2_diagnostics.json
```

High-sparse:

```text
results/stage2_w1_ability_highsparse.json
results/stage2_w1_ability_highsparse/models.csv
results/stage2_w1_ability_highsparse/ability_rows.csv
results/stage2_w1_ability_highsparse/mask_diagnostics.json
results/stage2_w1_ability_highsparse/seed0_diagnostics.json
results/stage2_w1_ability_highsparse/seed1_diagnostics.json
results/stage2_w1_ability_highsparse/seed2_diagnostics.json
```

Confirmation:

```text
results/stage2_w1_ability_confirm_060_065.json
results/stage2_w1_ability_confirm_060_065/models.csv
results/stage2_w1_ability_confirm_060_065/ability_rows.csv
results/stage2_w1_ability_confirm_060_065/mask_diagnostics.json
results/stage2_w1_ability_confirm_060_065/seed0_diagnostics.json
results/stage2_w1_ability_confirm_060_065/seed1_diagnostics.json
results/stage2_w1_ability_confirm_060_065/seed2_diagnostics.json
```
