# Stage 2 v3 R0 Review Brief

## Purpose

This note summarizes the current Stage 2 v3 experiment state in FFAP for external review. The goal is to determine whether the current `INCONCLUSIVE_R0` result reflects a real failure of matched IT SAE suitability, or a measurement/implementation issue in the R0 layer scan.

The current run has not produced a scientific PASS/REFRAME/FAIL verdict. It stopped before full causal/intervention evaluation, as intended by the pre-registered Stage 2 v3 gate.

## Project Context

The FFAP project tests whether causally validated SAE feature fidelity can serve as a better pruning/quantization saliency target than PPL or geometry baselines, especially for safety/alignment boundary behavior.

Stage 2 v2 exposed several refusal-measurement problems:

- The previous refusal line used a base-model SAE on an instruction-tuned model.
- The refusal outcome was closer to teacher-forced fixed-text preference than true refusal behavior.
- Continuation tokens leaked into attribution.
- Refusal causality was hard to distinguish from firing-rate geometry.

Stage 2 v3 was implemented to fix these measurement issues before moving to whole-model methods.

## Stage 2 v3 Design Summary

Stage 2 v3 is a held-out local measurement gate. It intentionally leaves v1/v2 unchanged and does not start W1 cross-layer importance or Stage 3.

Key changes:

- Model: `google/gemma-2-9b-it`
- SAE release: `gemma-scope-9b-it-res-canonical`
- Candidate SAEs:
  - `layer_9/width_16k/canonical`
  - `layer_20/width_16k/canonical`
  - `layer_31/width_16k/canonical`
- Data isolation:
  - Ability calibration/dev/test split is fixed and disjoint.
  - AdvBench harmful prompts are split into calibration/dev/test.
  - XSTest safe prompts are split into calibration/dev/test.
  - Split seed is fixed at `20260620`.
- Experimental seeds are fixed at `0,1,2`.
- Refusal attribution is prompt-final only, using a refusal-prefix vs compliance-prefix margin.
- SAE error node is preserved during intervention/ablation.
- Mean/resample ablation are primary; zero is retained only as robustness.
- Safety outcome uses generated responses judged through the StrongREJECT-style rubric and DeepSeek API.
- The full gate requires manual audit before final analysis.

Relevant commits:

- `df60565 Add Stage 2 v3 causal validation gate`
- `455812f Let Stage 2 v3 smoke bypass strict R0 gate`

## Smoke Run Result

Smoke was run only to validate plumbing. It is not scientific evidence.

Smoke completed end-to-end:

- `prefetch`: PASS
- `prepare`: PASS
- `layer-scan`: PASS via smoke override only
- `causal`: PASS
- `intervention`: PASS
- `judge`: PASS
- `manual-export`: `WAITING_HUMAN_LABELS`

Smoke output checks:

- `384/384` generated responses were judged.
- Judge failures: `0`
- Seven local arms were present:
  - `A_ability_causal`
  - `B_ability_geometry`
  - `A_refusal_causal`
  - `B_refusal_geometry`
  - `A_joint_causal`
  - `B_joint_geometry`
  - `C_random`
- Dense control was present.
- Equal protection budget was verified across arms.
- Actual sparsity was `0.5`.
- `protected_pruned_overlap=0`.
- No A/B/C protection masks were identical.

Important smoke caveat:

Smoke selected layer 31 only by `smoke_override=true`, because smoke sample sizes are too small for R0. This override exists only so downstream plumbing can be checked.

## Full R0 Layer Scan Result

Full `prepare` and `layer-scan` were run. The run stopped at R0:

```text
status: INCONCLUSIVE_R0
smoke_override: False
selected_layer: None
conclusion: No candidate layer passed both matched-SAE transfer and directional mediation gates.
```

The full run did not proceed to causal/intervention/judge, and should not proceed until R0 is understood.

Candidate summary:

```text
layer 9:
  eligible: false
  compatibility_pass: false
  mediation_pass: false
  decoded cosine: 0.94
  explained variance: 0.32
  L0: 63.5
  dead-feature rate: 0.91
  harmful direction subtraction mean: +0.02
  harmful CI: [0.016, 0.033]

layer 20:
  eligible: false
  compatibility_pass: false
  mediation_pass: false
  decoded cosine: 0.94
  explained variance: 0.21
  L0: 90.0
  dead-feature rate: 0.87
  harmful direction subtraction mean: +0.07
  harmful CI: [0.054, 0.092]

layer 31:
  eligible: false
  compatibility_pass: false
  mediation_pass: true
  decoded cosine: 0.88
  explained variance: 0.13
  L0: 105.8
  dead-feature rate: 0.84
  harmful direction subtraction mean: -0.83
  harmful CI: [-0.865, -0.801]
```

Interpretation:

- Layer 31 appears to have a refusal-relevant direction by the directional intervention criterion.
- However, none of the matched IT SAEs pass the current reconstruction compatibility gate.
- The primary failing metric is explained variance, not cosine.
- Therefore the current result is `INCONCLUSIVE_R0`, not a scientific failure of the pruning method.

## Current R0 Gate Definition

A candidate layer is eligible only if both conditions pass:

1. SAE compatibility:
   - decoded cosine >= `0.85`
   - explained variance >= `0.75`
   - L0 far from canonical only warns; it does not by itself fail.
2. Directional mediation:
   - subtracting harmful-vs-benign refusal direction from harmful prompt-final residual lowers refusal margin with paired CI below 0.
   - adding the direction to benign prompt-final residual raises over-refusal margin with paired CI above 0.

In the full run, layer 31 passes mediation but fails compatibility.

## Main Question For Review

Is the low explained variance a real indication that the matched IT SAE is unsuitable for prompt-final refusal measurement, or is the EV measurement currently wrong/mis-specified?

The suspicious pattern is:

```text
decoded cosine is high: 0.88-0.94
explained variance is low: 0.13-0.32
```

This can happen legitimately, but it is also compatible with a measurement bug or an inappropriate EV denominator.

## Most Important Checks Requested

Please review the R0 reconstruction/compatibility implementation, especially:

1. SAE Lens normalization
   - Are we calling `sae.encode()` / `sae.decode()` in a way that respects Gemma Scope SAE runtime normalization?
   - Does SAE Lens expect `sae.forward()` or explicit activation normalization hooks for correct reconstruction metrics?
   - Could raw residual activations require `run_time_activation_norm_fn_in/out` handling?

2. Prompt-final EV denominator
   - EV is currently computed on prompt-final residuals only.
   - The denominator is variance around the mean of the sampled prompt-final activations.
   - Is this too harsh or unstable for instruction prompts compared with token-level residual EV?

3. Layer and hook point
   - Are we capturing the correct residual stream location for these Gemma Scope IT residual SAEs?
   - Are the SAEs trained on the same hook point as `model.model.layers[layer]` forward output?
   - Could pre/post block residual mismatch explain high cosine but low EV?

4. Data distribution
   - Compatibility is measured on harmful/benign prompt-final instruction prompts.
   - Are these prompts too narrow or too far from SAE training distribution?
   - Should compatibility be checked on a broader instruction mixture or token-level corpus first?

5. Gate threshold
   - Is EV >= 0.75 appropriate for prompt-final instruction-token activations?
   - Should the hard threshold be applied to token-level EV, with prompt-final EV reported separately?

6. Directional mediation target
   - Layer 31 shows strong directional mediation.
   - Does this justify retaining layer 31 for a diagnostic branch even if SAE EV fails?
   - Or should R0 remain strict and stop until reconstruction is fixed?

## Relevant Local Files

New v3 implementation:

- `ffap/stage2_v3/config.py`
- `ffap/stage2_v3/data.py`
- `ffap/stage2_v3/causal.py`
- `ffap/stage2_v3/pipeline.py`
- `ffap/stage2_v3/judge.py`
- `ffap/stage2_v3/statistics.py`
- `scripts/stage2_gate_causal_v3.py`
- `scripts/stage2_v3_prefetch.py`
- `remote/run_stage2_gate_causal_v3.sh`
- `remote/prefetch_stage2_v3_assets.sh`

Most relevant function for this review:

- `ffap/stage2_v3/causal.py`
  - `prompt_feature_metrics`
  - `collect_prompt_hidden`
  - `run_layer_scan`
  - `refusal_decision_margins`
  - `refusal_attribution_scores`

## Relevant Result Files

Smoke:

- `stage2_v3_smoke_bundle.tgz`
- `results/stage2_v3_smoke/selected_layer.json`
- `results/stage2_v3_smoke/models.csv`
- `results/stage2_v3_smoke/responses.jsonl`
- `results/stage2_v3_smoke/responses_judged.jsonl`
- `results/stage2_v3_smoke/mask_diagnostics.json`

Full R0:

- `stage2_v3_layer_scan.json`
- `selected_layer.json`

## Current Recommended Next Step

Do not run full causal/intervention yet.

Before relaxing R0 or declaring the IT SAE unsuitable, add a small R0 diagnostic script/step that compares:

1. raw `sae.encode()` + `sae.decode()` reconstruction
2. SAE Lens `sae.forward()` reconstruction, if it applies runtime normalization differently
3. prompt-final EV
4. token-level EV on broader instruction/calibration text
5. hook-point variants if available

Only after this diagnostic should we decide whether to:

- fix SAE normalization/hook handling,
- change EV measurement from prompt-final to token-level,
- relax EV threshold with justification,
- select layer 31 for a diagnostic branch,
- or mark R0 genuinely inconclusive and revise the refusal SAE plan.

## Current Scientific Status

Current status:

```text
INCONCLUSIVE_R0
```

This means:

- The v3 pipeline is operational.
- The safety/refusal outcome pipeline can generate, judge, cache, and export manual audit rows.
- The local seven-arm intervention machinery passes smoke checks.
- The full experiment is blocked at the matched IT SAE R0 compatibility gate.

This does not yet support PASS, REFRAME, or FAIL for the main FFAP scientific claim.
