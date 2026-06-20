# FFAP Experiment Summary

Date: 2026-06-21

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED
- Version Label: ffap_experiment_summary_2026_06_21
- Data Source: local downloaded remote artifacts under `C:\Users\Yuhang\Downloads`, plus terminal excerpts provided during the run
- Local Action: data parsing and Markdown summarization only; no new remote experiment was launched

## Executive Summary

The experiment sequence reached three practical conclusions.

1. Stage 1 established the pruning phenomenon on `google/gemma-2-2b`: Wanda preserves WikiText PPL and SAE feature fidelity much better than local magnitude pruning at the same sparsity.
2. Stage 2 v2 correlation-style gates were unstable: one configuration looked like a `PASS_CANDIDATE`, but more constrained configurations collapsed to ties with geometry or PPL. This supported the decision to redesign Stage 2 rather than treat the correlation gate as final evidence.
3. Stage 2 v3 refusal measurement gate must stop at `INCONCLUSIVE_R0`: the matched Gemma Scope 9B-IT SAEs did not pass reconstruction compatibility, and diagnostics ruled out prompt-final denominator, wrapper normalization, HF pre/post hook mismatch, simple scale mismatch, `b_dec` input centering, and HF-vs-TransformerLens hook mismatch as sufficient fixes.

Current actionable state:

- Full v3 refusal causal/intervention/judge should not be run on current R0 artifacts.
- The ability-local signal from v2/earlier Stage 2 remains the only credible positive direction.
- The refusal branch needs a new measurement route, such as a different SAE source or a task-local refusal SAE.

## Data Inventory

Downloaded source files used in this summary:

| File | Role |
|---|---|
| `phase1_full_summary.csv/json` | Earlier Phase 1 / AAP-style baseline and Wanda sweep summary |
| `stage1_smoke.csv/json` | Stage 1 smoke result |
| `stage1_magnitude_sweep.csv` | Stage 1 local magnitude pruning sweep |
| `stage1_capability_eval.csv` | Stage 1 local magnitude downstream eval |
| `stage1_wanda_sweep.csv` | Stage 1 Wanda pruning sweep |
| `stage1_wanda_capability_eval.csv` | Stage 1 Wanda downstream eval |
| `stage1_ppl_vs_featuredamage.png` | Stage 1 visualization: PPL vs feature damage |
| `stage2_gate.csv/json` | Stage 2 v2 correlation gate, first configuration |
| `stage2_gate_top64_b8.csv/json` | Stage 2 v2 top64/batch8 configuration |
| `stage2_gate_top64_b8_l200.csv/json` | Stage 2 v2 top64/batch8/l200 configuration |
| `stage2_gate_top64_b8_l200_maxs05.csv/json` | Stage 2 v2 top64/batch8/l200/max-sparsity-0.5 configuration |
| `stage2_v3_smoke_bundle.tgz` | Stage 2 v3 smoke artifacts |
| `selected_layer.json` | Stage 2 v3 full R0 selected-layer artifact |
| `stage2_v3_layer_scan.json` | Stage 2 v3 full R0 layer scan log |
| `stage2_v3_r0_diagnostic.json` | Stage 2 v3 raw vs wrapped SAE diagnostic |
| `stage2_v3_r0_diagnostic_hookscale.json` | Stage 2 v3 pre/post hook and scale diagnostic |
| `stage2_v3_r0_diagnostic_bias.json` | Stage 2 v3 `b_dec` input-bias diagnostic |
| `stage2_v3_tl_alignment_diagnostic.json` | Stage 2 v3 TransformerLens alignment diagnostic |

Terminal excerpts used but not present as local downloaded files:

- Task 0 preflight and smoke outputs.
- Remote environment repair notes for PyTorch/CUDA.

## Remote Environment Notes

The final usable remote environment was the repaired `pbp` conda environment.

Known good state from the run:

| Component | State |
|---|---|
| Python | `/root/miniconda3/envs/pbp/bin/python`, Python 3.11.x |
| PyTorch | `2.12.0+cu130` after repair |
| CUDA | CUDA 13.0 path usable through PyTorch |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, about 95 GB VRAM |
| Important package versions reported earlier | `datasets 5.0.0`, `transformers 5.12.1`, `accelerate 1.14.0`, `lm-eval 0.4.12`, `numpy 2.4.4`, `scipy 1.17.1`, `sae-lens 6.44.3` |

Important run constraint:

- Downloads and caches were placed under `/root/autodl-tmp` or subdirectories.
- Hugging Face downloads were forced through HTTP by setting `HF_HUB_DISABLE_XET=1`.
- Local machine was used for static code checks and result summarization only.

## Task 0: Preflight And Smoke

### Purpose

Verify that the model, SAE, CUDA, and lm-eval pipeline can run before Stage 1.

### Key Events

Initial preflight failed when GPU was not enabled:

```text
status: FAIL_NO_CUDA
torch.cuda_available: false
nvidia-smi: Exec format error
```

After GPU and PyTorch environment repair, preflight and smoke were reported as passing.

### Smoke Result From Terminal Excerpt

| Item | Value |
|---|---:|
| Status | PASS |
| Model | `google/gemma-2-2b` |
| SAE | `gemma-scope-2b-pt-res-canonical`, `layer_12/width_16k/canonical` |
| Model load time | 12.567 s |
| Forward time | 0.503 s |
| Forward logits shape | `[1, 5, 256000]` |
| Peak allocated memory after SAE | 5.167 GiB |
| SAE activation shape | `[1, 5, 2304]` |
| SAE feature shape | `[5, 16384]` |
| SAE reconstruction MSE | 2615.803 |
| SAE L0 | 1068.0 |
| ARC-Easy smoke sample length | 20 |
| ARC-Easy acc | 1.0 |
| ARC-Easy acc_norm | 0.85 |

Interpretation:

- Task 0 verified plumbing only.
- The smoke sample was too small for scientific conclusions.

## Earlier Phase 1 / AAP-Style Summary

This section is separate from the Gemma/FFAP Stage 1-2 sequence. It is included because the result files were downloaded and are part of the broader experiment record.

Model family in this artifact: `Qwen2.5-7B-Instruct` with Wanda pruning.

### Phase 1 Summary Table

| Model | Role | Actual Sparsity | WikiText2 PPL | QA Avg4 | QA Avg4 Delta pp | BCR@0 | XSTest FPR | Unsafe Refusal Rate | IFEval Prompt Strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | baseline | 0.000 | 7.459 | 0.695 | 0.000 | n/a | 0.056 | 0.745 | 0.712 |
| wanda_0p30 | primary_phenomenon | 0.300 | 7.712 | 0.684 | -1.156 | 0.092 | 0.044 | 0.750 | 0.708 |
| wanda_0p40 | edge_region | 0.400 | 8.113 | 0.670 | -2.473 | 0.159 | 0.032 | 0.695 | 0.677 |
| wanda_0p50 | stress_region | 0.500 | 9.198 | 0.641 | -5.383 | 0.244 | 0.012 | 0.720 | 0.621 |

Phase 1 decision in the JSON:

```text
primary_model: wanda_0p30
candidates: [wanda_0p30]
supported_claim: At the primary sparsity point, preference-boundary damage is measurable while QA and instruction following are comparatively preserved.
next_step: Phase 2: test alignment-specificity against domain/task controls.
```

Interpretation:

- `wanda_0p30` was the main candidate: boundary damage was measurable while general QA and instruction following were comparatively preserved.
- Higher sparsities produced stronger degradation and served as stress regions.

## Stage 1: Gemma 2B Pruning Sweep

### Purpose

Build dense/local-magnitude/Wanda checkpoints and measure:

- WikiText-2 PPL.
- SAE active feature Jaccard vs dense.
- Decoded cosine damage.
- Downstream ability on ARC-Easy and HellaSwag.

### Stage 1 Smoke

Smoke status:

```text
PASS
```

Smoke conclusion:

```text
Stage 1 smoke completed for dense vs 20% local magnitude pruning on a small WikiText/SAE sample.
```

The remote also reported a saved local magnitude smoke checkpoint:

```text
outputs/stage1_smoke/local_magnitude_unstructured_s0.20
size: 5.0G
```

### PPL And Feature-Fidelity Sweep

Dense baseline:

| Method | Sparsity | PPL | Active Jaccard | Decoded Cos Delta | Feature L0 |
|---|---:|---:|---:|---:|---:|
| dense | 0.0 | 581.97 | n/a | n/a | 95.62 |

Local magnitude pruning:

| Sparsity | Actual Sparsity | PPL | PPL Relative Increase | Active Jaccard | Decoded Cos Delta | Feature L0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.200 | 668.71 | 0.149 | 0.804 | -0.007 | 82.55 |
| 0.3 | 0.301 | 1787.55 | 2.072 | 0.670 | -0.024 | 76.25 |
| 0.4 | 0.401 | 6680.44 | 10.479 | 0.601 | -0.054 | 74.02 |
| 0.5 | 0.501 | 48390.63 | 82.150 | 0.532 | -0.118 | 69.69 |
| 0.6 | 0.601 | 409697.51 | 702.984 | 0.517 | -0.155 | 103.89 |

Wanda pruning:

| Sparsity | Actual Sparsity | PPL | PPL Relative Increase | Active Jaccard | Decoded Cos Delta | Feature L0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.200 | 621.32 | 0.068 | 0.888 | -0.001 | 98.48 |
| 0.3 | 0.300 | 720.80 | 0.239 | 0.826 | -0.006 | 99.30 |
| 0.4 | 0.400 | 855.24 | 0.470 | 0.774 | -0.011 | 106.72 |
| 0.5 | 0.500 | 1165.33 | 1.002 | 0.729 | -0.021 | 119.43 |
| 0.6 | 0.600 | 1658.06 | 1.849 | 0.658 | -0.033 | 139.51 |

Stage 1 interpretation:

- Wanda dominated local magnitude on both PPL and feature fidelity.
- Local magnitude entered catastrophic PPL territory by 40-60% sparsity.
- Wanda degradation was much smoother: at 60% sparsity, PPL was about 1658 vs local magnitude about 409698.

### Stage 1 Capability Eval

Sample size: 50 per task.

Dense baseline:

| Task | Accuracy |
|---|---:|
| ARC-Easy | 0.82 |
| HellaSwag | 0.52 |

Local magnitude:

| Sparsity | ARC-Easy Acc | HellaSwag Acc |
|---:|---:|---:|
| 0.2 | 0.78 | 0.50 |
| 0.3 | 0.82 | 0.52 |
| 0.4 | 0.68 | 0.46 |
| 0.5 | 0.50 | 0.44 |
| 0.6 | 0.32 | 0.30 |

Wanda:

| Sparsity | ARC-Easy Acc | HellaSwag Acc |
|---:|---:|---:|
| 0.2 | 0.84 | 0.52 |
| 0.3 | 0.82 | 0.52 |
| 0.4 | 0.84 | 0.52 |
| 0.5 | 0.68 | 0.46 |
| 0.6 | 0.56 | 0.36 |

Interpretation:

- Wanda preserved sampled ability better than local magnitude at the same sparsity.
- Because sample length was only 50 per task, these are directional rather than final benchmark numbers.

## Stage 2 v2: Correlation Gate

### Purpose

Test whether causal-weighted feature fidelity predicts post-pruning degradation better than geometry-only feature metrics and PPL.

This was still a correlation-style gate and was later judged too fragile.

### First Gate Configuration

File: `stage2_gate.json`

Status:

```text
PASS_CANDIDATE
```

Rows: 10 checkpoint conditions.

Key Spearman correlations:

| Predictor | Spearman rho | p-value |
|---|---:|---:|
| causal_weighted_firing_rate_l1 | 0.890 | 0.000 |
| abs_causal_weighted_mean_activation_l1 | 0.870 | 0.000 |
| selected_firing_rate_l1 | 0.835 | 0.000 |
| abs_causal_weighted_firing_rate_l1 | 0.835 | 0.000 |
| selected_mean_activation_l1 | 0.820 | 0.000 |
| causal_weighted_mean_activation_l1 | 0.820 | 0.000 |
| ppl_relative_increase | 0.713 | 0.021 |

Interpretation at the time:

- Causal-weighted firing-rate L1 was higher than best geometry and PPL.
- This was only a candidate pass because `n=10` and because correlation-style gates were known to be fragile.

### Top64 / Batch8 Configuration

File: `stage2_gate_top64_b8.json`

Status:

```text
FAIL_OR_REVISE_CANDIDATE
```

Rows: 10.

| Predictor | Spearman rho | p-value |
|---|---:|---:|
| selected_mean_activation_l1 | 0.835 | 0.003 |
| selected_firing_rate_l1 | 0.835 | 0.003 |
| abs_causal_weighted_mean_activation_l1 | 0.835 | 0.003 |
| abs_causal_weighted_firing_rate_l1 | 0.835 | 0.003 |
| causal_weighted_mean_activation_l1 | 0.774 | 0.009 |
| causal_weighted_firing_rate_l1 | 0.750 | 0.012 |
| ppl_relative_increase | 0.713 | 0.021 |

Interpretation:

- Best causal-weighted metric tied geometry rather than exceeding it.
- This exposed the numerical-air-gap problem: causal weighting did not robustly separate from geometry.

### Top64 / Batch8 / L200 Configuration

File: `stage2_gate_top64_b8_l200.json`

Status:

```text
FAIL_OR_REVISE_CANDIDATE
```

Rows: 10.

| Predictor | Spearman rho | p-value |
|---|---:|---:|
| selected_mean_activation_l1 | 0.825 | 0.003 |
| selected_firing_rate_l1 | 0.825 | 0.003 |
| abs_causal_weighted_mean_activation_l1 | 0.825 | 0.003 |
| abs_causal_weighted_firing_rate_l1 | 0.825 | 0.003 |
| ppl_relative_increase | 0.775 | 0.008 |
| causal_weighted_mean_activation_l1 | 0.744 | 0.014 |
| causal_weighted_firing_rate_l1 | 0.700 | 0.024 |

Interpretation:

- Again, causal-weighted metrics tied geometry or fell below PPL.
- This reinforced that the v2 correlation gate was not a robust validation mechanism.

### Top64 / Batch8 / L200 / Max Sparsity 0.5

File: `stage2_gate_top64_b8_l200_maxs05.json`

Status:

```text
FAIL_OR_REVISE_CANDIDATE
```

Rows: 8.

| Predictor | Spearman rho | p-value |
|---|---:|---:|
| ppl_relative_increase | 0.786 | 0.021 |
| selected_mean_activation_l1 | 0.723 | 0.043 |
| selected_firing_rate_l1 | 0.723 | 0.043 |
| causal_weighted_mean_activation_l1 | 0.723 | 0.043 |
| abs_causal_weighted_mean_activation_l1 | 0.723 | 0.043 |
| abs_causal_weighted_firing_rate_l1 | 0.723 | 0.043 |
| causal_weighted_firing_rate_l1 | 0.634 | 0.091 |

Interpretation:

- PPL was strongest in this restricted setting.
- Causal and geometry metrics were indistinguishable in several cases.

### Stage 2 v2 Takeaway

The v2 gate should be treated as a diagnostic, not a paper-grade causal result.

Main failure modes:

- `n=10` or `n=8` is underpowered for comparing close correlations.
- Several causal-weighted predictors numerically collapsed onto geometry predictors.
- The causal scores were measured on mismatched objectives/data and then linearly aggregated to per-model metrics.

This justified moving to Stage 2 v3: a local held-out gate with task-matched refusal measurement, seven intervention arms, and human/judge audit infrastructure.

## Stage 2 v3: Smoke Run

### Purpose

Validate the full v3 pipeline mechanically before running an expensive scientific gate.

Configuration:

- Model: `google/gemma-2-9b-it`
- SAE release: `gemma-scope-9b-it-res-canonical`
- Candidate SAEs:
  - `layer_9/width_16k/canonical`
  - `layer_20/width_16k/canonical`
  - `layer_31/width_16k/canonical`
- Data:
  - ARC-Easy / HellaSwag for ability.
  - AdvBench harmful prompts.
  - XSTest safe prompts.
- Split seed: `20260620`
- Experimental seeds: `0,1,2`

Smoke status by step:

| Step | Status | Conclusion |
|---|---|---|
| prefetch | PASS | 9B-IT, three matched IT SAEs, AdvBench, and XSTest cached |
| prepare | PASS | Calibration/dev/final-test IDs fixed and disjoint |
| layer-scan | PASS via smoke override | Layer 31 selected for plumbing only, not R0 evidence |
| causal | PASS | Ability and prompt-final refusal causal scores saved for every seed |
| intervention | PASS | Local intervention completed without saving pruned checkpoints |
| judge | PASS | Every response received cached or fresh judge label |
| manual-export | WAITING_HUMAN_LABELS | Two blinded annotation sheets exported; analyze blocked until labels complete |

Smoke selected layer:

```text
layer: 31
smoke_override: true
```

Mask sanity check:

| Condition | Result |
|---|---|
| Arms | 7 local arms |
| Sparsity | 0.5 |
| Per-arm protected count | 1,321,205 for every arm |
| A/B/C masks identical | false for all checked pairs |
| Random arm Jaccard vs causal/geometry arms | about 0.020 |

Interpretation:

- Smoke validated pipeline wiring, judge/cache behavior, response generation, mask-budget equality, and mask contrast.
- Smoke numbers are not scientific evidence because the layer was selected by override and sample sizes were intentionally small.

## Stage 2 v3: Full R0 Layer Scan

### Purpose

Before running v3 causal/intervention, verify that a matched IT SAE layer is usable for refusal measurement.

Eligibility required both:

1. SAE reconstruction compatibility:
   - decoded cosine >= 0.85
   - explained variance >= 0.75
2. Directional mediation:
   - harmful direction subtraction lowers harmful refusal margin with CI below 0
   - benign direction addition raises benign over-refusal margin with CI above 0

### Full R0 Result

Status:

```text
INCONCLUSIVE_R0
```

Conclusion:

```text
No candidate layer passed both matched-SAE transfer and directional mediation gates.
```

Candidate table from `selected_layer.json`:

| Layer | Eligible | Compat Pass | Mediation Pass | EV | Cosine | L0 | Dead Rate | Harmful Subtract Mean | Harmful CI |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 9 | false | false | false | 0.321 | 0.945 | 63.5 | 0.905 | +0.025 | [0.016, 0.033] |
| 20 | false | false | false | 0.209 | 0.942 | 90.0 | 0.868 | +0.072 | [0.054, 0.092] |
| 31 | false | false | true | 0.126 | 0.877 | 105.8 | 0.841 | -0.833 | [-0.865, -0.801] |

Interpretation:

- Layer 31 had strong directional mediation.
- No layer passed SAE reconstruction compatibility.
- Therefore v3 correctly stopped before causal/intervention/judge.

## Stage 2 v3: R0 Diagnostics

### Diagnostic 1: Raw vs Wrapped SAE / Token-Level EV

File: `stage2_v3_r0_diagnostic.json`

Status:

```text
PASS
```

Finding:

- Raw and wrapped SAE paths were numerically identical.
- Wrapper did not fix reconstruction.
- Token-level EV was still negative or very poor.

Representative raw token-level metrics:

| Layer | Hook | Mode | EV | Cosine | L0 | Dead Rate | Norm Ratio | Optimal-Scaled EV |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 9 | post | raw | -7.33 | 0.90 | 330.5 | 0.06 | 1.69 | 0.12 |
| 20 | post | raw | -1.16 | 0.91 | 269.4 | 0.09 | 1.27 | 0.39 |
| 31 | post | raw | -3.86 | 0.92 | 345.7 | 0.11 | 1.55 | 0.28 |

Conclusion:

- Prompt-final-only EV was not the sole issue.
- Wrapper normalization was not the repair path.

### Diagnostic 2: Hook Point And Scale Scan

File: `stage2_v3_r0_diagnostic_hookscale.json`

Status:

```text
PASS
```

Key findings:

- `post` was consistently slightly better than `pre`.
- SAE metadata reported `hook_name = blocks.N.hook_resid_post`.
- Input scale scan could not recover acceptable EV.

Best scale-scan rows:

| Layer | Hook | Scale | EV | Cosine | L0 | Dead Rate | Optimal-Scaled EV |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | post | 0.125 | 0.18 | 0.53 | 31.0 | 0.94 | 0.57 |
| 20 | pre | 0.125 | 0.11 | 0.50 | 31.1 | 0.94 | 0.55 |
| 31 | post | 0.125 | -0.23 | 0.51 | 62.7 | 0.83 | 0.50 |

Conclusion:

- Not a simple pre/post mismatch.
- Not a simple global input scale mismatch.

### Diagnostic 3: `b_dec` Input Bias Mode

File: `stage2_v3_r0_diagnostic_bias.json`

Status:

```text
PASS
```

Finding:

- Forcing `apply_b_dec_to_input=True` made results worse.

Representative comparison:

| Layer | Hook | Bias Mode | EV | Cosine | L0 | Dead Rate | Optimal-Scaled EV |
|---:|---|---|---:|---:|---:|---:|---:|
| 9 | post | cfg | -7.33 | 0.90 | 330.5 | 0.06 | 0.12 |
| 9 | post | force_subtract | -9.97 | 0.74 | 567.3 | 0.02 | 0.09 |
| 20 | post | cfg | -1.16 | 0.91 | 269.4 | 0.09 | 0.39 |
| 20 | post | force_subtract | -2.10 | 0.67 | 613.6 | 0.01 | 0.28 |
| 31 | post | cfg | -3.86 | 0.92 | 345.7 | 0.11 | 0.28 |
| 31 | post | force_subtract | -4.75 | 0.68 | 535.6 | 0.02 | 0.23 |

Conclusion:

- `b_dec` input-centering is not the fix.

### Diagnostic 4: TransformerLens Alignment

File: `stage2_v3_tl_alignment_diagnostic.json`

Status:

```text
PASS
```

TransformerLens model summary:

| Field | Value |
|---|---|
| model_name | `gemma-2-9b-it` |
| original_architecture | `Gemma2ForCausalLM` |
| n_layers | 42 |
| d_model | 3584 |
| normalization_type | RMS |
| dtype | `torch.bfloat16` |
| device | cuda |
| default_prepend_bos | true |

Metrics:

| Layer | TL Hook | EV | Cosine | L0 | Dead Rate | Norm Ratio | Optimal-Scaled EV |
|---:|---|---:|---:|---:|---:|---:|---:|
| 9 | `blocks.9.hook_resid_post` | -7.32 | 0.90 | 329.8 | 0.10 | 1.69 | 0.12 |
| 20 | `blocks.20.hook_resid_post` | -1.15 | 0.91 | 269.3 | 0.13 | 1.27 | 0.40 |
| 31 | `blocks.31.hook_resid_post` | -3.86 | 0.93 | 345.7 | 0.16 | 1.55 | 0.28 |

Conclusion:

- TransformerLens official `hook_resid_post` produced essentially the same bad reconstruction as the HF hook.
- Therefore HF hook alignment is not the source of failure.

## Consolidated Interpretation

### What Is Supported

- Stage 1 supports that Wanda is a much stronger pruning baseline than local magnitude for preserving PPL and SAE feature fidelity on Gemma 2B.
- Stage 2 v2 supports that the simple correlation gate is fragile and not sufficient as proof.
- Stage 2 v3 smoke supports that the new seven-arm local intervention pipeline is mechanically runnable.
- Stage 2 v3 full R0 supports that current matched 9B-IT SAE reconstruction compatibility is insufficient for refusal-feature measurement.

### What Is Not Supported

- No valid Stage 2 v3 refusal PASS.
- No valid Stage 2 v3 safety REFRAME.
- No valid Stage 2 v3 FAIL of the FFAP hypothesis.
- No claim that refusal features do not exist.
- No claim that causal feature fidelity cannot work.

### Current Best Scientific Reading

The current refusal branch is blocked by measurement validity, not by a negative intervention result.

The correct gate status is:

```text
INCONCLUSIVE_R0
```

The reason:

```text
Matched Gemma Scope 9B-IT residual SAEs did not pass reconstruction compatibility for the current v3 local refusal measurement setup.
```

## Reproducibility And Quality Notes

What is strong:

- Results are backed by saved CSV/JSON artifacts.
- Stage 2 v3 smoke verified mask budget equality and mask contrast.
- Full v3 did not continue past a failed R0 gate.
- Multiple plausible implementation explanations were explicitly tested.

What remains weak:

- Stage 2 v2 correlations used small `n` and should not be treated as final evidence.
- Stage 1 capability evals used small sample size, 50 per task.
- Stage 2 v3 refusal branch did not reach full intervention/judge/analyze due R0.
- Task 0 details are partly from terminal excerpts rather than a local downloaded artifact.

## Recommended Next Actions

### Conservative Path

Stop the v3 refusal branch and return to ability/W1 mainline.

Rationale:

- R0 is a pre-registered stop condition.
- Continuing would produce uninterpretable refusal results.

### Measurement-Rebuild Path

If refusal remains central, rebuild R0 with a different measurement substrate:

1. Search other Gemma Scope IT SAE widths/layers/releases and require token-level EV/L0 sanity before causal attribution.
2. Train a task-local refusal SAE on the exact activation distribution used by v3.
3. Use a different model/SAE pair whose official reconstruction compatibility is already verified.

### Appendix Path

Keep layer 31 directional mediation as a residual-stream diagnostic only.

Do not treat it as SAE-feature causal fidelity evidence unless a compatible SAE is found.

## Final Status Table

| Stage | Status | Scientific Use |
|---|---|---|
| Task 0 preflight/smoke | PASS after environment repair | Plumbing only |
| Earlier Phase 1 / AAP-style run | Primary candidate `wanda_0p30` | Separate supporting background |
| Stage 1 Gemma 2B pruning sweep | PASS | Establishes Wanda vs local magnitude behavior |
| Stage 2 v2 correlation gate | Mixed / fragile | Diagnostic only; motivated redesign |
| Stage 2 v3 smoke | PASS / WAITING_HUMAN_LABELS at manual export | Plumbing only |
| Stage 2 v3 full R0 | INCONCLUSIVE_R0 | Stop condition |
| Stage 2 v3 full causal/intervention/judge | Not run | Correctly blocked by R0 |
