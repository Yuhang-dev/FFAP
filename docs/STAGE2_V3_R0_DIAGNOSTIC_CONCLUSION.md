# Stage 2 v3 R0 Diagnostic Conclusion

Date: 2026-06-21

## Verdict

Stage 2 v3 must remain stopped at `INCONCLUSIVE_R0`.

The current evidence does not support running v3 causal, intervention, judge, or manual-audit stages. The failure is not a PASS/REFRAME/FAIL result for FFAP. It is a refusal-measurement gate failure: the matched Gemma Scope 9B-IT canonical residual SAEs did not pass the reconstruction compatibility requirement on the tested layer/token settings.

## Decision

Do not proceed to Stage 2 v3 causal/intervention on the current R0 artifacts.

The strongest interpretation is:

- Layer 31 still shows refusal-direction mediation in the original layer scan.
- However, no candidate SAE has acceptable reconstruction compatibility.
- The mismatch persists under HF hooks and TransformerLens `hook_resid_post`.
- Therefore the current matched IT canonical SAE path is blocked for the v3 local refusal measurement gate.

This is not evidence that refusal features do not exist, and not evidence against the pruning method. It means the v3 refusal target is not yet measurable with this SAE setup.

The diagnostics below rule out the cheap fixes. They do not fully identify the root cause inside the `gemma-scope-9b-it-res-canonical` route. The leading remaining suspicion is release-specific activation-definition or loader alignment, not a simple implementation mistake in the current HF/TL hook collection.

## Inputs Reviewed

Downloaded files:

- `stage2_v3_layer_scan.json`
- `stage2_v3_r0_diagnostic.json`
- `stage2_v3_r0_diagnostic_hookscale.json`
- `stage2_v3_r0_diagnostic_bias.json`
- `stage2_v3_tl_alignment_diagnostic.json`

Relevant scripts:

- `scripts/stage2_v3_r0_diagnostic.py`
- `scripts/stage2_v3_tl_alignment_diagnostic.py`
- `ffap/stage2_v3/causal.py`

## R0 Diagnostic Chain

### 1. Prompt-final-only EV was not the full explanation

The original R0 scan computed SAE compatibility on prompt-final activations. That was a plausible source of low EV, so a token-level diagnostic was added.

Result: token-level EV remained poor.

Representative token-level results with raw SAE encode/decode:

| Layer | Hook | EV | Cosine | L0 | Dead Rate | Optimal-Scaled EV |
|---:|---|---:|---:|---:|---:|---:|
| 9 | post | -7.33 | 0.90 | 330.5 | 0.06 | 0.12 |
| 20 | post | -1.16 | 0.91 | 269.4 | 0.09 | 0.39 |
| 31 | post | -3.86 | 0.92 | 345.7 | 0.11 | 0.28 |

The high cosine and poor EV indicate reconstructions point in roughly similar directions but have incompatible magnitude/structure for the hard compatibility gate.

### 2. Runtime normalization wrapper did not help

Raw and wrapped SAE paths produced identical metrics in the first diagnostic.

Conclusion: the wrapper is not a repair path for this failure.

### 3. Pre/post HF hook mismatch was not the cause

The diagnostic compared:

- HF layer forward output, corresponding to post-block residual.
- HF layer forward pre-hook, corresponding to pre-block residual.

Result: post was consistently slightly better than pre. The SAE metadata also reports:

- `hook_name = blocks.N.hook_resid_post`
- `model_name = gemma-2-9b-it`

Conclusion: the failure is not explained by accidentally using pre-residual instead of post-residual.

### 4. Global input scale was not sufficient

Input scale scan tested `decode(encode(alpha*x)) / alpha`.

Best raw EV observed:

| Layer | Hook | Bias Mode | Scale | EV | Cosine | L0 | Optimal-Scaled EV |
|---:|---|---|---:|---:|---:|---:|---:|
| 20 | post | cfg | 0.125 | 0.18 | 0.53 | 31.0 | 0.57 |
| 20 | pre | cfg | 0.125 | 0.11 | 0.50 | 31.1 | 0.55 |
| 31 | post | cfg | 0.125 | -0.23 | 0.51 | 62.7 | 0.50 |

No scale setting approached the R0 compatibility threshold `EV >= 0.75`.

Conclusion: this is not a simple global residual scale mismatch.

### 5. `b_dec` input-centering was not the cause

The bias diagnostic compared:

- `cfg`
- `force_subtract`, which forces `apply_b_dec_to_input=True`

`force_subtract` was consistently worse:

| Layer | Hook | Bias Mode | EV | Cosine | L0 | Dead Rate | Optimal-Scaled EV |
|---:|---|---|---:|---:|---:|---:|---:|
| 9 | post | cfg | -7.33 | 0.90 | 330.5 | 0.06 | 0.12 |
| 9 | post | force_subtract | -9.97 | 0.74 | 567.3 | 0.02 | 0.09 |
| 20 | post | cfg | -1.16 | 0.91 | 269.4 | 0.09 | 0.39 |
| 20 | post | force_subtract | -2.10 | 0.67 | 613.6 | 0.01 | 0.28 |
| 31 | post | cfg | -3.86 | 0.92 | 345.7 | 0.11 | 0.28 |
| 31 | post | force_subtract | -4.75 | 0.68 | 535.6 | 0.02 | 0.23 |

Conclusion: forcing decoder-bias centering is not a valid fix.

### 6. TransformerLens `hook_resid_post` did not fix reconstruction

The TransformerLens diagnostic collected activations from the official hook names:

- `blocks.9.hook_resid_post`
- `blocks.20.hook_resid_post`
- `blocks.31.hook_resid_post`

TransformerLens model summary:

- `model_name = gemma-2-9b-it`
- `original_architecture = Gemma2ForCausalLM`
- `d_model = 3584`
- `n_layers = 42`
- `default_prepend_bos = True`

Results:

| Layer | Hook | EV | Cosine | L0 | Dead Rate | Optimal-Scaled EV |
|---:|---|---:|---:|---:|---:|---:|
| 9 | `blocks.9.hook_resid_post` | -7.32 | 0.90 | 329.8 | 0.10 | 0.12 |
| 20 | `blocks.20.hook_resid_post` | -1.15 | 0.91 | 269.3 | 0.13 | 0.40 |
| 31 | `blocks.31.hook_resid_post` | -3.86 | 0.93 | 345.7 | 0.16 | 0.28 |

These match the HF-hook diagnostics closely.

Conclusion: the failure is not caused by HF forward hooks being misaligned with TransformerLens `hook_resid_post`.

## Final Interpretation

The following implementation explanations have been tested and ruled out:

- Prompt-final-only EV denominator as the sole issue.
- Missing runtime normalization wrapper.
- HF pre/post hook mismatch.
- Simple global residual scale mismatch.
- `b_dec` input-centering mismatch.
- HF hook implementation mismatch relative to TransformerLens `hook_resid_post`.

The remaining defensible conclusion is narrower: the selected `gemma-scope-9b-it-res-canonical` route does not pass the v3 R0 reconstruction compatibility gate for this local refusal-measurement setup, after ruling out the cheap implementation fixes above. The exact source of the mismatch remains unresolved within that release/loader/activation-definition path.

## Consequences For The Experiment

Stage 2 v3 should output or be treated as:

```text
INCONCLUSIVE_R0
```

Do not interpret this as:

- `PASS`
- `REFRAME`
- `FAIL`
- evidence that refusal causality is absent
- evidence that FFAP cannot work

Interpret it as:

```text
Matched-SAE transfer/reconstruction compatibility is insufficient for the current refusal measurement gate; cheap implementation fixes were ruled out, but the exact matched-IT canonical release mismatch remains unidentified.
```

## Recommended Next Paths

### Path A: Stop v3 refusal branch for now

This is the conservative default.

The ability-local result from v2 remains the only credible positive signal. The refusal branch remains unclaimed.

### Path B: Replace the SAE source for refusal

Options:

- Search for a different Gemma Scope IT SAE width/layer/release with acceptable token-level EV.
- Train a task-local refusal SAE on the exact activation distribution used by v3.
- Use a different model/SAE pair where official reconstruction compatibility is already verified.

This is a measurement rebuild, not a small patch.

### Path C: Continue only as a diagnostic appendix

Layer 31 directional mediation can be discussed as a residual-stream direction diagnostic, but not as SAE-feature causal fidelity evidence. It should not drive the v3 intervention gate.

## Pre-Registered Gate Status

The pre-registered v3 requirement was:

```text
No eligible layer -> INCONCLUSIVE_R0 -> stop.
```

That condition is now met.

The correct next action is to stop and ask for human confirmation before designing a new refusal-measurement route.
