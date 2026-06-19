# Stage 2 v2 remote run

Stage 2 v2 uses one instruction-tuned model for both capability and refusal so
that A/B/C are genuine matched-checkpoint interventions. The default is
`google/gemma-2-2b-it` with the frozen layer-12 Gemma Scope 16K SAE.

The run is deliberately split into causal measurement, intervention, and
analysis. Every step writes a JSON log and the final analysis stops before
Stage 3.

## Required inputs

- `google/gemma-2-2b-it`
- `gemma-scope-2b-pt-res-canonical`, `layer_12/width_16k/canonical`
- AdvBench `harmful_behaviors.csv`, passed with `--advbench-path`

## Smoke test

```bash
git pull

bash remote/run_stage2_gate_causal_v2.sh \
  --step all \
  --advbench-path /root/autodl-tmp/ffap/data/advbench/harmful_behaviors.csv \
  --ability-limit 8 \
  --refusal-limit 8 \
  --ablation-eval-limit 4 \
  --validation-features-per-tail 1 \
  --sparsities 0.30 \
  --ppl-blocks 2 \
  --feature-blocks 2 \
  --calib-blocks 2 \
  --bootstrap-samples 500 \
  --no-save-checkpoints \
  --artifact-dir results/stage2_v2_smoke/artifacts \
  --model-csv results/stage2_v2_smoke_models.csv \
  --example-csv results/stage2_v2_smoke_examples.csv \
  --out-json results/stage2_gate_v2_smoke.json
```

## Full run

```bash
bash remote/run_stage2_gate_causal_v2.sh \
  --step causal \
  --advbench-path /root/autodl-tmp/ffap/data/advbench/harmful_behaviors.csv

bash remote/run_stage2_gate_causal_v2.sh \
  --step intervention \
  --advbench-path /root/autodl-tmp/ffap/data/advbench/harmful_behaviors.csv
```

To repeat only statistics and gate rendering:

```bash
bash remote/run_stage2_gate_causal_v2.sh --step analyze
```

The final artifact is `results/stage2_gate_v2.json`. Do not start Stage 3 until
its `gate_status` has been reviewed by a human.
