# Stage 2 v3 remote run

Stage 2 v3 is a new held-out local measurement gate. It leaves Stage 2 v1/v2
unchanged, uses `google/gemma-2-9b-it` with the matched official 16K IT SAEs at
layers 9/20/31, and never starts W1 or Stage 3.

## One-time setup and prefetch

The Gemma account must have 9B-IT access. Downloads use normal HTTP rather than
Xet. Install only the added API client from the official PyPI index:

```bash
cd /root/autodl-tmp/ffap
git pull
python -m pip install --index-url https://pypi.org/simple openai
bash remote/prefetch_stage2_v3_assets.sh
```

The DeepSeek key is needed only for `--step judge`. Do not put it in the repo:

```bash
export DEEPSEEK_API_KEY='...'
```

## Smoke

Smoke validates plumbing only. It writes to `results/stage2_v3_smoke` and its
numbers are not scientific evidence.

```bash
bash remote/run_stage2_gate_causal_v3.sh --step prepare --smoke
bash remote/run_stage2_gate_causal_v3.sh --step layer-scan --smoke
bash remote/run_stage2_gate_causal_v3.sh --step causal --smoke
bash remote/run_stage2_gate_causal_v3.sh --step intervention --smoke
bash remote/run_stage2_gate_causal_v3.sh --step judge --smoke
bash remote/run_stage2_gate_causal_v3.sh --step manual-export --smoke
```

## Full measurement gate

Run each expensive stage separately so completed artifacts survive shutdowns:

```bash
bash remote/run_stage2_gate_causal_v3.sh --step prepare
bash remote/run_stage2_v3_r0_diagnostic.sh
bash remote/run_stage2_gate_causal_v3.sh --step layer-scan
bash remote/run_stage2_gate_causal_v3.sh --step causal
bash remote/run_stage2_gate_causal_v3.sh --step intervention
bash remote/run_stage2_gate_causal_v3.sh --step judge
bash remote/run_stage2_gate_causal_v3.sh --step manual-export
```

The R0 diagnostic compares raw SAE `encode/decode` against the optional runtime
wrapper on token-level and prompt-final activations. The Stage 2 v3 gate uses raw
SAE calls by default; only enable the wrapper with `--use-sae-runtime-wrapper`
after the diagnostic shows it is needed.

Two blinded files are then created:

```text
results/stage2_v3/manual_labels_annotator1.csv
results/stage2_v3/manual_labels_annotator2.csv
```

After two annotators fill the rubric columns and adjudication fields for
disagreements, run:

```bash
bash remote/run_stage2_gate_causal_v3.sh --step analyze
```

The mandatory-stop result is:

```text
results/stage2_gate_v3_local.json
```

`PASS`, `REFRAME`, `FAIL`, and `INCONCLUSIVE` all require human confirmation.
No command in this pipeline starts whole-model W1 or Stage 3.
