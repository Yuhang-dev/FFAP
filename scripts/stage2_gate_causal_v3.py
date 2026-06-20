from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ffap.stage2_v3.config import DEFAULT_SAE_IDS, Stage2V3Config
from ffap.stage2_v3.pipeline import run


GPU_STEPS = {"all", "layer-scan", "causal", "intervention"}


def _ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _floats(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _strings(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Stage 2 v3: matched-IT, held-out local causal validation gate"
    )
    result.add_argument(
        "--step",
        choices=("all", "prepare", "layer-scan", "causal", "intervention", "judge", "manual-export", "analyze"),
        default="all",
    )
    result.add_argument("--model-id", default="google/gemma-2-9b-it")
    result.add_argument("--sae-release", default="gemma-scope-9b-it-res-canonical")
    result.add_argument("--sae-ids", default=",".join(DEFAULT_SAE_IDS))
    result.add_argument("--ability-tasks", default="arc_easy,hellaswag")
    result.add_argument("--ability-calibration-per-task", type=int, default=512)
    result.add_argument("--ability-dev-per-task", type=int, default=128)
    result.add_argument("--ability-test-per-task", type=int, default=256)
    result.add_argument("--advbench-path", default=None)
    result.add_argument("--advbench-dataset", default="walledai/AdvBench")
    result.add_argument("--harmful-calibration", type=int, default=256)
    result.add_argument("--harmful-dev", type=int, default=128)
    result.add_argument("--harmful-test", type=int, default=128)
    result.add_argument("--xstest-dataset", default="walledai/XSTest")
    result.add_argument("--benign-calibration", type=int, default=100)
    result.add_argument("--benign-dev", type=int, default=50)
    result.add_argument("--benign-test", type=int, default=100)
    result.add_argument("--split-seed", type=int, default=20260620)
    result.add_argument("--seeds", default="0,1,2")
    result.add_argument("--sparsities", default="0.50,0.60,0.70,0.80")
    result.add_argument("--max-length", type=int, default=256)
    result.add_argument("--batch-examples", type=int, default=2)
    result.add_argument("--generation-batch-size", type=int, default=4)
    result.add_argument("--max-new-tokens", type=int, default=128)
    result.add_argument("--protect-fraction", type=float, default=0.02)
    result.add_argument("--causal-top-fraction", type=float, default=0.05)
    result.add_argument("--causal-sharpen-power", type=float, default=2.0)
    result.add_argument("--bootstrap-samples", type=int, default=10000)
    result.add_argument("--judge-base-url", default="https://api.deepseek.com")
    result.add_argument("--judge-model", default="deepseek-v4-flash")
    result.add_argument("--judge-max-retries", type=int, default=5)
    result.add_argument("--manual-sample-size", type=int, default=200)
    result.add_argument("--output-root", type=Path, default=Path("results/stage2_v3"))
    result.add_argument("--log-root", type=Path, default=Path("logs"))
    result.add_argument("--final-json", type=Path, default=Path("results/stage2_gate_v3_local.json"))
    result.add_argument("--device", default="cuda")
    result.add_argument("--smoke", action="store_true")
    return result


def config_from_args(args: argparse.Namespace) -> Stage2V3Config:
    config = Stage2V3Config(
        step=args.step,
        model_id=args.model_id,
        sae_release=args.sae_release,
        sae_ids=_strings(args.sae_ids),
        ability_tasks=args.ability_tasks,
        ability_calibration_per_task=args.ability_calibration_per_task,
        ability_dev_per_task=args.ability_dev_per_task,
        ability_test_per_task=args.ability_test_per_task,
        advbench_path=args.advbench_path,
        advbench_dataset=args.advbench_dataset,
        harmful_calibration=args.harmful_calibration,
        harmful_dev=args.harmful_dev,
        harmful_test=args.harmful_test,
        xstest_dataset=args.xstest_dataset,
        benign_calibration=args.benign_calibration,
        benign_dev=args.benign_dev,
        benign_test=args.benign_test,
        split_seed=args.split_seed,
        seeds=_ints(args.seeds),
        sparsities=_floats(args.sparsities),
        max_length=args.max_length,
        batch_examples=args.batch_examples,
        generation_batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        protect_fraction=args.protect_fraction,
        causal_top_fraction=args.causal_top_fraction,
        causal_sharpen_power=args.causal_sharpen_power,
        bootstrap_samples=args.bootstrap_samples,
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
        judge_max_retries=args.judge_max_retries,
        manual_sample_size=args.manual_sample_size,
        output_root=args.output_root,
        log_root=args.log_root,
        final_json=args.final_json,
        extra={"smoke": bool(args.smoke)},
    )
    if args.smoke:
        config.ability_calibration_per_task = min(config.ability_calibration_per_task, 8)
        config.ability_dev_per_task = min(config.ability_dev_per_task, 4)
        config.ability_test_per_task = min(config.ability_test_per_task, 8)
        config.harmful_calibration = min(config.harmful_calibration, 8)
        config.harmful_dev = min(config.harmful_dev, 4)
        config.harmful_test = min(config.harmful_test, 8)
        config.benign_calibration = min(config.benign_calibration, 8)
        config.benign_dev = min(config.benign_dev, 4)
        config.benign_test = min(config.benign_test, 8)
        config.seeds = (0, 1, 2)
        config.sparsities = (0.5,)
        config.calib_blocks = 2
        config.ppl_blocks = 2
        config.validation_features_per_tail = 1
        config.ablation_eval_limit = 4
        config.bootstrap_samples = 500
        config.manual_sample_size = 12
        config.output_root = Path("results/stage2_v3_smoke")
        config.final_json = Path("results/stage2_gate_v3_local_smoke.json")
    return config


def validate(config: Stage2V3Config, device: str) -> None:
    if len(config.seeds) < 3:
        raise ValueError("Stage 2 v3 requires at least three experimental seeds.")
    if len(config.sae_ids) != 3:
        raise ValueError("R0 scan requires the three official 9B-IT canonical SAE layers.")
    if any(not 0 < value < 1 for value in config.sparsities):
        raise ValueError("Every sparsity must be in (0, 1).")
    if config.step in GPU_STEPS and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("This Stage 2 v3 step requires an active CUDA GPU.")


def main() -> int:
    args = parser().parse_args()
    config = config_from_args(args)
    validate(config, args.device)
    result = run(config, args.device)
    status = result.get("gate_status", result.get("status", "UNKNOWN"))
    print(f"status: {status}")
    if result.get("conclusion"):
        print(result["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
