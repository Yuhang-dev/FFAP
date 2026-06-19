from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ffap.json_utils import write_json


def safe_name(model_ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_ref.strip("/"))


def parse_checkpoint(path: Path) -> dict[str, Any]:
    name = path.name
    match = re.search(r"_s([0-9.]+)$", name)
    sparsity = float(match.group(1)) if match else None
    method = name[: match.start()] if match else "checkpoint"
    return {
        "model_ref": str(path),
        "label": name,
        "method": method,
        "sparsity": sparsity,
    }


def discover_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    if args.include_dense:
        models.append(
            {
                "model_ref": args.dense_model,
                "label": "dense",
                "method": "dense",
                "sparsity": 0.0,
            }
        )
    for path in sorted(Path().glob(args.checkpoint_glob)):
        if path.is_dir() and (path / "config.json").exists():
            models.append(parse_checkpoint(path))
    if not models:
        raise RuntimeError("No models discovered for capability evaluation.")
    return models


def run_lm_eval(
    model_ref: str,
    tasks: str,
    limit: int,
    batch_size: str,
    output_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cli = shutil.which("lm_eval")
    if cli:
        command = [cli]
    else:
        command = [sys.executable, "-m", "lm_eval"]
    command.extend(
        [
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model_ref},dtype=bfloat16,device_map=cuda:0,trust_remote_code=False",
            "--tasks",
            tasks,
            "--limit",
            str(limit),
            "--batch_size",
            batch_size,
            "--output_path",
            str(output_dir),
        ]
    )
    started = time.time()
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
        check=False,
    )
    json_files = sorted(output_dir.rglob("*.json"), key=lambda path: path.stat().st_mtime)
    parsed = None
    if json_files:
        parsed = json.loads(json_files[-1].read_text(encoding="utf-8"))
    return {
        "command": command,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout_tail": proc.stdout.splitlines()[-80:],
        "stderr_tail": proc.stderr.splitlines()[-80:],
        "output_dir": str(output_dir),
        "json_files": [str(path) for path in json_files],
        "parsed_results_file": str(json_files[-1]) if json_files else None,
        "parsed": parsed,
    }


def metric_rows(model_info: dict[str, Any], eval_result: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = eval_result.get("parsed") or {}
    results = parsed.get("results", {})
    rows = []
    for task_name, metrics in results.items():
        row = {
            "label": model_info["label"],
            "model_ref": model_info["model_ref"],
            "method": model_info["method"],
            "sparsity": model_info["sparsity"],
            "task": task_name,
            "sample_len": metrics.get("sample_len"),
            "acc": metrics.get("acc,none"),
            "acc_stderr": metrics.get("acc_stderr,none"),
            "acc_norm": metrics.get("acc_norm,none"),
            "acc_norm_stderr": metrics.get("acc_norm_stderr,none"),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "method",
        "sparsity",
        "task",
        "sample_len",
        "acc",
        "acc_stderr",
        "acc_norm",
        "acc_norm_stderr",
        "model_ref",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 capability eval via lm-eval")
    parser.add_argument("--dense-model", default="google/gemma-2-2b")
    parser.add_argument(
        "--checkpoint-glob",
        default="outputs/stage1_magnitude_sweep/*",
        help="Glob of saved pruned checkpoint directories.",
    )
    parser.add_argument("--include-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tasks", default="arc_easy,hellaswag")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--raw-output-dir", default="results/stage1_capability_eval/raw")
    parser.add_argument("--out-json", default="logs/stage1_capability_eval.json")
    parser.add_argument("--out-csv", default="results/stage1_capability_eval.csv")
    args = parser.parse_args()

    started = time.time()
    payload: dict[str, Any] = {
        "task": "stage1_capability_eval",
        "timestamp_unix": started,
        "config": vars(args),
        "status": "STARTED",
    }
    all_rows: list[dict[str, Any]] = []
    runs = []

    try:
        models = discover_models(args)
        payload["models"] = models
        for model_info in models:
            run_dir = Path(args.raw_output_dir) / safe_name(model_info["label"])
            result = run_lm_eval(
                model_info["model_ref"],
                args.tasks,
                args.limit,
                args.batch_size,
                run_dir,
                args.timeout_sec,
            )
            if result["returncode"] != 0:
                raise RuntimeError(
                    f"lm-eval failed for {model_info['label']} with code "
                    f"{result['returncode']}"
                )
            rows = metric_rows(model_info, result)
            all_rows.extend(rows)
            runs.append(
                {
                    "model": model_info,
                    "elapsed_sec": result["elapsed_sec"],
                    "parsed_results_file": result["parsed_results_file"],
                    "rows": rows,
                }
            )
            write_csv(Path(args.out_csv), all_rows)
            write_json(args.out_json, {**payload, "status": "RUNNING", "runs": runs})

        payload.update(
            {
                "status": "PASS",
                "elapsed_sec": round(time.time() - started, 3),
                "runs": runs,
                "outputs": {"json": args.out_json, "csv": args.out_csv},
                "conclusion": "Stage 1 capability evaluation completed.",
            }
        )
        write_csv(Path(args.out_csv), all_rows)
        write_json(args.out_json, payload)
        print(f"wrote {args.out_json}")
        print(f"wrote {args.out_csv}")
        print("status: PASS")
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "FAIL",
                "elapsed_sec": round(time.time() - started, 3),
                "runs": runs,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "conclusion": "Stage 1 capability evaluation failed.",
            }
        )
        write_json(args.out_json, payload)
        if all_rows:
            write_csv(Path(args.out_csv), all_rows)
        print(f"wrote {args.out_json}")
        print(f"status: FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

