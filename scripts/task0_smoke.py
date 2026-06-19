from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ffap.json_utils import write_json


def now() -> float:
    return time.time()


def format_error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def load_sae_compat(release: str, sae_id: str, device: str) -> tuple[Any, dict[str, Any]]:
    from sae_lens import SAE

    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    metadata: dict[str, Any] = {"release": release, "sae_id": sae_id}
    if isinstance(loaded, tuple):
        sae = loaded[0]
        if len(loaded) > 1:
            metadata["cfg"] = loaded[1]
        if len(loaded) > 2 and loaded[2] is not None:
            metadata["sparsity_loaded"] = True
    else:
        sae = loaded
    if hasattr(sae, "eval"):
        sae.eval()
    if hasattr(sae, "to"):
        sae.to(device)
    return sae, metadata


def capture_layer_activation(model: Any, layer_index: int, **forward_kwargs: Any) -> Any:
    activations: list[Any] = []
    layer = model.model.layers[layer_index]

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        activations.append(hidden.detach())

    handle = layer.register_forward_hook(hook)
    try:
        with forward_kwargs["torch"].no_grad():
            _ = model(**{k: v for k, v in forward_kwargs.items() if k != "torch"})
    finally:
        handle.remove()
    if not activations:
        raise RuntimeError(f"No activation captured for layer {layer_index}")
    return activations[-1]


def run_lm_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.lm_eval_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cli = shutil.which("lm_eval")
    command = [
        cli if cli else sys.executable,
        *(["-m", "lm_eval"] if cli is None else []),
        "--model",
        "hf",
        "--model_args",
        (
            f"pretrained={args.model_id},dtype=bfloat16,"
            "device_map=cuda:0,trust_remote_code=False"
        ),
        "--tasks",
        args.lm_eval_task,
        "--limit",
        str(args.lm_eval_limit),
        "--batch_size",
        args.lm_eval_batch_size,
        "--output_path",
        str(output_dir),
    ]
    started = now()
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.lm_eval_timeout_sec,
        check=False,
    )
    result: dict[str, Any] = {
        "command": command,
        "returncode": proc.returncode,
        "elapsed_sec": round(now() - started, 3),
        "stdout_tail": proc.stdout.splitlines()[-80:],
        "stderr_tail": proc.stderr.splitlines()[-80:],
        "output_dir": str(output_dir),
    }
    json_files = sorted(output_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime)
    result["json_files"] = [str(path) for path in json_files]
    if json_files:
        latest = json_files[-1]
        try:
            parsed = json.loads(latest.read_text(encoding="utf-8"))
            result["parsed_results_file"] = str(latest)
            result["arc_easy"] = parsed.get("results", {}).get(args.lm_eval_task, {})
        except Exception as exc:
            result["parse_error"] = format_error(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FFAP Task 0 smoke test")
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--prompt", default="Feature fidelity matters because")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    parser.add_argument("--lm-eval-task", default="arc_easy")
    parser.add_argument("--lm-eval-limit", type=int, default=20)
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-output-dir", default="results/task0_lm_eval")
    parser.add_argument("--lm-eval-timeout-sec", type=int, default=1800)
    parser.add_argument("--skip-lm-eval", action="store_true")
    parser.add_argument("--out", default="logs/task0_smoke.json")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "task": "task0_smoke",
        "timestamp_unix": now(),
        "config": vars(args),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        },
        "status": "STARTED",
    }

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; Task 0 requires the GPU remote.")

        torch.cuda.reset_peak_memory_stats()
        device = "cuda:0"

        t0 = now()
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        model.eval()
        payload["model_load"] = {
            "elapsed_sec": round(now() - t0, 3),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }

        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
        t1 = now()
        with torch.no_grad():
            dense_outputs = model(**inputs)
        payload["forward"] = {
            "elapsed_sec": round(now() - t1, 3),
            "logits_shape": list(dense_outputs.logits.shape),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }

        t2 = now()
        sae, sae_metadata = load_sae_compat(args.sae_release, args.sae_id, device)
        activation = capture_layer_activation(model, args.layer, torch=torch, **inputs)
        flat_activation = activation.reshape(-1, activation.shape[-1]).to(device)
        with torch.no_grad():
            features = sae.encode(flat_activation)
            reconstruction = sae.decode(features)
            mse = torch.mean((reconstruction.float() - flat_activation.float()) ** 2)
            l0 = (features.abs() > 0).float().sum(dim=-1).mean()
        payload["sae"] = {
            **sae_metadata,
            "elapsed_sec": round(now() - t2, 3),
            "activation_shape": list(activation.shape),
            "flat_activation_shape": list(flat_activation.shape),
            "feature_shape": list(features.shape),
            "reconstruction_mse": float(mse.detach().cpu()),
            "l0": float(l0.detach().cpu()),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }

        del sae, features, reconstruction, flat_activation, activation, dense_outputs, model
        torch.cuda.empty_cache()

        if args.skip_lm_eval:
            payload["lm_eval"] = {"skipped": True}
        else:
            payload["lm_eval"] = run_lm_eval(args)
            if payload["lm_eval"]["returncode"] != 0:
                raise RuntimeError("lm-eval failed; see lm_eval.stderr_tail in log.")

        payload["status"] = "PASS"
        payload["conclusion"] = (
            "Task 0 smoke passed: dense forward, SAE reconstruction, and ARC-Easy "
            "baseline command completed."
        )
        write_json(args.out, payload)
        print(f"wrote {args.out}")
        print("status: PASS")
        return 0
    except Exception as exc:
        payload["status"] = "FAIL"
        payload["error"] = format_error(exc)
        payload["conclusion"] = "Task 0 smoke did not pass; inspect error and logs."
        write_json(args.out, payload)
        print(f"wrote {args.out}")
        print(f"status: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

