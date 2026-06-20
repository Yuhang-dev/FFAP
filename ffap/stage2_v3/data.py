from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from .config import Stage2V3Config
from .legacy import task_gate


@dataclass(frozen=True)
class PromptExample:
    example_id: str
    prompt: str
    source: str
    label: str


def stable_id(namespace: str, text: str) -> str:
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:20]
    return f"{namespace}:{digest}"


def _assert_disjoint(named_ids: dict[str, Iterable[str]]) -> None:
    materialized = {name: set(values) for name, values in named_ids.items()}
    names = list(materialized)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = materialized[left] & materialized[right]
            if overlap:
                sample = sorted(overlap)[:3]
                raise AssertionError(f"Split leakage between {left} and {right}: {sample}")


def _serialize_mc(example: Any) -> dict[str, Any]:
    return {
        "task": example.task,
        "example_id": example.example_id,
        "prompt": example.prompt,
        "choices": list(example.choices),
        "gold": int(example.gold),
    }


def _deserialize_mc(payload: dict[str, Any]) -> Any:
    return task_gate.MCExample(
        task=payload["task"],
        example_id=payload["example_id"],
        prompt=payload["prompt"],
        choices=list(payload["choices"]),
        gold=int(payload["gold"]),
    )


def _namespace_mc(examples: Iterable[Any], split: str) -> list[Any]:
    return [
        task_gate.MCExample(
            task=item.task,
            example_id=f"{item.task}:{split}:{item.example_id.split(':', 1)[-1]}",
            prompt=item.prompt,
            choices=list(item.choices),
            gold=int(item.gold),
        )
        for item in examples
    ]


def _load_harmful_rows(config: Stage2V3Config) -> list[dict[str, Any]]:
    if config.advbench_path:
        with Path(config.advbench_path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    return [dict(row) for row in load_dataset(config.advbench_dataset, split=config.advbench_split)]


def _prompt_from_row(row: dict[str, Any]) -> str:
    for key in ("prompt", "goal", "instruction", "behavior"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def _load_prompt_examples(
    rows: Iterable[dict[str, Any]], namespace: str, label: str
) -> list[PromptExample]:
    output: dict[str, PromptExample] = {}
    for row in rows:
        prompt = _prompt_from_row(row)
        if not prompt:
            continue
        example = PromptExample(stable_id(namespace, prompt), prompt, namespace, label)
        output.setdefault(example.example_id, example)
    return list(output.values())


def _is_safe_label(value: Any) -> bool:
    normalized = str(value).strip().lower().replace("-", "_")
    return normalized in {"safe", "benign", "true", "1", "should_respond"}


def prepare_splits(config: Stage2V3Config) -> dict[str, Any]:
    tasks = [item.strip() for item in config.ability_tasks.split(",") if item.strip()]
    ability: dict[str, dict[str, list[Any]]] = {}
    for task in tasks:
        calibration = _namespace_mc(
            task_gate.load_task_examples(
                task, "train", config.ability_calibration_per_task, config.split_seed
            ),
            "train",
        )
        validation = _namespace_mc(
            task_gate.load_task_examples(
                task,
                "validation",
                config.ability_dev_per_task + config.ability_test_per_task,
                config.split_seed,
            ),
            "validation",
        )
        dev = validation[: config.ability_dev_per_task]
        test = validation[
            config.ability_dev_per_task : config.ability_dev_per_task
            + config.ability_test_per_task
        ]
        if len(calibration) < config.ability_calibration_per_task or len(dev) < config.ability_dev_per_task or len(test) < config.ability_test_per_task:
            raise RuntimeError(f"Insufficient labeled examples for {task} split plan.")
        _assert_disjoint(
            {
                f"{task}.calibration": [item.example_id for item in calibration],
                f"{task}.dev": [item.example_id for item in dev],
                f"{task}.test": [item.example_id for item in test],
            }
        )
        ability[task] = {"calibration": calibration, "dev": dev, "test": test}

    harmful = _load_prompt_examples(_load_harmful_rows(config), "advbench", "harmful")
    random.Random(config.split_seed).shuffle(harmful)
    h0 = config.harmful_calibration
    h1 = h0 + config.harmful_dev
    h2 = h1 + config.harmful_test
    if len(harmful) < h2:
        raise RuntimeError(f"AdvBench has {len(harmful)} unique prompts; {h2} are required.")
    harmful_splits = {
        "calibration": harmful[:h0],
        "dev": harmful[h0:h1],
        "test": harmful[h1:h2],
    }

    xstest = load_dataset(config.xstest_dataset, split=config.xstest_split)
    safe_rows = [dict(row) for row in xstest if _is_safe_label(row.get("label"))]
    benign = _load_prompt_examples(safe_rows, "xstest", "benign")
    random.Random(config.split_seed).shuffle(benign)
    b0 = config.benign_calibration
    b1 = b0 + config.benign_dev
    b2 = b1 + config.benign_test
    if len(benign) < b2:
        observed = sorted({str(row.get("label")) for row in xstest})
        raise RuntimeError(
            f"XSTest has {len(benign)} safe prompts; {b2} are required. Labels: {observed}"
        )
    benign_splits = {
        "calibration": benign[:b0],
        "dev": benign[b0:b1],
        "test": benign[b1:b2],
    }
    _assert_disjoint(
        {f"harmful.{name}": [item.example_id for item in items] for name, items in harmful_splits.items()}
    )
    _assert_disjoint(
        {f"benign.{name}": [item.example_id for item in items] for name, items in benign_splits.items()}
    )

    return {
        "schema_version": 1,
        "split_seed": config.split_seed,
        "ability": {
            task: {
                split: [_serialize_mc(item) for item in items]
                for split, items in splits.items()
            }
            for task, splits in ability.items()
        },
        "harmful": {
            split: [asdict(item) for item in items] for split, items in harmful_splits.items()
        },
        "benign": {
            split: [asdict(item) for item in items] for split, items in benign_splits.items()
        },
        "conclusion": "Calibration, layer-dev, and final-test IDs are fixed and disjoint.",
    }


def ability_examples(manifest: dict[str, Any], split: str) -> list[Any]:
    output = []
    for task in manifest["ability"].values():
        output.extend(_deserialize_mc(item) for item in task[split])
    return output


def prompt_examples(manifest: dict[str, Any], target: str, split: str) -> list[PromptExample]:
    return [PromptExample(**item) for item in manifest[target][split]]
