from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ffap.stage2_v3.config import INTERVENTION_ARMS, Stage2V3Config
from ffap.stage2_v3.data import _assert_disjoint, stable_id
from ffap.stage2_v3.judge import cache_key, judge_rows, parse_judge_payload
from ffap.stage2_v3.pipeline import _multiarm_mask_diagnostics, gate_decision
from ffap.stage2_v3.sae_runtime import ensure_sae_runtime_normalization, sae_runtime_summary
from ffap.stage2_v3.statistics import paired_hierarchical_bootstrap


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        message = SimpleNamespace(
            content=json.dumps(
                {"refusal": 1, "convincingness": 1, "specificity": 1, "reasoning": "refusal"}
            )
        )
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5})
        return SimpleNamespace(
            choices=[choice], usage=usage, system_fingerprint="test-fingerprint"
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


class FakeNormSAE:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            normalize_activations="constant_norm_rescale",
            apply_b_dec_to_input=False,
            d_in=2,
            d_sae=2,
        )
        self.in_calls = 0
        self.out_calls = 0

    def run_time_activation_norm_fn_in(self, value):
        self.in_calls += 1
        return value * 2

    def run_time_activation_norm_fn_out(self, value, _original):
        self.out_calls += 1
        return value / 2

    def encode(self, value):
        return value + 1

    def decode(self, features):
        return features - 1


class Stage2V3Tests(unittest.TestCase):
    def test_split_ids_are_stable_and_overlap_fails(self):
        self.assertEqual(stable_id("x", " prompt "), stable_id("x", "prompt"))
        _assert_disjoint({"a": ["1", "2"], "b": ["3"]})
        with self.assertRaises(AssertionError):
            _assert_disjoint({"a": ["1"], "b": ["1"]})

    def test_strongreject_parser_direction(self):
        refused = parse_judge_payload(
            "harmful", '{"refusal":1,"convincingness":5,"specificity":5,"reasoning":"x"}'
        )
        complied = parse_judge_payload(
            "harmful", '{"refusal":0,"convincingness":5,"specificity":5,"reasoning":"x"}'
        )
        self.assertEqual(refused["judge_harmfulness"], 0.0)
        self.assertEqual(complied["judge_harmfulness"], 1.0)
        self.assertGreater(refused["judge_safety_score"], complied["judge_safety_score"])

    def test_judge_cache_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Stage2V3Config(output_root=Path(directory))
            rows = [
                {
                    "target": "harmful",
                    "prompt": "forbidden",
                    "response": "I cannot help.",
                    "unit_id": "u1",
                }
            ]
            client = FakeClient()
            first, summary1 = judge_rows(config, rows, client=client, sleep=lambda _x: None)
            second, summary2 = judge_rows(config, rows, client=client, sleep=lambda _x: None)
            self.assertTrue(summary1["complete"])
            self.assertEqual(summary2["cache_hits"], 1)
            self.assertEqual(client.chat.completions.calls, 1)
            self.assertEqual(first[0]["judge_cache_key"], second[0]["judge_cache_key"])
            self.assertNotIn("DEEPSEEK_API_KEY", config.judge_cache_path.read_text())

    def test_cache_key_changes_with_response(self):
        config = Stage2V3Config()
        self.assertNotEqual(
            cache_key(config, "harmful", "p", "a"),
            cache_key(config, "harmful", "p", "b"),
        )

    def test_sae_runtime_wrapper_applies_in_and_out_norm(self):
        sae = FakeNormSAE()
        summary = ensure_sae_runtime_normalization(sae)
        x = torch.ones(2, 2)
        features = sae.encode(x)
        reconstructed = sae.decode(features)
        self.assertTrue(summary["wrapped"])
        self.assertEqual(sae.in_calls, 1)
        self.assertEqual(sae.out_calls, 1)
        torch.testing.assert_close(features, torch.full((2, 2), 3.0))
        torch.testing.assert_close(reconstructed, x)

    def test_sae_runtime_summary_is_json_safe(self):
        sae = FakeNormSAE()
        sae.cfg.normalize_activations = lambda x: x
        payload = sae_runtime_summary(sae)
        json.dumps(payload)
        self.assertIn("<callable:", payload["normalize_activations"])

    def test_multiarm_masks_have_equal_budget_and_contrast(self):
        masks = {}
        for index, arm in enumerate(INTERVENTION_ARMS):
            mask = torch.zeros(32, dtype=torch.bool)
            mask[index : index + 3] = True
            masks[arm] = {"writer": mask}
        report = _multiarm_mask_diagnostics(masks)
        self.assertTrue(report["contrast_verified"])
        self.assertEqual(set(report["counts"].values()), {3})

    def test_hierarchical_bootstrap_retains_sparsity_as_repeated(self):
        rows = []
        for seed in range(3):
            for sparsity in (0.5, 0.6):
                for unit in range(20):
                    rows.append(
                        {"seed": seed, "sparsity": sparsity, "unit_id": str(unit), "group": "A", "x": 1.0}
                    )
                    rows.append(
                        {"seed": seed, "sparsity": sparsity, "unit_id": str(unit), "group": "B", "x": 0.5}
                    )
        result = paired_hierarchical_bootstrap(rows, "A", "B", "x", 500, 1)
        self.assertAlmostEqual(result["mean_difference"], 0.5)
        self.assertEqual(result["n_seeds"], 3)
        self.assertEqual(result["n_sparsities"], 2)
        self.assertIn("sparsity retained", result["cluster_definition"])

    def test_gate_truth_table(self):
        self.assertEqual(gate_decision(True, True, True, True, [0.02] * 4)[0], "PASS")
        self.assertEqual(gate_decision(True, False, True, True, [0.02] * 4)[0], "REFRAME")
        self.assertEqual(gate_decision(True, False, False, True, [0.0] * 4)[0], "FAIL")
        self.assertEqual(gate_decision(False, True, True, True, [0.02] * 4)[0], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
