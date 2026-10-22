import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "integrations" / "rlinf_v03" / "aggregate_streamvln_g8_algorithm.py"
SPEC = importlib.util.spec_from_file_location("aggregate_streamvln_g8_algorithm", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestStreamVLNG8Aggregate(unittest.TestCase):
    def result(self, seed, bc_sr, rl_sr, bc_spl, rl_spl, bc_timeouts=1, rl_timeouts=1):
        aggregate = lambda sr, spl, timeout: {
            "episode_count": 22,
            "success_rate": sr,
            "spl": spl,
            "timeout_count": timeout,
            "mode_counts": {"navigate": 4, "stop": 2},
            "primitive_counts": {"0": 2, "1": 4},
        }
        return {
            "schema_version": 1,
            "kind": "streamvln-g8-matched-bc-vs-bc-rl",
            "seed": seed,
            "all_training_and_evaluation_guards_passed": True,
            "algorithm_gate_member": rl_sr > bc_sr and rl_spl >= bc_spl and rl_timeouts <= bc_timeouts,
            "source_contract": {"ttt": False, "secondary_post_training": False, "val_unseen_used": False},
            "development": {
                "bc": aggregate(bc_sr, bc_spl, bc_timeouts),
                "bc_rl": aggregate(rl_sr, rl_spl, rl_timeouts),
            },
            "guards": {"identity": True},
            "counts": {
                "development_episodes_per_arm": 22,
                "model_loads": 1,
                "train_episodes": 32,
                "val_unseen_episodes": 0,
            },
        }

    def run_aggregate(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_bytes((ROOT / "integrations" / "rlinf_v03" / "streamvln_g8_algorithm_manifest.json").read_bytes())
            paths = []
            for row in rows:
                path = root / f"{row['seed']}.json"
                path.write_text(json.dumps(row))
                paths.append(path)
            return MODULE.aggregate(manifest, paths)

    def test_passes_exact_preregistered_rule(self):
        rows = [
            self.result(2026092808, 0.10, 0.20, 0.08, 0.10),
            self.result(2026092809, 0.10, 0.15, 0.08, 0.09),
            self.result(2026092810, 0.10, 0.10, 0.08, 0.08),
        ]
        report = self.run_aggregate(rows)
        self.assertTrue(report["algorithm_gate_passed"])
        self.assertEqual(report["next_stage"], "freeze_final_val_unseen_schedule")
        self.assertEqual(report["pooled"]["bc_rl"]["episode_count"], 66)

    def test_fails_without_seed_majority(self):
        rows = [
            self.result(2026092808, 0.10, 0.20, 0.08, 0.10),
            self.result(2026092809, 0.10, 0.10, 0.08, 0.08),
            self.result(2026092810, 0.10, 0.10, 0.08, 0.08),
        ]
        report = self.run_aggregate(rows)
        self.assertFalse(report["gates"]["seed_majority_success_rate"])
        self.assertFalse(report["algorithm_gate_passed"])

    def test_rejects_missing_seed(self):
        rows = [
            self.result(2026092808, 0.10, 0.20, 0.08, 0.10),
            self.result(2026092809, 0.10, 0.15, 0.08, 0.09),
        ]
        with self.assertRaisesRegex(ValueError, "exactly one result"):
            self.run_aggregate(rows)


if __name__ == "__main__":
    unittest.main()
