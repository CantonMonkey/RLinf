import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest


class StreamVLNG720ProtocolTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    INTEGRATION = ROOT / "integrations" / "rlinf_v03"

    def test_manifest_freezes_three_independent_arms_and_atomic_rule(self):
        path = self.INTEGRATION / "streamvln_g720_experiment.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "0286ffdcb1a42da579c76eeb6600d4123bb6edb242c1d7ec506836fb7d1a311d",
        )
        manifest = json.loads(path.read_text())
        protocol = manifest["protocol"]
        self.assertEqual(manifest["selection"]["seeds"], [2026092801, 2026092802, 2026092803])
        self.assertEqual(set(protocol["arms"]), {"2026092801", "2026092802", "2026092803"})
        scenes = [row["scene"] for arm in protocol["arms"].values() for row in arm]
        self.assertEqual(len(scenes), 12)
        self.assertEqual(len(set(scenes)), 12)
        self.assertEqual(protocol["collection_contract"]["min_rows"], 8)
        self.assertEqual(protocol["collection_contract"]["max_rows"], 12)
        self.assertEqual(protocol["candidate_update_limit"], 2)
        self.assertEqual(protocol["postupdate_ratio_max_deviation"], 0.5)
        self.assertEqual(protocol["postupdate_distribution_kl_max"], 0.1)
        self.assertTrue(protocol["candidate_two_requires_candidate_one_accepted"])
        self.assertFalse(protocol["outcome_based_seed_update_or_threshold_selection"])
        self.assertEqual(protocol["val_unseen_episode_count"], 0)
        self.assertFalse(manifest["g8_ready"])

    def test_runner_has_complete_candidate_boundary_and_no_selection_path(self):
        path = self.INTEGRATION / "run_streamvln_atomic_guard_g720.py"
        source = path.read_text()
        ast.parse(source)
        self.assertIn("before_snapshot = capture_training_state", source)
        self.assertIn("after_snapshot = capture_training_state", source)
        self.assertIn("restore_training_state(manager, core, before_snapshot)", source)
        self.assertIn("final_state_equals_last_accepted_state", source)
        self.assertIn("all_rejected_candidates_exactly_restored", source)
        self.assertNotIn("sample_target_mode", source)
        self.assertNotIn("save_head_checkpoint", source)
        self.assertNotIn("val_unseen = collect_phase", source)
        self.assertEqual(source.count("manager.optimizer_step()"), 1)

        helper = (self.INTEGRATION / "streamvln_atomic_state.py").read_text()
        self.assertIn('core.load_high_level_state(clone_tree(snapshot["parameters"]))', helper)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None,
    "exact state restore test requires torch and numpy",
)
class AtomicStateRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(cls_root := Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"))
        cls.integration = cls_root

    def test_round_trip_restores_every_hashed_component(self):
        import numpy as np
        import torch
        from streamvln_atomic_state import (
            capture_training_state,
            gradients_are_clear,
            restore_training_state,
            training_state_hashes,
        )

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.policy = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
                self.value = torch.nn.Parameter(torch.tensor([0.5]))
                self.actor_version = 3
                self.behavior_version = 2

            def high_level_state(self):
                return {"policy": self.policy, "value": self.value}

            def load_high_level_state(self, state):
                with torch.no_grad():
                    self.policy.copy_(state["policy"])
                    self.value.copy_(state["value"])

        class Scaler:
            def __init__(self):
                self.value = {"scale": 7.0}

            def state_dict(self):
                return dict(self.value)

            def load_state_dict(self, state):
                self.value = dict(state)

        class Manager:
            pass

        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        core = Core()
        manager = Manager()
        manager.optimizer = torch.optim.AdamW(core.parameters(), lr=1e-3)
        manager.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            manager.optimizer, lr_lambda=lambda _: 1.0
        )
        manager.grad_scaler = Scaler()
        manager.optimizer_steps = 0
        (core.policy.square().sum() + core.value.square().sum()).backward()
        manager.optimizer.step()
        manager.optimizer_steps += 1
        manager.lr_scheduler.step()

        before = capture_training_state(manager, core)
        before_hashes = training_state_hashes(before)
        random.random()
        np.random.random()
        torch.rand(3)
        with torch.no_grad():
            core.policy.add_(9)
            core.value.mul_(4)
        (core.policy.sum() + core.value.sum()).backward()
        manager.optimizer.step()
        manager.lr_scheduler.step()
        manager.grad_scaler.value["scale"] = 3.0
        manager.optimizer_steps += 1
        core.actor_version += 1
        core.behavior_version += 1

        evidence = restore_training_state(manager, core, before)
        after = capture_training_state(manager, core)
        self.assertTrue(evidence["exact"])
        self.assertEqual(before_hashes, training_state_hashes(after))
        self.assertTrue(gradients_are_clear(core))


if __name__ == "__main__":
    unittest.main()
