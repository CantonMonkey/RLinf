import ast
import hashlib
import json
from pathlib import Path
import unittest


class StreamVLNG721ProtocolTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    INTEGRATION = ROOT / "integrations" / "rlinf_v03"

    def test_manifest_freezes_disjoint_decision_and_audit_arms(self):
        path = self.INTEGRATION / "streamvln_g721_experiment.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "daea9d8c8891e24550008069ff1f820ce649f02ab45c6aab45886729da53cdda",
        )
        manifest = json.loads(path.read_text())
        protocol = manifest["protocol"]
        self.assertEqual(
            manifest["selection"]["seeds"], [2026092811, 2026092812, 2026092813]
        )
        self.assertEqual(set(protocol["arms"]), {"2026092811", "2026092812", "2026092813"})
        scenes = []
        for arm in protocol["arms"].values():
            self.assertEqual(len(arm["decision"]), 4)
            self.assertEqual(len(arm["audit"]), 4)
            self.assertFalse(
                {row["scene"] for row in arm["decision"]}
                & {row["scene"] for row in arm["audit"]}
            )
            scenes.extend(
                row["scene"] for role in ("decision", "audit") for row in arm[role]
            )
        self.assertEqual(len(scenes), 24)
        self.assertEqual(len(set(scenes)), 24)
        self.assertFalse(set(scenes) & set(manifest["selection"]["excluded_scenes"]))
        self.assertEqual(protocol["candidate_update_limit"], 3)
        self.assertTrue(protocol["stop_candidate_loop_after_first_rejection"])
        self.assertEqual(protocol["audit_contract"]["ratio_max_deviation"], 0.5)
        self.assertEqual(protocol["audit_contract"]["structured_kl_max"], 0.1)
        self.assertFalse(protocol["audit_contract"]["may_affect_commit_stop_or_checkpoint"])
        self.assertEqual(protocol["val_unseen_episode_count"], 0)
        self.assertFalse(manifest["g8_ready"])

    def test_runner_has_restore_probe_and_nonselective_audit_boundary(self):
        path = self.INTEGRATION / "run_streamvln_safety_transfer_g721.py"
        source = path.read_text()
        ast.parse(source)
        self.assertIn("reference_probe = fixed_continuation_probe", source)
        self.assertIn("continuation = compare_probes(reference_probe, restored_probe)", source)
        self.assertIn("probe_cleanup_restore = restore_training_state", source)
        self.assertIn("audit_state = evaluate_state", source)
        self.assertIn("audit_restore = restore_training_state", source)
        self.assertIn("all_rejected_continuations_equal_reference", source)
        self.assertIn("final_state_equals_last_accepted_state", source)
        self.assertNotIn("sample_target_mode", source)
        self.assertNotIn("save_head_checkpoint", source)
        self.assertNotIn('split="val_unseen"', source)
        self.assertNotIn("val_unseen = collect_phase", source)


if __name__ == "__main__":
    unittest.main()
