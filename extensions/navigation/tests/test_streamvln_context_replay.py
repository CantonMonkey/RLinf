"""Contract tests for independent StreamVLN context and replay rows."""

import sys
import unittest
from pathlib import Path


STAGING = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03"
sys.path.insert(0, str(STAGING))

from streamvln_context import ContextLedger, SessionKey
from streamvln_replay import ReplaySpec, pack_snapshot, stack_rows, unpack_row, unstack_row


class StreamVLNContextReplayTests(unittest.TestCase):
    def make_ledger(self, env_id=0, serial=1, instruction="walk to the table"):
        return ContextLedger(
            SessionKey(env_id, serial, f"scene-{env_id}", f"episode-{serial}"),
            instruction,
            (2, 2, 3),
        )

    @staticmethod
    def frame(value):
        return bytes([value] * 12)

    def test_two_environments_never_share_history(self):
        first = self.make_ledger(env_id=0, instruction="left")
        second = self.make_ledger(env_id=1, instruction="right")
        first.record_observation(0, self.frame(1), (2, 2, 3))
        first.set_proposal([1, 2, 3, 1])
        second.record_observation(0, self.frame(9), (2, 2, 3))
        second.set_proposal([3, 3, 1, 0])
        self.assertEqual(first.snapshot()["block_frames"], [self.frame(1)])
        self.assertEqual(second.snapshot()["block_frames"], [self.frame(9)])
        first.acknowledge([1, 2, 3, 1])
        first.record_observation(4, self.frame(2), (2, 2, 3))
        first.set_proposal([0])
        self.assertEqual(first.snapshot()["block_actions"], [1, 2, 3, 1])
        self.assertEqual(second.step, 0)
        self.assertEqual(second.pending_actions, [3, 3, 1, 0])

    def test_block_boundary_selects_exact_eight_history_frames(self):
        ledger = self.make_ledger()
        proposal = [1, 2, 3, 1]
        for step in range(0, 33, 4):
            ledger.record_observation(step, self.frame(step // 4), (2, 2, 3))
            ledger.set_proposal(proposal)
            if step < 32:
                ledger.acknowledge(proposal)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["block_start"], 32)
        self.assertEqual(snapshot["history_steps"], list(range(0, 32, 4)))
        self.assertEqual(snapshot["block_steps"], [32])
        self.assertEqual(len(snapshot["history_frames"]), 8)
        self.assertEqual(snapshot["block_actions"], [])

    def test_fixed_rows_survive_stack_shuffle_and_independent_restore(self):
        spec = ReplaySpec(height=2, width=2, instruction_bytes=32, scene_bytes=16, episode_bytes=16)
        rows = []
        originals = []
        for env_id, pixel in ((0, 4), (1, 7)):
            ledger = self.make_ledger(env_id=env_id, instruction=f"go {env_id}")
            ledger.record_observation(0, self.frame(pixel), (2, 2, 3))
            ledger.set_proposal([1, 3, 0])
            snapshot = ledger.snapshot()
            originals.append(snapshot)
            rows.append(pack_snapshot(snapshot, spec))
        batch = stack_rows(rows)
        restored = [
            unpack_row(unstack_row(batch, index), spec)
            for index in (1, 0)
        ]
        self.assertEqual(restored[0]["env_id"], 1)
        self.assertEqual(restored[0]["block_frames"], originals[1]["block_frames"])
        self.assertEqual(restored[1]["env_id"], 0)
        self.assertEqual(restored[1]["current_actions"], [1, 3, 0])
        self.assertEqual(len(rows[0]["history_rgb"]), 8)
        self.assertEqual(len(rows[0]["block_rgb"]), 8)
        self.assertEqual(len(rows[0]["block_actions"]), 32)
        self.assertEqual(len(rows[0]["current_actions"]), 4)

    def test_rejects_action_mismatch_and_non_aligned_observation(self):
        ledger = self.make_ledger()
        ledger.record_observation(0, self.frame(1), (2, 2, 3))
        ledger.set_proposal([1, 2, 0])
        with self.assertRaisesRegex(ValueError, "disagree"):
            ledger.acknowledge([1, 3])
        ledger.acknowledge([1, 2])
        with self.assertRaisesRegex(ValueError, "four-action chronology"):
            ledger.record_observation(2, self.frame(2), (2, 2, 3))


if __name__ == "__main__":
    unittest.main()