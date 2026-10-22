import importlib.util
import math
from pathlib import Path
import unittest


MODULE = Path(__file__).parents[1] / "integrations" / "rlinf_v03" / "streamvln_chunk_ppo.py"
SPEC = importlib.util.spec_from_file_location("streamvln_chunk_ppo", MODULE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class StreamVLNChunkPPOTest(unittest.TestCase):
    def test_collapse_ignores_padded_rewards(self):
        rewards, durations = MOD.collapse_chunk_rewards(
            [[1.0, 2.0, 99.0, 99.0], [3.0, 4.0, 5.0, 6.0]],
            [[True, True, False, False], [True, True, True, True]],
            0.5,
        )
        self.assertEqual(durations, [2, 4])
        self.assertEqual(rewards, [2.0, 7.0])

    def test_variable_duration_gae_uses_physical_step_powers(self):
        advantage, returns, deltas = MOD.variable_duration_gae(
            chunk_rewards=[2.0, 7.0],
            values=[0.25, 0.5],
            next_values=[0.5, 0.0],
            durations=[2, 4],
            dones=[False, True],
            gamma=0.5,
            gae_lambda=0.8,
        )
        expected_delta_1 = 2.0 + 0.5**2 * 0.5 - 0.25
        expected_delta_2 = 7.0 - 0.5
        expected_adv_2 = expected_delta_2
        expected_adv_1 = expected_delta_1 + (0.5 * 0.8) ** 2 * expected_adv_2
        self.assertTrue(math.isclose(deltas[0], expected_delta_1))
        self.assertTrue(math.isclose(deltas[1], expected_delta_2))
        self.assertTrue(math.isclose(advantage[0], expected_adv_1))
        self.assertTrue(math.isclose(advantage[1], expected_adv_2))
        self.assertTrue(math.isclose(returns[0], expected_adv_1 + 0.25))

    def test_stop_like_single_step_has_gamma_bootstrap(self):
        advantage, _, delta = MOD.variable_duration_gae(
            [1.0], [0.2], [0.4], [1], [False], 0.9, 0.95
        )
        self.assertTrue(math.isclose(delta[0], 1.0 + 0.9 * 0.4 - 0.2))
        self.assertEqual(advantage, delta)

    def test_rejects_nonprefix_mask(self):
        with self.assertRaisesRegex(ValueError, "prefix"):
            MOD.collapse_chunk_rewards(
                [[1.0, 2.0, 3.0, 4.0]],
                [[True, False, True, False]],
                0.99,
            )


if __name__ == "__main__":
    unittest.main()
