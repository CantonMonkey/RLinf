from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"


class StreamVLNOriginalPPOStagingTest(unittest.TestCase):
    def test_policy_is_original_and_language_only(self):
        text = (STAGE / "streamvln_ppo_policy.py").read_text(encoding="utf-8")
        self.assertIn('source_kind": "upstream-original"', text)
        self.assertIn('"memory_adapter": False', text)
        self.assertIn('rf"^model\\.layers\\.', text)
        self.assertIn('("q_proj", "v_proj")', text)
        self.assertNotIn("navigation_policy", text)
        self.assertNotIn("ttt_fast_state", text.lower())

    def test_rlinf_generic_lora_is_disabled(self):
        text = (STAGE / "streamvln_ppo.yaml").read_text(encoding="utf-8")
        self.assertIn("is_lora: false", text)
        self.assertIn("lora_layers: [24, 25, 26, 27]", text)
        self.assertIn("lora_targets: [q_proj, v_proj]", text)

    def test_runner_pins_upstream_and_fsdp(self):
        text = (STAGE / "run_streamvln_ppo_g3.py").read_text(encoding="utf-8")
        self.assertIn("e48f6ff7e9201d93aae003e8f64b04c00cec13bc", text)
        self.assertIn("G3FSDPManager(FSDPModelManager)", text)
        self.assertIn('"wrapper": type(manager.model).__name__', text)
        self.assertIn('"secondary_post_training_used": False', text)
        self.assertIn('"bootstrap": "gamma^k V_next"', text)
        self.assertIn('"trace": "(gamma*lambda)^k"', text)
        self.assertIn("def canonical_lora_state", text)
        self.assertIn('len(saved_lora) == 16', text)
        self.assertIn('adapter_max_error == 0.0', text)
        self.assertIn('"adapter_load_verified": adapter_load_verified', text)


if __name__ == "__main__":
    unittest.main()
