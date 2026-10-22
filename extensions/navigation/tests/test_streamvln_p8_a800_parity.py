"""Narrow local tests for the one-shot A800 parity entrypoint."""

import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(STAGE))

import run_streamvln_p8_a800_parity as parity  # noqa: E402
import streamvln_p8_clean_e48_reference as reference  # noqa: E402


def _audit(seed: str, *, dtype="torch.bfloat16", shape=None) -> dict:
    return {
        "dtype": dtype,
        "shape": [1, 1] if shape is None else shape,
        "sha256": (seed * 64)[:64],
        "nbytes": 2,
    }


def _side() -> dict:
    generation = {
        "output_ids": _audit("a", dtype="torch.int64"),
        "decoded": "↑←→STOP",
        "parsed_queue": [1, 2, 3, 0],
        "pop_cadence": [1, 2, 3, 0],
        "past_key_values_structure": {"type": "tuple"},
        "model_curr_t": 1,
        "inputs_embeds_seq_len": 17,
    }
    return {
        "preprocess": {
            "input_ids": _audit("b", dtype="torch.int64"),
            "tensors": {
                "images": _audit("c", shape=[1, 1, 3, 384, 384]),
                "depths": _audit("d", shape=[1, 1, 384, 384]),
                "poses": _audit("e", shape=[1, 1, 4, 4]),
                "intrinsics": _audit("f", shape=[1, 1, 4, 4]),
            },
        },
        "initial_generation": copy.deepcopy(generation),
        "continuation": {**copy.deepcopy(generation), "model_curr_t": 2, "inputs_embeds_seq_len": 29},
        "history_and_reset": {
            "frame_count": 9,
            "frame_steps": [0, 4, 8, 12, 16, 20, 24, 28, 32],
            "tensors": {
                "images": _audit("0", shape=[1, 9, 3, 384, 384]),
                "depths": _audit("1", shape=[1, 9, 384, 384]),
                "poses": _audit("2", shape=[1, 9, 4, 4]),
                "intrinsics": _audit("3", shape=[1, 9, 4, 4]),
            },
            "reset_step": 32,
            "reset_state": {"curr_t": 2, "cache_nonempty": True, "curr_t_after": 0, "cache_empty_after": True},
            "boundary_generation": {**copy.deepcopy(generation), "model_curr_t": 1},
            "exact_queue_exhausted": True,
            "queue_cross_reset_state": {
                "physical_step": 33,
                "reset_step": 32,
                "crossed": True,
                "residual_after_reset": 1,
                "pending_after_pop33": 0,
            },
            "queue_cross_reset_rejected": True,
        },
    }


class A800ParityTests(unittest.TestCase):
    def test_manifest_mismatch_rejects_before_heavy_import_and_preserves_output(self):
        release_manifest = ROOT.parent / "P8_RELEASE_ARTIFACT_MANIFEST.json"
        if not release_manifest.is_file():
            self.skipTest("Historical release manifest is not included in the public source export")
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            output.write_text('{"status":"older"}\n', encoding="utf-8")
            manifest = Path(directory) / "wrong-manifest.json"
            wrong_manifest = json.loads(
                release_manifest.read_text(encoding="utf-8")
            )
            wrong_manifest["source"]["python_files"][0]["sha256"] = "0" * 64
            manifest.write_text(
                json.dumps(wrong_manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            args = parity.build_parser().parse_args(
                [
                    "--source-root", str(Path(directory) / "missing-source"),
                    "--import-root", str(Path(directory) / "missing-import"),
                    "--checkpoint", str(Path(directory) / "missing-checkpoint"),
                    "--tokenizer", str(Path(directory) / "missing-tokenizer"),
                    "--vision", str(Path(directory) / "missing-vision"),
                    "--manifest", str(manifest),
                    "--device", "cuda:3",
                    "--local-rank", "3",
                    "--output", str(output),
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "manifest digest"):
                parity.run(args, heavy_loader=lambda _: calls.append("heavy"))
            self.assertEqual(calls, [])
            self.assertEqual(output.read_text(encoding="utf-8"), '{"status":"older"}\n')

    def test_receipt_schema_and_comparator_reject_token_or_tensor_drift(self):
        side = _side()
        comparison = parity.compare_parity_sides(side, copy.deepcopy(side))
        self.assertEqual(comparison["status"], "pass")
        preflight = {
            "anchor": "f" * 64,
            "manifest_path": "/release/manifest.json",
            "source_root": "/source",
            "import_root": "/source/streamvln",
            "device": "cuda:3",
            "verification": {
                "source": {"evaluator_sha256": "1" * 64, "model_source_sha256": "2" * 64},
                "artifacts": {"checkpoint": {"content_manifest_sha256": "3" * 64}},
            },
        }
        receipt = parity.build_receipt(preflight_receipt=preflight, reference=side, adapter=copy.deepcopy(side))
        self.assertEqual(receipt["schema_name"], parity.SCHEMA_NAME)
        self.assertEqual(receipt["status"], "pass")
        token_drift = copy.deepcopy(side)
        token_drift["initial_generation"]["output_ids"]["sha256"] = "9" * 64
        with self.assertRaisesRegex(RuntimeError, "parity drift"):
            parity.compare_parity_sides(side, token_drift)
        tensor_drift = copy.deepcopy(side)
        tensor_drift["preprocess"]["tensors"]["images"]["sha256"] = "8" * 64
        with self.assertRaisesRegex(RuntimeError, "parity drift"):
            parity.compare_parity_sides(side, tensor_drift)

    def test_cli_requires_explicit_arguments_and_reads_only_manifest_path(self):
        with self.assertRaises(SystemExit):
            parity.build_parser().parse_args([])
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text('{"schema_name":"fixture"}\n', encoding="utf-8")
            args = parity.build_parser().parse_args(
                [
                    "--source-root", directory,
                    "--import-root", str(Path(directory) / "streamvln"),
                    "--checkpoint", "checkpoint",
                    "--tokenizer", "tokenizer",
                    "--vision", "vision",
                    "--manifest", str(manifest),
                    "--device", "cuda:7",
                    "--local-rank", "7",
                    "--output", str(Path(directory) / "receipt.json"),
                ]
            )
            self.assertEqual(args.device, "cuda:7")
            self.assertEqual(args.local_rank, 7)
            self.assertEqual(parity._read_manifest(args.manifest), {"schema_name": "fixture"})
            evaluator = Path(directory) / "streamvln_eval.py"
            evaluator.write_text(
                "class VLNEvaluator:\n"
                "    def preprocess_qwen(self, sources, tokenizer: transformers.PreTrainedTokenizer, has_image=False, add_system=False):\n"
                "        return sources, []\n"
                "    def parse_actions(self, output):\n"
                "        return []\n",
                encoding="utf-8",
            )
            namespace = reference.load_verified_evaluator_namespace(
                evaluator, torch_module=object()
            )
            owner = object.__new__(namespace.VLNEvaluator)
            self.assertEqual(
                owner.preprocess_qwen([[{"from": "human", "value": ""}]], object()),
                ([[{"from": "human", "value": ""}]], []),
            )
            generate_parameters = inspect.signature(
                reference.CleanE48ReferenceRunner.generate
            ).parameters
            self.assertIn("conjunction", generate_parameters)
            self.assertTrue(generate_parameters["conjunction"].kind.name.startswith("KEYWORD"))


if __name__ == "__main__":
    unittest.main()
