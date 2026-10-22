from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import navigation_rlt_parallel as parallel  # noqa: E402
from navigation_rlt_parallel import (  # noqa: E402
    PLAN_SCHEMA,
    PLAN_VERSION,
    ParallelContractError,
    build_plan,
    main,
    sha256_file,
)


def _episodes(prefix: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "episode_id": f"{prefix}-{index}",
            "scene": f"scene-{index // 4}",
            "instruction_text_sha256": hashlib.sha256(
                f"{prefix}-instruction-{index}".encode("utf-8")
            ).hexdigest(),
        }
        for index in range(count)
    ]


def _write_split(path: Path, development: list[dict[str, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "upstream_split_read": "train",
                "validation_split_reads": 0,
                "splits": {
                    "train_update": {"episodes": _episodes("update", 6)},
                    "train_development": {"episodes": development},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


class NavigationRLTParallelEvalPlanTests(unittest.TestCase):
    def test_atomic_writer_preserves_competing_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            output_path = directory / "plan.json"
            real_link = parallel.os.link

            def competing_link(source, target, **kwargs):
                Path(target).write_bytes(b"sentinel")
                return real_link(source, target, **kwargs)

            with patch.object(parallel.os, "link", side_effect=competing_link):
                with self.assertRaises(FileExistsError):
                    parallel.atomic_json(output_path, {"replacement": True})

            self.assertEqual(output_path.read_bytes(), b"sentinel")
            self.assertEqual(list(directory.glob("plan.json.staging-*")), [])

    def test_atomic_writer_rejects_dangling_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            output_path = directory / "plan.json"
            missing = directory / "missing.json"
            try:
                os.symlink(missing, output_path)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"symbolic links unavailable: {error}")

            with self.assertRaises(FileExistsError):
                parallel.atomic_json(output_path, {"replacement": True})

            self.assertTrue(output_path.is_symlink())
            self.assertFalse(missing.exists())

    def test_create_cli_does_not_replace_reviewed_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            split_path = directory / "split.json"
            output_path = directory / "eval-plan.json"
            _write_split(split_path, _episodes("development", 32))
            output_path.write_bytes(b"reviewed sentinel")

            with self.assertRaises(FileExistsError):
                main(
                    [
                        "create",
                        "--split-manifest",
                        str(split_path),
                        "--expected-split-sha256",
                        sha256_file(split_path),
                        "--run-id",
                        "navigation-rlt-matched-eval",
                        "--split-label",
                        "train_development",
                        "--worker-count",
                        "1",
                        "--episode-count",
                        "32",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(output_path.read_bytes(), b"reviewed sentinel")

    def test_explicit_train_development_cli_builds_evaluator_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            split_path = directory / "split.json"
            output_path = directory / "eval-plan.json"
            _write_split(split_path, _episodes("development", 32))

            with redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "create",
                        "--split-manifest",
                        str(split_path),
                        "--expected-split-sha256",
                        sha256_file(split_path),
                        "--run-id",
                        "navigation-rlt-matched-eval",
                        "--split-label",
                        "train_development",
                        "--worker-count",
                        "1",
                        "--episode-count",
                        "32",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(result, 0)
            plan = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["schema_name"], PLAN_SCHEMA)
            self.assertEqual(plan["schema_version"], PLAN_VERSION)
            self.assertEqual(plan["split_label"], "train_development")
            self.assertEqual(plan["validation_split_reads"], 0)
            self.assertEqual(plan["episode_count"], 32)
            self.assertEqual(
                [row["episode_order"] for row in plan["workers"][0]["episodes"]],
                list(range(32)),
            )

    def test_train_update_remains_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            split_path = Path(directory_name) / "split.json"
            _write_split(split_path, _episodes("development", 32))

            plan = build_plan(
                split_path,
                run_id="navigation-rlt-rollout",
                expected_split_sha256=sha256_file(split_path),
                worker_count=3,
                expected_episode_count=6,
            )

            self.assertEqual(plan["split_label"], "train_update")
            self.assertEqual(plan["episode_count"], 6)

    def test_train_development_fails_closed_on_count_and_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            split_path = Path(directory_name) / "split.json"
            development = _episodes("development", 32)
            _write_split(split_path, development)
            split_sha = sha256_file(split_path)

            with self.assertRaises(ParallelContractError):
                build_plan(
                    split_path,
                    run_id="navigation-rlt-matched-eval",
                    expected_split_sha256=split_sha,
                    worker_count=1,
                    expected_episode_count=31,
                    split_label="train_development",
                )

            development[0]["scene"] = "tampered-scene"
            _write_split(split_path, development)
            with self.assertRaises(ParallelContractError):
                build_plan(
                    split_path,
                    run_id="navigation-rlt-matched-eval",
                    expected_split_sha256=split_sha,
                    worker_count=1,
                    expected_episode_count=32,
                    split_label="train_development",
                )

            development[-1] = dict(development[0])
            _write_split(split_path, development)
            with self.assertRaises(ParallelContractError):
                build_plan(
                    split_path,
                    run_id="navigation-rlt-matched-eval",
                    expected_split_sha256=sha256_file(split_path),
                    worker_count=1,
                    expected_episode_count=32,
                    split_label="train_development",
                )


if __name__ == "__main__":
    unittest.main()
