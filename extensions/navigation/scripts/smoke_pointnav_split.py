"""Bounded CPU-only real Habitat service smoke for train and default val splits.

This script runs scripted actions; its outcome is a protocol/data-split check,
not a learned navigation result. It writes only inside --output-dir.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import select
import subprocess
from pathlib import Path


ACTIONS = ("stop", "move_forward", "turn_left", "turn_right")


def semantic_key(episode):
    return json.dumps([
        Path(episode["scene_id"]).name,
        episode["start_position"],
        episode["start_rotation"],
        episode["goals"],
    ], sort_keys=True, separators=(",", ":"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_line(process, timeout=120):
    readable, _, _ = select.select([process.stdout], [], [], timeout)
    if not readable:
        raise TimeoutError("PointNav service response timed out")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"PointNav service closed stdout (returncode={process.poll()})")
    result = json.loads(line)
    if "error" in result:
        raise RuntimeError("PointNav service: " + result["error"])
    return result


def request(process, payload):
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()
    return read_json_line(process, timeout=30)


def ensure_policy_response(response):
    assert len(response["obs"]) == 2
    assert all(math.isfinite(float(value)) for value in response["obs"])
    assert "position_xyz" not in response
    assert "position_xyz" not in response["info"]
    assert "goal_position_xyz" not in response["info"]


def assert_reward(previous_geodesic, step):
    info = step["info"]
    expected = (
        previous_geodesic - info["geodesic_distance"]
        + 10.0 * int(info["success"])
        - 0.01 - 0.2 * int(info["collision"])
    )
    assert math.isclose(step["reward"], expected, rel_tol=0, abs_tol=1e-5)


def run_service(*, python_bin, service_script, asset_root, split, indices,
                explicit_split, output_dir):
    dataset_path = (
        asset_root / "datasets/pointnav/habitat-test-scenes/v1" / split
        / f"{split}.json.gz"
    ).resolve(strict=True)
    with gzip.open(dataset_path, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]
    expected = [episodes[index] for index in indices]
    assert len({episode["scene_id"] for episode in expected}) == 1
    if split == "train":
        val_path = (
            asset_root / "datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"
        )
        with gzip.open(val_path, "rt", encoding="utf-8") as handle:
            val_episodes = json.load(handle)["episodes"]
        val_keys = {semantic_key(episode) for episode in val_episodes}
        if any(semantic_key(episode) in val_keys for episode in expected):
            raise ValueError("selected train episode overlaps val by scene/start/goal")

    command = [
        str(python_bin), "-B", str(service_script),
        "--asset-root", str(asset_root), "--max-steps", "4",
    ]
    if explicit_split:
        command.extend(("--split", split))
    if len(indices) > 1:
        command.extend(("--episode-indices", ",".join(map(str, indices))))
    else:
        command.extend(("--episode-index", str(indices[0])))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["LIBGL_ALWAYS_SOFTWARE"] = "1"
    trace = []
    stderr_path = output_dir / f"{split}-service-stderr.log"
    with stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, bufsize=1, env=environment,
        )
        try:
            ready = read_json_line(process)
            assert ready["ready"] is True
            assert tuple(ready["actions"]) == ACTIONS
            assert ready["split"] == split
            assert Path(ready["dataset_path"]).resolve() == dataset_path
            trace.append({"event": "ready", "response": ready})
            for index, expected_episode in zip(indices, expected):
                reset = request(process, {"cmd": "reset"})
                ensure_policy_response(reset)
                assert reset["info"]["episode_id"] == str(expected_episode["episode_id"])
                assert Path(reset["info"]["scene_id"]).name == Path(
                    expected_episode["scene_id"]
                ).name
                trace.append({"event": "reset", "index": index, "response": reset})
                diagnostic = request(process, {"cmd": "diagnostic"})
                assert len(diagnostic["position_xyz"]) == 3
                assert len(diagnostic["goal_position_xyz"]) == 3
                trace.append({"event": "diagnostic", "index": index,
                              "response": diagnostic})
                previous = reset["info"]["geodesic_distance"]
                for action in (1, 0):
                    step = request(process, {"cmd": "step", "action": action})
                    ensure_policy_response(step)
                    assert_reward(previous, step)
                    assert step["info"]["stop_called"] == (action == 0)
                    assert step["info"]["episode_id"] == str(
                        expected_episode["episode_id"]
                    )
                    if action == 0:
                        assert step["terminated"] is True
                        assert step["truncated"] is False
                    else:
                        assert step["terminated"] is False
                        assert step["truncated"] is False
                    trace.append({"event": "step", "index": index,
                                  "action": action, "response": step})
                    previous = step["info"]["geodesic_distance"]
                frozen = request(process, {"cmd": "step", "action": 3})
                assert frozen["reward"] == 0.0
                assert not frozen["terminated"] and not frozen["truncated"]
                assert frozen["info"]["terminal_frozen"] is True
                trace.append({"event": "frozen", "index": index,
                              "response": frozen})
            closed = request(process, {"cmd": "close"})
            assert closed == {"closed": True}
            assert process.wait(timeout=30) == 0
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    trace_path = output_dir / f"{split}-trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for event in trace:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
    terminal = [event["response"]["info"] for event in trace
                if event["event"] == "step" and event["action"] == 0]
    return {
        "split": split,
        "explicit_split": explicit_split,
        "indices": indices,
        "episode_ids": [item["episode_id"] for item in terminal],
        "scene_ids": [item["scene_id"] for item in terminal],
        "successes": [item["success"] for item in terminal],
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "trace_file": trace_path.name,
        "trace_sha256": sha256(trace_path),
        "stderr_file": stderr_path.name,
        "stderr_sha256": sha256(stderr_path),
        "service_script_sha256": sha256(service_script),
        "cpu_only_env": {"CUDA_VISIBLE_DEVICES": "",
                         "LIBGL_ALWAYS_SOFTWARE": "1"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--service-script", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-indices", type=int, nargs=2, default=[50, 51])
    args = parser.parse_args()
    if len(set(args.train_indices)) != 2:
        parser.error("--train-indices requires two distinct indices")
    # Keep the environment's python symlink intact; resolve() can escape its prefix.
    python_bin = args.python_bin.expanduser().absolute()
    if not python_bin.is_file():
        raise FileNotFoundError(python_bin)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        run_service(
            python_bin=python_bin,
            service_script=args.service_script.resolve(strict=True),
            asset_root=args.asset_root.resolve(strict=True), split="train",
            indices=args.train_indices, explicit_split=True, output_dir=output_dir,
        ),
        run_service(
            python_bin=python_bin,
            service_script=args.service_script.resolve(strict=True),
            asset_root=args.asset_root.resolve(strict=True), split="val",
            indices=[0], explicit_split=False, output_dir=output_dir,
        ),
    ]
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps({"runs": runs}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(result_path)


if __name__ == "__main__":
    main()
