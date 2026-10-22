"""Build and verify the frozen train-only manifest for the StreamVLN P8 diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


SELECTION_SEED = 2026092811
BASE_WRAPPER_INIT_SEED = 2026092811
RESIDUAL_INIT_SEED = 2026092812
TINY_TRAIN_COUNT = 8
EQUIVALENCE_COUNT = 4
TRAIN_DEVELOPMENT_COUNT = 8
TRAIN_SHA256 = "340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea"
SELECTION_NAMESPACE = "p8g8-diagnostic-v1"
RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
STREAM_COMMIT = "e48f6ff7e9201d93aae003e8f64b04c00cec13bc"
HABITAT_SIM_COMMIT = "f179b584bcd713c5a2a998132211e2cae881d6d1"
MANIFEST_LINE_NORMALIZATION = 'MANIFEST_SHA256 = "<NORMALIZED>"'

REMOTE_PATHS = {
    "rlinf_root": "/srv/navigation/repos/RLinf-v0.3-streamvln-p8",
    "integration_root": "/srv/navigation/repos/RLinf-v0.3-streamvln-p8/p8/scripts",
    "stream_source": "/srv/navigation/repos/StreamVLN-upstream-g4",
    "asset_root": "/cache/navigation-example",
    "base_model": "/cache/navigation-example/models/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3-f1f76c66083c",
    "tokenizer": "/cache/navigation-example/hf/hub/models--lmms-lab--LLaVA-Video-7B-Qwen2/snapshots/013210b3aff822f1558b166d39c1046dd109520f",
    "vision_tower": "/cache/navigation-example/hf/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    "rlinf_python": "/srv/navigation/envs/rlinf-v03-cu118/bin/python",
    "habitat_python": "/srv/navigation/envs/habitat-cpu-egl/bin/python3.9",
    "habitat_server": "/srv/navigation/repos/RLinf-v0.3-streamvln-p8/scripts/habitat_r2r_service.py",
    "habitat_sim_root": "/srv/navigation/repos/habitat-sim-cpu-egl",
    "habitat_sim_python": "/srv/navigation/repos/habitat-sim-cpu-egl/src_python",
    "habitat_env_lib": "/srv/navigation/envs/habitat-cpu-egl/lib",
    "egl_vendor_json": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
}

RUNTIME_IDENTITIES = {
    "rlinf": {
        "python_path": REMOTE_PATHS["rlinf_python"],
        "python_sha256": "1482b4166a0b5c6a1bf4253cf8637c973b362aa70ca3b556a2d2234f4bff96ec",
        "python_version": "3.10.6",
        "packages": {"numpy": "1.26.4", "omegaconf": "2.3.1", "peft": "0.21.0", "torch": "2.6.0+cu118", "transformers": "4.57.6"},
    },
    "habitat": {
        "python_path": REMOTE_PATHS["habitat_python"],
        "python_sha256": "19ea30fb0b3b8f5b2677ce88e5b8bbca1849584c5d2ebeb2306221a1714f4096",
        "python_version": "3.9.19",
        "packages": {"gym": "0.23.0", "habitat-lab": "0.2.4", "numpy": "1.26.4"},
    },
    "habitat_sim_commit": HABITAT_SIM_COMMIT,
    "egl_vendor_json_sha256": "cd3881df4227c13ca7f9cfe6c78820f21d5f7a00a41b299214f6f827e63be369",
}

MODEL_INPUT_FILES = {
    "base_model": [
        (".complete", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        ("README.md", 5174, "0336368f8274a5ed09f54cee8d56a851ecee96026d9f391a6735762c16608a60"),
        ("SOURCE.txt", 122, "a2de678fdb31390a03bb4fec5cb7960f08ce20e580dca5b7860b25a48ca4d81c"),
        ("config.json", 4416, "58b839efed10e0bca3fb17d411400dceca8c1c27d48b202ed9c48a976414647a"),
        ("generation_config.json", 243, "24216e06e770ea4f6ce9f88ab782043391d886c0cf8fae9e775bcf0148febb6e"),
        ("model-00001-of-00004.safetensors", 4877660776, "6a1a4fc8ba9ab5e8832d2d23dc6d21d18bf1f84daf03b904409721f93927a87a"),
        ("model-00002-of-00004.safetensors", 4932751008, "d7b781ef3772d39537f81fb87a31b5db04118fbf9edcd78dbe60ad2d67d77f49"),
        ("model-00003-of-00004.safetensors", 4994571904, "9a79a47a31c62fc2beab869cebf2e7bf956dbf630c14ef6017148fa7e41fe628"),
        ("model-00004-of-00004.safetensors", 1255812224, "5e9eb8f127c0ec37b175c266b0cbae3106a591ec3d93698aee500f7735f041fe"),
        ("model.safetensors.index.json", 80769, "d1f86714849892ee46d17fd82a90b9f2690e0083d2b2339ccd827673d4c5c2e3"),
    ],
    "tokenizer": [
        ("added_tokens.json", 100, "416774498bd7218f05be8c7faa0b375eccdcacf66cb5ab2766bdeff84e466cf5"),
        ("generation_config.json", 248, "eea6735f058019c875227b53de54f34353f01bbdbb0e9d8fc6474567e1334c29"),
        ("merges.txt", 1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
        ("special_tokens_map.json", 367, "f4f79e08d97f4d1c87f8d89264f525c8789da3b73b3bb55d1e12f692f41a7b1b"),
        ("tokenizer.json", 7028015, "f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8"),
        ("tokenizer_config.json", 1538, "b36d4cfd510e8942ba3a5f4733026c5f1697b2810331be4fc9459c672e451e6a"),
        ("vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    ],
    "vision_tower": [
        ("config.json", 576, "adc04928d8fd19a61822584fe0cf2e813e5ebac17f3e49fb1ea096860ae6457b"),
        ("model.safetensors", 3511950624, "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a"),
        ("preprocessor_config.json", 368, "f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8"),
    ],
}

OVERLAY_FILES = {
    "integration/build_streamvln_p8_train_diagnostic_manifest.py": ("integration_root", "build_streamvln_p8_train_diagnostic_manifest.py"),
    "integration/run_streamvln_p8_train_diagnostic.py": ("integration_root", "run_streamvln_p8_train_diagnostic.py"),
    "integration/streamvln_p8_live_diagnostic.py": ("integration_root", "streamvln_p8_live_diagnostic.py"),
    "integration/streamvln_p8_residual_policy.py": ("integration_root", "streamvln_p8_residual_policy.py"),
    "integration/streamvln_p8_topology.py": ("integration_root", "streamvln_p8_topology.py"),
    "integration/streamvln_executor_contract.py": ("integration_root", "streamvln_executor_contract.py"),
    "integration/streamvln_ppo_policy.py": ("integration_root", "streamvln_ppo_policy.py"),
    "integration/streamvln_topology.py": ("integration_root", "streamvln_topology.py"),
    "rlinf/envs/__init__.py": ("rlinf_root", "rlinf/envs/__init__.py"),
    "rlinf/envs/vln/vln_env.py": ("rlinf_root", "rlinf/envs/vln/vln_env.py"),
    "rlinf/envs/streamvln/__init__.py": ("rlinf_root", "rlinf/envs/streamvln/__init__.py"),
    "rlinf/envs/streamvln/streamvln_env.py": ("rlinf_root", "rlinf/envs/streamvln/streamvln_env.py"),
    "rlinf/envs/streamvln/streamvln_executor_contract.py": ("rlinf_root", "rlinf/envs/streamvln/streamvln_executor_contract.py"),
    "scripts/habitat_r2r_service.py": ("rlinf_root", "scripts/habitat_r2r_service.py"),
    "scripts/habitat_r2r_rgb_instruction_smoke.py": ("rlinf_root", "scripts/habitat_r2r_rgb_instruction_smoke.py"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_implementation_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.name == "run_streamvln_p8_train_diagnostic.py":
        text = data.decode("utf-8")
        text, count = re.subn(r'^MANIFEST_SHA256 = [^\r\n]+$', MANIFEST_LINE_NORMALIZATION, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError("P8 runner has no unique MANIFEST_SHA256 line")
        data = text.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_identities(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        raw = json.load(source)
    episodes = raw.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"R2R train split has no episodes: {path}")
    rows, seen = [], set()
    for index, episode in enumerate(episodes):
        scene_id = str(episode["scene_id"])
        row = {"index": index, "episode_id": str(episode["episode_id"]), "scene": Path(scene_id).stem, "source_scene_id": scene_id}
        identity = (row["index"], row["episode_id"], row["scene"])
        if identity in seen:
            raise ValueError(f"duplicate R2R train identity: {identity}")
        seen.add(identity)
        rows.append(row)
    return rows


def identity_rank(seed: int, role: str, row: dict) -> str:
    text = f"{SELECTION_NAMESPACE}|{seed}|{role}|{row['scene']}|{row['index']}|{row['episode_id']}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_distinct_scenes(rows: list[dict], *, seed: int, role: str, count: int, excluded_scenes: frozenset[str] = frozenset()) -> list[dict]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["scene"] not in excluded_scenes:
            by_scene[row["scene"]].append(row)
    if len(by_scene) < count:
        raise ValueError(f"insufficient distinct scenes for {role}: {len(by_scene)} < {count}")
    candidates = [min(scene_rows, key=lambda row: identity_rank(seed, role, row)) for scene_rows in by_scene.values()]
    return sorted(candidates, key=lambda row: identity_rank(seed, role, row))[:count]


def public_identity(row: dict) -> dict:
    return {key: row[key] for key in ("index", "episode_id", "scene")}


def normalized_r2r_scene_path(source_scene_id: str) -> PurePosixPath:
    """Return the canonical MP3D-relative GLB path or reject the identity."""

    if not isinstance(source_scene_id, str) or not source_scene_id or "\\" in source_scene_id:
        raise ValueError(f"unexpected R2R scene path: {source_scene_id}")
    relative_scene = source_scene_id
    for prefix in ("data/scene_datasets/", "scene_datasets/"):
        if relative_scene.startswith(prefix):
            relative_scene = relative_scene.removeprefix(prefix)
            break
    candidate = PurePosixPath(relative_scene)
    parts = candidate.parts
    if (
        candidate.is_absolute()
        or len(parts) != 3
        or parts[0] != "mp3d"
        or parts[1] in {"", ".", ".."}
        or parts[2] != f"{parts[1]}.glb"
    ):
        raise ValueError(f"unexpected R2R scene path: {source_scene_id}")
    return candidate


def selected_scene_assets(asset_root: Path, rows: list[dict]) -> list[dict]:
    result, seen = [], set()
    for row in rows:
        if row["scene"] in seen:
            continue
        seen.add(row["scene"])
        source_scene_id = row["source_scene_id"]
        relative_scene = normalized_r2r_scene_path(source_scene_id)
        if row["scene"] != relative_scene.parts[1]:
            raise ValueError(f"R2R scene identity mismatch: {source_scene_id}")
        scene_root = (asset_root / "scene_datasets" / "mp3d").resolve(strict=True)
        glb = (scene_root / relative_scene.parts[1] / relative_scene.parts[2]).resolve(strict=True)
        if not glb.is_relative_to(scene_root):
            raise ValueError(f"R2R scene escapes MP3D root: {source_scene_id}")
        navmesh = glb.with_suffix(".navmesh").resolve(strict=True)
        if not navmesh.is_relative_to(scene_root):
            raise ValueError(f"R2R navmesh escapes MP3D root: {source_scene_id}")
        result.append({
            "scene": row["scene"],
            "glb": {"path": str(glb), "bytes": glb.stat().st_size, "sha256": sha256_file(glb)},
            "navmesh": {"path": str(navmesh), "bytes": navmesh.stat().st_size, "sha256": sha256_file(navmesh)},
        })
    if len(result) != 16:
        raise RuntimeError("P8 manifest must bind exactly 16 distinct selected scenes")
    return result


def model_input_inventory() -> dict:
    return {
        name: {"root": REMOTE_PATHS[name], "files": [
            {"relative_path": relative, "bytes": size, "sha256": digest}
            for relative, size, digest in files
        ]}
        for name, files in MODEL_INPUT_FILES.items()
    }


def overlay_inventory() -> dict:
    files = []
    for logical_name, (root_key, relative) in sorted(OVERLAY_FILES.items()):
        path = (Path(REMOTE_PATHS[root_key]) / relative).resolve(strict=True)
        files.append({
            "logical_name": logical_name,
            "path": str(path),
            "sha256": normalized_implementation_sha256(path),
            "normalization": "manifest_sha_line" if path.name == "run_streamvln_p8_train_diagnostic.py" else "none",
        })
    return {"manifest_sha_line_replacement": MANIFEST_LINE_NORMALIZATION, "files": files}


def _metadata_identity(python_path: Path, package_names: list[str]) -> dict:
    program = (
        "import importlib.metadata,json,platform;"
        f"names={package_names!r};"
        "print(json.dumps({'python_version':platform.python_version(),'packages':{name:importlib.metadata.version(name) for name in names}},sort_keys=True))"
    )
    completed = subprocess.run([str(python_path), "-c", program], capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode:
        raise RuntimeError(f"runtime metadata probe failed for {python_path}: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def git_head(path: Path) -> str:
    completed = subprocess.run(["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode:
        raise RuntimeError(f"cannot resolve git head for {path}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def verify_external_contract(manifest: dict) -> None:
    if git_head(Path(REMOTE_PATHS["rlinf_root"])) != RLINF_COMMIT:
        raise RuntimeError("P8 RLinf commit changed")
    if git_head(Path(REMOTE_PATHS["stream_source"])) != STREAM_COMMIT:
        raise RuntimeError("P8 StreamVLN commit changed")
    if git_head(Path(REMOTE_PATHS["habitat_sim_root"])) != HABITAT_SIM_COMMIT:
        raise RuntimeError("P8 Habitat-Sim commit changed")
    for name, inventory in manifest["released_inputs"].items():
        if name == "adapter_checkpoint":
            continue
        root = Path(inventory["root"]).resolve(strict=True)
        actual_names = sorted(path.name for path in root.iterdir() if path.is_file())
        expected_names = sorted(item["relative_path"] for item in inventory["files"])
        if actual_names != expected_names:
            raise RuntimeError(f"P8 {name} file set changed")
        for item in inventory["files"]:
            path = (root / item["relative_path"]).resolve(strict=True)
            if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"P8 {name} identity changed: {item['relative_path']}")
    for name in ("rlinf", "habitat"):
        expected = manifest["runtime_identity"][name]
        # Execute the declared virtual-environment path itself.  Resolving the
        # symlink first would launch the base interpreter and discard the
        # venv's pyvenv.cfg/package context.
        python_path = Path(expected["python_path"]).expanduser()
        if not python_path.is_absolute():
            python_path = python_path.absolute()
        if not python_path.exists():
            raise RuntimeError(f"P8 {name} Python path does not exist: {python_path}")
        if sha256_file(python_path) != expected["python_sha256"]:
            raise RuntimeError(f"P8 {name} Python hash changed")
        actual = _metadata_identity(python_path, list(expected["packages"]))
        wanted = {"python_version": expected["python_version"], "packages": expected["packages"]}
        if actual != wanted:
            raise RuntimeError(f"P8 {name} runtime versions changed: {actual}")
    egl = Path(REMOTE_PATHS["egl_vendor_json"]).resolve(strict=True)
    if sha256_file(egl) != manifest["runtime_identity"]["egl_vendor_json_sha256"]:
        raise RuntimeError("P8 EGL vendor identity changed")


def build_manifest(asset_root: Path, *, verify_external: bool = True) -> dict:
    asset_root = asset_root.resolve(strict=True)
    if verify_external and str(asset_root) != REMOTE_PATHS["asset_root"]:
        raise RuntimeError("formal P8 manifest must use the frozen remote asset root")
    train_path = (asset_root / "datasets" / "r2r" / "train" / "train.json.gz").resolve(strict=True)
    train_hash = sha256_file(train_path)
    if train_hash != TRAIN_SHA256:
        raise RuntimeError(f"P8 train dataset hash changed: {train_hash} != {TRAIN_SHA256}")
    rows = load_identities(train_path)
    tiny_train = select_distinct_scenes(rows, seed=SELECTION_SEED, role="tiny_train", count=8)
    tiny_scenes = frozenset(row["scene"] for row in tiny_train)
    train_development = select_distinct_scenes(rows, seed=SELECTION_SEED, role="train_development", count=8, excluded_scenes=tiny_scenes)
    equivalence = tiny_train[:4]
    manifest = {
        "schema_version": 2,
        "stage": "P8/G8 train-only causal diagnostic",
        "created": "2026-09-28",
        "seeds": {"selection": SELECTION_SEED, "base_wrapper_init": BASE_WRAPPER_INIT_SEED, "residual_init": RESIDUAL_INIT_SEED},
        "selection_timing": "before_any_p8_model_load",
        "selection_uses_outcome_fields": False,
        "selection_rule": "sha256('p8g8-diagnostic-v1|seed|role|scene|index|episode_id'); one identity per distinct scene",
        "source": {
            "commits": {"rlinf": RLINF_COMMIT, "streamvln": STREAM_COMMIT, "habitat_sim": HABITAT_SIM_COMMIT},
            "train_path": str(train_path), "train_sha256": train_hash, "allowed_splits": ["train"],
            "val_seen_read": False, "val_unseen_read": False,
        },
        "paths": dict(REMOTE_PATHS),
        "runtime_identity": json.loads(json.dumps(RUNTIME_IDENTITIES)),
        "released_inputs": {**model_input_inventory(), "adapter_checkpoint": None},
        "episodes": {
            "tiny_train": [public_identity(row) for row in tiny_train],
            "equivalence": [public_identity(row) for row in equivalence],
            "train_development": [public_identity(row) for row in train_development],
            "counts": {"tiny_train_episodes": 8, "tiny_train_scenes": 8, "equivalence_episodes": 4, "train_development_episodes": 8, "train_development_scenes": 8},
            "selected_scene_assets": selected_scene_assets(asset_root, tiny_train + train_development),
        },
        "runtime": {
            "model_load_count": 1, "visible_gpu_count": 1, "device": "cuda:0", "deterministic_eval": False,
            "context_tokens": 4096, "temperature": 1.0, "max_primitive_steps": 160, "habitat_startup_timeout_s": 180.0,
            "determinism": {
                "seed_python": True, "seed_numpy": True, "seed_torch_cpu": True, "seed_torch_cuda": True,
                "cublas_workspace_config": ":4096:8", "cudnn_deterministic": True, "cudnn_benchmark": False,
                "torch_use_deterministic_algorithms": False, "deterministic_sampling": True,
            },
        },
        "architecture": {"topology_model_dim": 128, "node_feature_dim": 8, "edge_feature_dim": 4, "max_waypoint_m": 1.25, "value_head_trainable": False},
        "executor": {
            "action_chunk_size": 4, "action_names": ["STOP", "↑", "←", "→"], "action_token_ids": [50669, 76286, 71858, 51018],
            "forward_step_m": 0.25, "turn_angle_deg": 15.0, "topology_node_radius_m": 0.5, "success_radius_m": 3.0,
            "expert_lookahead_m": 1.0, "blocked_forward_threshold": 2, "recover_target": "parent",
        },
        "training": {
            "expert_max_decisions_per_episode": 32, "warmup_updates": 500, "dagger_max_decisions_per_episode": 20,
            "final_updates": 1500, "batch_size": 32, "learning_rate": 0.0001, "weight_decay": 0.0,
            "metric_interval_updates": 25, "consecutive_gate_b_passes": 2, "duplicate_rows_by_state_hash": False,
            "loss": {"token": {"coefficient": 1.0}, "intent": {"coefficient": 1.0}, "waypoint": {"coefficient": 1.0, "type": "SmoothL1Loss", "beta": 1.0, "reduction": "mean"}},
            "optimizer": {"type": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False, "gradient_clip": None, "scheduler": None},
            "sampling": {"method": "torch.randint", "replacement": True},
            "class_weights": {
                "method": "inverse_frequency", "normalization": "total_rows_divided_by_present_class_count_times_class_count",
                "phase_sources": {"warmup": "deduplicated_expert_rows", "final": "deduplicated_expert_plus_dagger_rows"},
                "computed_once_per_phase": True, "row_duplication": False,
            },
            "stopping": {"rule": "first_two_consecutive_passing_checks", "check_interval_updates": 25, "only_stopping_point": True},
            "value_head_trainable": False, "trainable_precision": "fp32", "frozen_backbone_precision": "bf16", "ppo_updates": 0,
        },
        "gates": {
            "a": {"mismatch_count_max": 0, "optimizer_steps": 0, "parameter_hash_unchanged": True},
            "b": {"token_top1_accuracy_min": 0.90, "navigate_intent_recall_min": 0.90, "stop_intent_recall_min": 0.90, "stop_token_recall_min": 0.90, "waypoint_mean_l2_m_max": 0.15, "sample_replay_ratio_deviation_max": 0.001, "consecutive_passes": 2},
            "c": {"learned_success_count_min": 1, "successful_stop_count_min": 1, "max_success_deficit_vs_keep_original": 1, "primitive_category_count_min": 2, "decision_mode_count_min": 2, "turn_count_for_handedness_check": 20, "single_turn_direction_fraction_max": 0.95, "recover_required_when_blocked": True, "recover_alignment_miss_max": 0},
        },
        "implementation": overlay_inventory() if verify_external else {"manifest_sha_line_replacement": MANIFEST_LINE_NORMALIZATION, "files": []},
        "irreversibility": {"protocol_changes_allowed_until": "before_first_base_model_load", "scientific_failure_rerun_allowed": False, "unique_live_diagnostic": True},
        "claim_boundary": {"train_only": True, "val_seen_access": False, "val_unseen_access": False, "ppo": False, "original_streamvln_only": True, "ttt_or_secondary_post_training": False, "robot_execution": False, "remote_push": False},
    }
    if verify_external:
        verify_external_contract(manifest)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = build_manifest(args.asset_root.expanduser().resolve(strict=True))
    output = args.output.expanduser().resolve()
    write_json_atomic(output, manifest)
    print(json.dumps({
        "output": str(output), "sha256": sha256_file(output), "counts": manifest["episodes"]["counts"],
        "selected_scene_assets": len(manifest["episodes"]["selected_scene_assets"]),
        "allowed_splits": manifest["source"]["allowed_splits"], "val_seen_read": False, "val_unseen_read": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
