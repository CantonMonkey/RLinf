# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only checks for the isolated RLinf v0.3 GPU placement guard."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from ray._private.accelerators.nvidia_gpu import NvidiaGPUAcceleratorManager

from rlinf.scheduler.cluster.cluster import Cluster, PathEnvMergeMode
from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager
from rlinf.scheduler.worker.worker import Worker

KEY = NvidiaGPUManager.ALLOWED_PHYSICAL_GPUS_ENV


@pytest.fixture(autouse=True)
def host_with_eight_gpus(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("VISIBLE_DEVICES", raising=False)
    with patch.object(
        NvidiaGPUAcceleratorManager,
        "get_current_node_num_accelerators",
        return_value=8,
    ):
        yield


def test_logical_zero_maps_to_physical_seven(monkeypatch):
    monkeypatch.setenv(KEY, "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("VISIBLE_DEVICES", "0")

    assert NvidiaGPUManager.get_num_devices() == 1
    worker_env = NvidiaGPUManager.get_accelerator_env_var(["0"])
    assert worker_env["CUDA_VISIBLE_DEVICES"] == "7"
    assert worker_env["MUJOCO_EGL_DEVICE_ID"] == "7"
    assert NvidiaGPUManager.get_visible_devices() == [0]

    # The actual Worker method must index the one-card Cluster with logical 0.
    worker = SimpleNamespace(
        _accelerator_type=AcceleratorType.NV_GPU,
        _cluster_node_rank=0,
        _is_ray_actor=True,
    )
    with patch(
        "rlinf.scheduler.worker.worker.Cluster",
        return_value=SimpleNamespace(accelerator_ranks=[[0]]),
    ):
        Worker._setup_accelerator_info(worker)
    assert worker.global_accelerator_ids == [0]


def test_worker_runtime_env_cannot_restore_cuda_zero(monkeypatch):
    monkeypatch.setenv(KEY, "7")
    node_env = {KEY: "7", "CUDA_VISIBLE_DEVICES": "0"}
    worker_env = {
        KEY: "7",
        "VISIBLE_DEVICES": "0",
        **NvidiaGPUManager.get_accelerator_env_var(["0"]),
    }
    merged = Cluster.merge_worker_env_vars(node_env, worker_env, PathEnvMergeMode.APPEND)
    assert merged["CUDA_VISIBLE_DEVICES"] == "7"
    Cluster._validate_gpu_worker_env(merged, worker_env)

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES disagrees"):
        Cluster._validate_gpu_worker_env({**merged, "CUDA_VISIBLE_DEVICES": "0"}, worker_env)
    with pytest.raises(ValueError, match="changed"):
        Cluster._validate_gpu_worker_env({**merged, KEY: "0"}, worker_env)


def test_node_probe_must_enumerate_one_gpu_with_same_allowlist(monkeypatch):
    monkeypatch.setenv(KEY, "7")
    node = SimpleNamespace(node_rank=0, default_env_vars={KEY: "7"}, num_accelerators=1)
    Cluster._validate_gpu_allowlist_on_nodes([node])

    with pytest.raises(ValueError, match="disagrees"):
        Cluster._validate_gpu_allowlist_on_nodes([
            SimpleNamespace(node_rank=0, default_env_vars={}, num_accelerators=8)
        ])


@pytest.mark.parametrize("raw", ["", "7,7", "-1", " 7", "7 ", "x", "8"])
def test_invalid_or_unavailable_allowlist_fails(monkeypatch, raw):
    monkeypatch.setenv(KEY, raw)
    with pytest.raises(ValueError):
        NvidiaGPUManager.get_num_devices()


def test_logical_rank_and_worker_cuda_mismatch_fail(monkeypatch):
    monkeypatch.setenv(KEY, "7")
    with pytest.raises(ValueError, match="outside"):
        NvidiaGPUManager.get_accelerator_env_var(["1"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="outside"):
        NvidiaGPUManager.get_visible_devices()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="disagrees"):
        NvidiaGPUManager.get_visible_devices()


def test_unset_allowlist_keeps_upstream_behavior(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,7")
    assert NvidiaGPUManager.get_num_devices() == 8
    assert NvidiaGPUManager.get_accelerator_env_var(["0"])["CUDA_VISIBLE_DEVICES"] == "0"
    assert NvidiaGPUManager.get_visible_devices() == [0, 7]
