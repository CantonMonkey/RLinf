#!/usr/bin/env python3
"""Bounded native LightNav-0 competence probe over its public wire protocols.

The probe deliberately contains no LightNav model code and no RLinf policy code.  It
connects the released ``lightnav-serve`` WebSocket endpoint to the released remote
Habitat ZMQ environment, sends the environment RGB/instruction to LightNav, and
executes the first velocity-relevant SE(2) waypoint with the official velocity
conversion.  This gives a no-training reference before any topology/residual adapter
is introduced.

Optional runtime packages (NumPy, Pillow, pyzmq and websockets) are imported only by
the functions that need them.  Importing this module is therefore safe in the small
RLinf contract-test environment.

Decode or protocol failures are invalid evidence.  They raise and produce an invalid
receipt; they are never converted into a zero-velocity/STOP fallback.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import pickle
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit


RECEIPT_SCHEMA = "lightnav_l0_native_probe_receipt"
RECEIPT_VERSION = 1
PICKLE_PROTOCOL = 4
DEFAULT_MAX_WS_BYTES = 64 * 1024 * 1024
MOTION_ATOL = 1e-6


class ProbeError(RuntimeError):
    """Base class for a probe result that cannot be treated as valid evidence."""


class HabitatProtocolError(ProbeError):
    """The remote Habitat endpoint violated its request/reply contract."""


class LightNavProtocolError(ProbeError):
    """The LightNav endpoint returned a malformed or inconsistent response."""


class LightNavDecodeError(LightNavProtocolError):
    """LightNav explicitly rejected or failed to decode a model prediction."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, float) and (not math.isfinite(value) or value != result):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _json_safe(value: object) -> object:
    """Convert common scalar/container values without importing NumPy eagerly."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path_value: str | Path, receipt: Mapping[str, object]) -> Path:
    """Durably replace ``path`` with one finite, canonical JSON receipt."""

    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_safe(receipt),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def redact_endpoint(endpoint: str) -> str:
    """Remove user info, query and fragment before an endpoint enters a receipt."""

    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def extract_instruction(observation: Mapping[str, object]) -> str:
    raw = observation.get("instruction")
    if isinstance(raw, Mapping):
        raw = raw.get("text")
    if not isinstance(raw, str) or not raw.strip():
        raise HabitatProtocolError("Habitat observation has no non-empty instruction")
    return raw


def encode_rgb_jpeg_base64(rgb: object, *, quality: int = 90) -> str:
    """Encode one HWC uint8 RGB observation using the official JPEG wire field."""

    if not 1 <= int(quality) <= 100:
        raise ValueError("jpeg quality must be in [1, 100]")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on runtime extras
        raise ImportError("JPEG encoding requires NumPy and Pillow") from exc
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise HabitatProtocolError(
            f"Habitat RGB must be HWC uint8 with three channels, got {array.shape}/{array.dtype}"
        )
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=int(quality))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def login_request(client_id: str) -> dict[str, object]:
    if not isinstance(client_id, str) or not client_id:
        raise ValueError("client_id must be a non-empty string")
    return {"action": "login", "data": {"clientId": client_id}}


def reset_request() -> dict[str, object]:
    return {"action": "reset", "data": {}}


def next_request(*, seq: int, image_b64: str, instruction: str) -> dict[str, object]:
    seq = _integer(seq, "seq", minimum=0)
    if not isinstance(image_b64, str) or not image_b64:
        raise ValueError("image_b64 must be non-empty text")
    if not isinstance(instruction, str) or not instruction:
        raise ValueError("instruction must be non-empty text for a prediction")
    return {
        "action": "next",
        "data": {"seq": seq, "image": image_b64, "instruction": instruction},
    }


def parse_actions_payload(value: object) -> tuple[tuple[float, float, float], ...]:
    """Parse the official dict plus its documented legacy list spellings."""

    trajectories: object
    if isinstance(value, Mapping):
        trajectories = value.get("actions")
    else:
        trajectories = value
    if not isinstance(trajectories, list) or not trajectories:
        raise LightNavProtocolError("LightNav response has no non-empty waypoint trajectory")
    if (
        len(trajectories) == 1
        and isinstance(trajectories[0], list)
        and trajectories[0]
        and isinstance(trajectories[0][0], (list, tuple))
    ):
        trajectories = trajectories[0]
    if trajectories and not isinstance(trajectories[0], (list, tuple)):
        if len(trajectories) % 3 != 0:
            raise LightNavProtocolError("flat LightNav actions length is not divisible by three")
        trajectories = [trajectories[index : index + 3] for index in range(0, len(trajectories), 3)]

    rows: list[tuple[float, float, float]] = []
    for index, raw_row in enumerate(trajectories):
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != 3:
            raise LightNavProtocolError(f"LightNav waypoint {index} must contain three values")
        try:
            row = tuple(_finite(item, f"waypoint[{index}]") for item in raw_row)
        except ValueError as exc:
            raise LightNavProtocolError(str(exc)) from exc
        rows.append((row[0], row[1], row[2]))
    return tuple(rows)


def select_first_motion_waypoint(
    waypoints: Sequence[Sequence[float]], *, atol: float = MOTION_ATOL
) -> tuple[int, tuple[float, float, float]]:
    """Select the first waypoint relevant to the official unicycle mapping.

    LightNav waypoints remain full SE(2) triples in the result.  Lateral displacement
    is retained for evidence, while the official Habitat controller uses forward and
    yaw.  A lateral-only chunk is therefore invalid for this executor rather than being
    silently treated as STOP.
    """

    for index, row in enumerate(waypoints):
        if len(row) != 3:
            raise LightNavProtocolError("LightNav waypoint must be an SE(2) triple")
        forward, lateral, yaw = (_finite(value, "waypoint") for value in row)
        if abs(forward) > atol or abs(yaw) > atol:
            return index, (forward, lateral, yaw)
    raise LightNavProtocolError("non-STOP LightNav trajectory has no velocity-relevant waypoint")


@dataclass(frozen=True)
class LightNavPrediction:
    seq: int
    stop: bool
    waypoints: tuple[tuple[float, float, float], ...]
    selected_index: int
    selected_waypoint: tuple[float, float, float]
    latency_ms: float
    timings_ms: Mapping[str, float]
    raw_text_sha256: str


def parse_lightnav_response(payload: object, *, expected_seq: int) -> LightNavPrediction:
    """Validate one official ``next`` reply and preserve its full SE(2) action."""

    try:
        outer = _mapping(payload, "LightNav response")
        if outer.get("action") != "next":
            raise LightNavProtocolError("LightNav response action is not 'next'")
        data = _mapping(outer.get("data"), "LightNav response data")
        rc = _integer(data.get("rc"), "LightNav response rc")
        seq = _integer(data.get("seq"), "LightNav response seq", minimum=0)
    except ValueError as exc:
        raise LightNavProtocolError(str(exc)) from exc
    if seq != expected_seq:
        raise LightNavProtocolError(
            f"LightNav response seq {seq} does not match request {expected_seq}"
        )
    if rc != 0:
        message = data.get("msg")
        raise LightNavDecodeError(f"LightNav prediction failed with rc={rc}: {message}")
    if "actions" not in data:
        raise LightNavProtocolError("successful LightNav prediction has no actions")
    waypoints = parse_actions_payload(data["actions"])
    try:
        stop = _bool(data.get("stop"), "LightNav stop")
        latency_ms = _nonnegative(data.get("latency_ms"), "LightNav latency_ms")
    except ValueError as exc:
        raise LightNavProtocolError(str(exc)) from exc

    all_zero = all(all(abs(value) <= MOTION_ATOL for value in row) for row in waypoints)
    if stop != all_zero:
        raise LightNavProtocolError("LightNav stop flag differs from the decoded all-zero trajectory")
    if stop:
        selected_index = 0
        selected = (0.0, 0.0, 0.0)
    else:
        selected_index, selected = select_first_motion_waypoint(waypoints)

    timings_raw = data.get("timings_ms", {})
    timings: dict[str, float] = {}
    if timings_raw is not None:
        if not isinstance(timings_raw, Mapping):
            raise LightNavProtocolError("LightNav timings_ms must be an object")
        for name, value in timings_raw.items():
            try:
                timings[str(name)] = _nonnegative(value, f"timings_ms.{name}")
            except ValueError as exc:
                raise LightNavProtocolError(str(exc)) from exc
    raw_text = data.get("raw_text", "")
    if not isinstance(raw_text, str):
        raise LightNavProtocolError("LightNav raw_text must be text")
    return LightNavPrediction(
        seq=seq,
        stop=stop,
        waypoints=waypoints,
        selected_index=selected_index,
        selected_waypoint=selected,
        latency_ms=latency_ms,
        timings_ms=timings,
        raw_text_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class VelocitySpec:
    time_step_s: float
    linear_range_mps: tuple[float, float]
    angular_range_dps: tuple[float, float]

    @classmethod
    def from_info(cls, info: Mapping[str, object]) -> "VelocitySpec":
        try:
            dt = _finite(info.get("habitat_time_step"), "habitat_time_step")
            linear = tuple(info.get("lin_vel_range", ()))
            angular = tuple(info.get("ang_vel_range", ()))
        except TypeError as exc:
            raise HabitatProtocolError("Habitat velocity ranges must be sequences") from exc
        if dt <= 0.0 or len(linear) != 2 or len(angular) != 2:
            raise HabitatProtocolError("Habitat did not expose a valid velocity-control contract")
        try:
            lin = (_finite(linear[0], "lin_vel_range[0]"), _finite(linear[1], "lin_vel_range[1]"))
            ang = (_finite(angular[0], "ang_vel_range[0]"), _finite(angular[1], "ang_vel_range[1]"))
        except ValueError as exc:
            raise HabitatProtocolError(str(exc)) from exc
        if lin[1] <= lin[0] or ang[1] <= ang[0]:
            raise HabitatProtocolError("Habitat velocity ranges must be strictly increasing")
        return cls(dt, lin, ang)


def _normalize_velocity(value: float, limits: tuple[float, float]) -> float:
    lower, upper = limits
    raw = 2.0 * (value - lower) / (upper - lower) - 1.0
    return min(1.0, max(-1.0, raw))


def waypoint_to_velocity_action(
    waypoint: Sequence[float], spec: VelocitySpec
) -> dict[str, object]:
    """Apply LightNav's official forward/yaw to Habitat normalized velocity map."""

    if len(waypoint) != 3:
        raise ValueError("waypoint must be an SE(2) triple")
    forward, _lateral, yaw = (_finite(value, "waypoint") for value in waypoint)
    linear_mps = forward / spec.time_step_s
    angular_dps = math.degrees(yaw) / spec.time_step_s
    return {
        "action": "velocity_control",
        "action_args": {
            "linear_velocity": _normalize_velocity(linear_mps, spec.linear_range_mps),
            "angular_velocity": _normalize_velocity(angular_dps, spec.angular_range_dps),
        },
    }


def zero_waypoint_action(spec: VelocitySpec) -> dict[str, object]:
    """Return physical zero velocity (usually normalized ``[-1, 0]``), not raw zeros."""

    return waypoint_to_velocity_action((0.0, 0.0, 0.0), spec)


def _physical_velocity(action: Mapping[str, object], spec: VelocitySpec) -> tuple[float, float]:
    args = _mapping(action.get("action_args"), "velocity action args")
    linear_raw = _finite(args.get("linear_velocity"), "linear_velocity")
    angular_raw = _finite(args.get("angular_velocity"), "angular_velocity")
    lin = spec.linear_range_mps[0] + (linear_raw + 1.0) * 0.5 * (
        spec.linear_range_mps[1] - spec.linear_range_mps[0]
    )
    ang = spec.angular_range_dps[0] + (angular_raw + 1.0) * 0.5 * (
        spec.angular_range_dps[1] - spec.angular_range_dps[0]
    )
    return lin, ang


def classify_velocity_action(action: Mapping[str, object], spec: VelocitySpec) -> str:
    linear, angular = _physical_velocity(action, spec)
    moving = abs(linear) > MOTION_ATOL
    turning = abs(angular) > MOTION_ATOL
    if not moving and not turning:
        return "STOP"
    if moving and turning:
        prefix = "FORWARD" if linear > 0.0 else "REVERSE"
        return f"{prefix}_{'LEFT' if angular > 0.0 else 'RIGHT'}"
    if moving:
        return "FORWARD" if linear > 0.0 else "REVERSE"
    return "LEFT" if angular > 0.0 else "RIGHT"


class ZmqHabitatClient:
    """Small dependency-lazy clone of LightNav's public RemoteEnvClient."""

    def __init__(self, address: str, *, timeout_ms: int = 600_000, recv_retries: int = 2):
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - runtime only
            raise ImportError("ZMQ Habitat transport requires pyzmq") from exc
        self._zmq = zmq
        self.address = address
        self.timeout_ms = int(timeout_ms)
        self.recv_retries = int(recv_retries)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.address)
        self.connected = True

    def _command(self, command: str, data: object = None) -> Mapping[str, object]:
        if not self.connected:
            raise HabitatProtocolError("Habitat client is closed")
        try:
            self.socket.send(pickle.dumps({"command": command, "data": data}, protocol=PICKLE_PROTOCOL))
        except self._zmq.Again as exc:
            raise HabitatProtocolError(f"Habitat send timeout for {command}") from exc
        attempts = max(1, self.recv_retries + 1)
        raw = None
        for attempt in range(attempts):
            try:
                raw = self.socket.recv()
                break
            except self._zmq.Again as exc:
                if attempt + 1 == attempts:
                    raise HabitatProtocolError(f"Habitat receive timeout for {command}") from exc
        try:
            response = pickle.loads(raw)
        except Exception as exc:  # noqa: BLE001 - untrusted wire payload
            raise HabitatProtocolError("Habitat returned an invalid pickle payload") from exc
        if not isinstance(response, Mapping):
            raise HabitatProtocolError("Habitat response must be an object")
        if response.get("status") == "error":
            raise HabitatProtocolError(f"Habitat server error: {response.get('message')}")
        if response.get("status") != "success":
            raise HabitatProtocolError("Habitat response lacks success status")
        return response

    def reset(self) -> tuple[Mapping[str, object], Mapping[str, object]]:
        response = self._command("reset", {"seed": None, "options": None})
        return _mapping(response.get("obs"), "Habitat reset observation"), _mapping(
            response.get("info", {}), "Habitat reset info"
        )

    def step(self, action: Mapping[str, object]):
        response = self._command("step", action)
        try:
            reward = _finite(response.get("reward"), "Habitat reward")
            terminated = _bool(response.get("terminated"), "Habitat terminated")
            truncated = _bool(response.get("truncated"), "Habitat truncated")
        except ValueError as exc:
            raise HabitatProtocolError(str(exc)) from exc
        return (
            _mapping(response.get("obs"), "Habitat step observation"),
            reward,
            terminated,
            truncated,
            _mapping(response.get("info", {}), "Habitat step info"),
        )

    def close(self) -> None:
        # This probe may share a long-lived Habitat service with another client.
        # Local teardown must never send the server's process-level ``close`` RPC;
        # service lifecycle is owned by the launcher.
        self.connected = False
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None


class LightNavWebSocketClient:
    """Synchronous client for the official LightNav login/reset/next protocol."""

    def __init__(
        self,
        uri: str,
        *,
        timeout_s: float = 60.0,
        connect_factory: Callable[..., object] | None = None,
    ) -> None:
        self.uri = uri
        self.timeout_s = _nonnegative(timeout_s, "WebSocket timeout")
        self._connect_factory = connect_factory
        self.socket = None

    def open(self) -> None:
        if self.socket is not None:
            return
        factory = self._connect_factory
        if factory is None:
            try:
                from websockets.sync.client import connect
            except ImportError as exc:  # pragma: no cover - runtime only
                raise ImportError("LightNav WebSocket transport requires websockets>=12") from exc
            factory = connect
        self.socket = factory(
            self.uri,
            open_timeout=self.timeout_s,
            close_timeout=self.timeout_s,
            max_size=DEFAULT_MAX_WS_BYTES,
        )

    def _rpc(self, request: Mapping[str, object]) -> Mapping[str, object]:
        if self.socket is None:
            raise LightNavProtocolError("LightNav WebSocket is not open")
        self.socket.send(json.dumps(request, ensure_ascii=False, separators=(",", ":")))
        try:
            raw = self.socket.recv(timeout=self.timeout_s)
        except TypeError:  # websockets 12 accepts timeout; a compatible fake may not.
            raw = self.socket.recv()
        try:
            response = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LightNavProtocolError("LightNav returned invalid JSON") from exc
        if not isinstance(response, Mapping):
            raise LightNavProtocolError("LightNav response must be an object")
        return response

    @staticmethod
    def _ack(response: Mapping[str, object], action: str) -> None:
        if response.get("action") != action:
            raise LightNavProtocolError(f"LightNav {action} acknowledgement action mismatch")
        try:
            data = _mapping(response.get("data"), f"LightNav {action} data")
            rc = _integer(data.get("rc"), f"LightNav {action} rc")
        except ValueError as exc:
            raise LightNavProtocolError(str(exc)) from exc
        if rc != 0:
            raise LightNavProtocolError(f"LightNav {action} failed with rc={rc}: {data.get('msg')}")

    def login(self, client_id: str) -> None:
        response = self._rpc(login_request(client_id))
        self._ack(response, "login")

    def reset(self) -> None:
        response = self._rpc(reset_request())
        self._ack(response, "reset")

    def predict(self, *, seq: int, image_b64: str, instruction: str) -> LightNavPrediction:
        response = self._rpc(next_request(seq=seq, image_b64=image_b64, instruction=instruction))
        return parse_lightnav_response(response, expected_seq=seq)

    def close(self) -> None:
        if self.socket is None:
            return
        try:
            self.socket.close()
        finally:
            self.socket = None


def _metric(info: Mapping[str, object], names: Sequence[str], label: str) -> float:
    for name in names:
        if name in info:
            try:
                return _finite(info[name], name)
            except ValueError as exc:
                raise HabitatProtocolError(str(exc)) from exc
    raise HabitatProtocolError(f"Habitat info has no required {label} metric")


def _optional_metric(info: Mapping[str, object], names: Sequence[str]) -> float | None:
    for name in names:
        if name in info:
            try:
                return _finite(info[name], name)
            except ValueError as exc:
                raise HabitatProtocolError(str(exc)) from exc
    return None


def _collision_observation(info: Mapping[str, object]) -> tuple[float | None, bool | None, str]:
    """Return cumulative count, per-step flag, and provenance without inventing collisions."""

    try:
        if "collision_count" in info:
            return _nonnegative(info["collision_count"], "collision_count"), None, "collision_count"
        raw = info.get("collisions")
        if isinstance(raw, Mapping):
            if "count" in raw:
                return _nonnegative(raw["count"], "collisions.count"), None, "collisions.count"
            if "is_collision" in raw:
                return None, _bool(raw["is_collision"], "collisions.is_collision"), "collisions.is_collision"
        if "collision" in info:
            value = info["collision"]
            if type(value) is bool:
                return None, value, "collision"
            return _nonnegative(value, "collision"), None, "collision"
    except ValueError as exc:
        raise HabitatProtocolError(str(exc)) from exc
    return None, None, "unavailable"


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(values: Sequence[float]) -> dict[str, object]:
    finite = [_nonnegative(value, "latency") for value in values]
    return {
        "count": len(finite),
        "mean_ms": statistics.fmean(finite) if finite else None,
        "p50_ms": _percentile(finite, 0.50),
        "p95_ms": _percentile(finite, 0.95),
        "max_ms": max(finite) if finite else None,
    }


@dataclass(frozen=True)
class ProbeConfig:
    episodes: int = 8
    max_steps: int = 500
    jpeg_quality: int = 90
    client_id: str = "rlinf-lightnav-l0"
    split_label: str = "train"
    require_collision_metric: bool = False

    def __post_init__(self) -> None:
        _integer(self.episodes, "episodes", minimum=1)
        _integer(self.max_steps, "max_steps", minimum=1)
        if not 1 <= int(self.jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1,100]")
        if not self.client_id:
            raise ValueError("client_id must be non-empty")
        if not self.split_label:
            raise ValueError("split_label must be non-empty")


@dataclass
class EpisodeRun:
    receipt: dict[str, object]
    server_latencies_ms: list[float]
    roundtrip_latencies_ms: list[float]


def run_episode(
    env: object,
    model: object,
    config: ProbeConfig,
    *,
    image_encoder: Callable[..., str] = encode_rgb_jpeg_base64,
) -> EpisodeRun:
    observation, reset_info = env.reset()
    if not isinstance(observation, Mapping) or not isinstance(reset_info, Mapping):
        raise HabitatProtocolError("Habitat reset must return observation and info objects")
    model.reset()
    instruction = extract_instruction(observation)
    velocity = VelocitySpec.from_info(reset_info)
    initial_distance = _metric(
        reset_info, ("distance_to_goal", "goal_distance", "navigation_error"), "navigation error"
    )
    initial_path = _optional_metric(reset_info, ("path_length",)) or 0.0
    min_distance = initial_distance
    final_info: Mapping[str, object] = reset_info
    terminated = truncated = False
    forced_stop = False
    model_stop = False
    seq = 0
    action_counts: Counter[str] = Counter()
    selected_indices: Counter[str] = Counter()
    server_latencies: list[float] = []
    roundtrip_latencies: list[float] = []
    collision_cumulative: float | None = None
    collision_events = 0
    collision_sources: Counter[str] = Counter()
    step_count = 0
    final_stop_source = "none"
    causal_steps: list[dict[str, object]] = []

    reset_cumulative, reset_event, reset_source = _collision_observation(reset_info)
    if reset_cumulative is not None:
        collision_cumulative = reset_cumulative
        collision_sources[reset_source] += 1
    if reset_event is not None:
        collision_sources[reset_source] += 1
        if reset_event:
            collision_events += 1

    for step_index in range(config.max_steps):
        if step_index == config.max_steps - 1:
            selected_waypoint = (0.0, 0.0, 0.0)
            trajectory = (selected_waypoint,)
            action = zero_waypoint_action(velocity)
            forced_stop = True
            final_stop_source = "max_step_forced_stop"
            action_counts["STOP"] += 1
            causal = {
                "step": step_index,
                "seq": None,
                "raw_text_sha256": None,
                "trajectory_sha256": canonical_sha256(trajectory),
                "selected_index": None,
                "selected_se2": list(selected_waypoint),
                "server_latency_ms": None,
                "stop_flag": True,
                "stop_source": final_stop_source,
            }
        else:
            if "rgb" not in observation:
                raise HabitatProtocolError("Habitat observation has no RGB frame")
            image_b64 = image_encoder(
                observation["rgb"], quality=config.jpeg_quality
            )
            started = time.monotonic()
            prediction = model.predict(
                seq=seq,
                image_b64=image_b64,
                instruction=instruction,
            )
            roundtrip_ms = (time.monotonic() - started) * 1000.0
            if not isinstance(prediction, LightNavPrediction):
                raise LightNavProtocolError("model client did not return LightNavPrediction")
            seq += 1
            server_latencies.append(prediction.latency_ms)
            roundtrip_latencies.append(roundtrip_ms)
            selected_indices[str(prediction.selected_index)] += 1
            action = waypoint_to_velocity_action(prediction.selected_waypoint, velocity)
            category = classify_velocity_action(action, velocity)
            if category == "STOP" and not prediction.stop:
                raise LightNavProtocolError(
                    "non-STOP LightNav waypoint maps to physical zero velocity"
                )
            action_counts[category] += 1
            causal = {
                "step": step_index,
                "seq": prediction.seq,
                "raw_text_sha256": prediction.raw_text_sha256,
                "trajectory_sha256": canonical_sha256(prediction.waypoints),
                "selected_index": prediction.selected_index,
                "selected_se2": list(prediction.selected_waypoint),
                "server_latency_ms": prediction.latency_ms,
                "stop_flag": prediction.stop,
                "stop_source": "model_stop" if prediction.stop else "none",
            }
            if prediction.stop:
                if category != "STOP":
                    raise LightNavProtocolError("LightNav STOP did not map to physical zero velocity")
                model_stop = True
                final_stop_source = "model_stop"

        action_args = _mapping(action.get("action_args"), "velocity action args")
        causal["normalized_command"] = {
            "linear": _finite(action_args.get("linear_velocity"), "linear_velocity"),
            "angular": _finite(action_args.get("angular_velocity"), "angular_velocity"),
        }
        causal["action_category"] = classify_velocity_action(action, velocity)

        observation, _reward, terminated, truncated, final_info = env.step(action)
        if not isinstance(observation, Mapping) or not isinstance(final_info, Mapping):
            raise HabitatProtocolError("Habitat step must return observation and info objects")
        step_count += 1
        distance = _metric(
            final_info, ("distance_to_goal", "goal_distance", "navigation_error"), "navigation error"
        )
        min_distance = min(min_distance, distance)
        causal["navigation_error_m"] = distance
        causal["path_length_m"] = _optional_metric(final_info, ("path_length",))
        causal["terminated"] = bool(terminated)
        causal["truncated"] = bool(truncated)
        cumulative, event, source = _collision_observation(final_info)
        if cumulative is not None:
            collision_cumulative = cumulative
            collision_sources[source] += 1
        if event is not None:
            collision_sources[source] += 1
            if event:
                collision_events += 1
        causal["collision_observation"] = {
            "cumulative": cumulative,
            "event": event,
            "source": source,
        }
        causal_steps.append(causal)

        if model_stop or forced_stop:
            if not (terminated or truncated):
                raise HabitatProtocolError("physical STOP did not terminate the Habitat episode")
        if terminated or truncated:
            break

    if not (terminated or truncated):
        raise HabitatProtocolError("episode exceeded the client step budget without termination")

    success_value = final_info.get("success")
    if isinstance(success_value, bool):
        success = success_value
    else:
        try:
            success = bool(round(_finite(success_value, "success")))
        except ValueError as exc:
            raise HabitatProtocolError(str(exc)) from exc
    spl = _nonnegative(_metric(final_info, ("spl",), "SPL"), "spl")
    final_distance = _metric(
        final_info, ("distance_to_goal", "goal_distance", "navigation_error"), "navigation error"
    )
    path_length = _nonnegative(_metric(final_info, ("path_length",), "path length"), "path_length")

    collision_count: float | None
    collision_source: str
    if collision_cumulative is not None:
        collision_count = collision_cumulative
        collision_source = "cumulative_metric"
    elif collision_sources:
        collision_count = float(collision_events)
        collision_source = "per_step_metric"
    else:
        collision_count = None
        collision_source = "unavailable"
        if config.require_collision_metric:
            raise HabitatProtocolError("Habitat did not expose a collision metric")

    stop_called = model_stop or forced_stop
    timeout = bool(forced_stop or (truncated and not model_stop))
    scene = str(final_info.get("scene_id", reset_info.get("scene_id", "")))
    episode_id = str(final_info.get("episode_id", reset_info.get("episode_id", "")))
    if not scene or not episode_id:
        raise HabitatProtocolError("Habitat info lacks scene_id or episode_id")
    receipt = {
        "scene_id": scene,
        "episode_id": episode_id,
        "raw_episode_id": str(final_info.get("raw_episode_id", reset_info.get("raw_episode_id", ""))),
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "instruction_length": len(instruction),
        "success": success,
        "spl": spl,
        "initial_navigation_error_m": initial_distance,
        "final_navigation_error_m": final_distance,
        "minimum_navigation_error_m": min_distance,
        "geodesic_progress_m": initial_distance - final_distance,
        "path_length_m": path_length,
        "path_length_delta_m": path_length - initial_path,
        "collision_count": collision_count,
        "collision_source": collision_source,
        "steps": step_count,
        "stop_called": stop_called,
        "model_stop": model_stop,
        "forced_stop": forced_stop,
        "successful_stop": bool(stop_called and success),
        "failed_stop": bool(stop_called and not success),
        "timeout": timeout,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": str(final_info.get("termination_reason", "unknown")),
        "stop_source": final_stop_source,
        "action_distribution": dict(sorted(action_counts.items())),
        "selected_waypoint_index_distribution": dict(sorted(selected_indices.items())),
        "causal_steps": causal_steps,
        "server_latency": latency_summary(server_latencies),
        "roundtrip_latency": latency_summary(roundtrip_latencies),
    }
    return EpisodeRun(receipt, server_latencies, roundtrip_latencies)


def aggregate_episodes(
    episodes: Sequence[Mapping[str, object]],
    *,
    server_latencies_ms: Sequence[float],
    roundtrip_latencies_ms: Sequence[float],
) -> dict[str, object]:
    if not episodes:
        raise ValueError("cannot aggregate an empty episode set")
    count = len(episodes)
    action_counts: Counter[str] = Counter()
    collisions: list[float] = []
    for row in episodes:
        action_counts.update({str(key): int(value) for key, value in row["action_distribution"].items()})
        if row.get("collision_count") is not None:
            collisions.append(float(row["collision_count"]))
    successes = sum(bool(row["success"]) for row in episodes)
    timeouts = sum(bool(row["timeout"]) for row in episodes)
    model_stops = sum(bool(row["model_stop"]) for row in episodes)
    forced_stops = sum(bool(row["forced_stop"]) for row in episodes)
    successful_stops = sum(bool(row["successful_stop"]) for row in episodes)
    failed_stops = sum(bool(row["failed_stop"]) for row in episodes)
    final_errors = [float(row["final_navigation_error_m"]) for row in episodes]
    paths = [float(row["path_length_m"]) for row in episodes]
    progress = [float(row["geodesic_progress_m"]) for row in episodes]
    return {
        "episode_count": count,
        "success_count": successes,
        "sr": successes / count,
        "mean_spl": statistics.fmean(float(row["spl"]) for row in episodes),
        "mean_navigation_error_m": statistics.fmean(final_errors),
        "median_navigation_error_m": statistics.median(final_errors),
        "mean_geodesic_progress_m": statistics.fmean(progress),
        "mean_path_length_m": statistics.fmean(paths),
        "total_path_length_m": sum(paths),
        "collision": {
            "available_episode_count": len(collisions),
            "unavailable_episode_count": count - len(collisions),
            "total_count": sum(collisions) if collisions else None,
            "mean_count": statistics.fmean(collisions) if collisions else None,
        },
        "model_stop_count": model_stops,
        "forced_stop_count": forced_stops,
        "successful_stop_count": successful_stops,
        "failed_stop_count": failed_stops,
        "timeout_count": timeouts,
        "timeout_rate": timeouts / count,
        "action_distribution": dict(sorted(action_counts.items())),
        "server_latency": latency_summary(server_latencies_ms),
        "roundtrip_latency": latency_summary(roundtrip_latencies_ms),
    }


def run_probe(env: object, model: object, config: ProbeConfig) -> dict[str, object]:
    """Run a fixed-count native probe using injected or real transport clients."""

    model.login(config.client_id)
    rows: list[dict[str, object]] = []
    server_latencies: list[float] = []
    roundtrip_latencies: list[float] = []
    identities: set[tuple[str, str]] = set()
    for _index in range(config.episodes):
        episode = run_episode(env, model, config)
        identity = (str(episode.receipt["scene_id"]), str(episode.receipt["episode_id"]))
        if identity in identities:
            raise HabitatProtocolError("Habitat cycled to a repeated episode before the requested count")
        identities.add(identity)
        rows.append(episode.receipt)
        server_latencies.extend(episode.server_latencies_ms)
        roundtrip_latencies.extend(episode.roundtrip_latencies_ms)
    return {
        "schema_name": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_VERSION,
        "status": "completed",
        "probe_kind": "released_lightnav_native_no_training_reference",
        "split_label": config.split_label,
        "split_label_provenance": "client_declared; verify against Habitat server launch receipt",
        "training_performed": False,
        "episodes_requested": config.episodes,
        "max_steps": config.max_steps,
        "jpeg_quality": config.jpeg_quality,
        "collision_metric_required": config.require_collision_metric,
        "episodes": rows,
        "aggregate": aggregate_episodes(
            rows,
            server_latencies_ms=server_latencies,
            roundtrip_latencies_ms=roundtrip_latencies,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--habitat-server", required=True, help="ZMQ endpoint, e.g. tcp://127.0.0.1:5555")
    parser.add_argument("--lightnav-server", required=True, help="LightNav WebSocket endpoint")
    parser.add_argument("--output", type=Path, required=True, help="atomic JSON receipt path")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--client-id", default="rlinf-lightnav-l0")
    parser.add_argument("--split-label", default="train")
    parser.add_argument("--zmq-timeout-ms", type=int, default=600_000)
    parser.add_argument("--zmq-recv-retries", type=int, default=2)
    parser.add_argument("--ws-timeout-s", type=float, default=600.0)
    parser.add_argument("--require-collision-metric", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time()
    config = ProbeConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        jpeg_quality=args.jpeg_quality,
        client_id=args.client_id,
        split_label=args.split_label,
        require_collision_metric=bool(args.require_collision_metric),
    )
    env = None
    model = None
    try:
        env = ZmqHabitatClient(
            args.habitat_server,
            timeout_ms=args.zmq_timeout_ms,
            recv_retries=args.zmq_recv_retries,
        )
        model = LightNavWebSocketClient(args.lightnav_server, timeout_s=args.ws_timeout_s)
        model.open()
        receipt = run_probe(env, model, config)
        receipt["habitat_server"] = redact_endpoint(args.habitat_server)
        receipt["lightnav_server"] = redact_endpoint(args.lightnav_server)
        receipt["started_unix_s"] = started
        receipt["completed_unix_s"] = time.time()
        atomic_write_json(args.output, receipt)
        print(json.dumps(receipt["aggregate"], ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - the receipt is the failure boundary
        invalid = {
            "schema_name": RECEIPT_SCHEMA,
            "schema_version": RECEIPT_VERSION,
            "status": "invalid",
            "probe_kind": "released_lightnav_native_no_training_reference",
            "training_performed": False,
            "split_label": config.split_label,
            "habitat_server": redact_endpoint(args.habitat_server),
            "lightnav_server": redact_endpoint(args.lightnav_server),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        atomic_write_json(args.output, invalid)
        print(f"LightNav L0 probe invalid: {type(exc).__name__}: {exc}")
        return 2
    finally:
        if model is not None:
            model.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
