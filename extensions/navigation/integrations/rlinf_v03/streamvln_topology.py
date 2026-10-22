"""Online topological state and local waypoint executor for StreamVLN G5."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


STOP_ACTION = 0
FORWARD_ACTION = 1
LEFT_ACTION = 2
RIGHT_ACTION = 3
NAVIGATE_MODE = 0
STOP_MODE = 1
RECOVER_MODE = 2
MODE_NAMES = ("navigate", "stop", "recover")
FEATURE_NAMES = (
    "dead_reckoned_x",
    "dead_reckoned_z",
    "heading_sin",
    "heading_cos",
    "node_count",
    "edge_count",
    "current_node_visits",
    "consecutive_blocked_forwards",
    "blocked_forward_ratio",
    "backtrack_distance",
    "has_parent",
    "decision_fraction",
    "last_forward_succeeded",
    "turn_fraction",
)
CAUSAL_NODE_FEATURE_DIM = 8
CAUSAL_EDGE_FEATURE_DIM = 4
CAUSAL_GLOBAL_FEATURE_NAMES = (
    "node_fraction",
    "edge_fraction",
    "current_node_visits",
    "consecutive_blocked_forwards",
    "blocked_forward_ratio",
    "parent_forward",
    "parent_lateral",
    "parent_distance",
    "has_parent",
    "decision_fraction",
    "last_forward_succeeded",
    "turn_fraction",
    "mean_node_distance",
    "max_node_distance",
)


@dataclass
class TopologyNode:
    x: float
    z: float
    parent: int | None
    visits: int = 1


@dataclass(frozen=True)
class CausalTopologyPayload:
    """Torch-free relative graph payload shared by the policy adapter and tests."""

    node_features: tuple[tuple[float, ...], ...]
    adjacency: tuple[tuple[bool, ...], ...]
    edge_features: tuple[tuple[tuple[float, ...], ...], ...]
    current_index: int
    parent_index: tuple[int, ...]
    global_features: tuple[float, ...]


class OnlineTopology:
    """A goal-free graph built from executed actions and collision feedback."""

    def __init__(
        self,
        *,
        max_decisions: int,
        forward_step_m: float = 0.25,
        turn_degrees: float = 15.0,
        node_radius_m: float = 0.5,
    ) -> None:
        if max_decisions < 1:
            raise ValueError("max_decisions must be positive")
        if forward_step_m <= 0 or turn_degrees <= 0 or node_radius_m <= 0:
            raise ValueError("topology motion constants must be positive")
        self.max_decisions = int(max_decisions)
        self.forward_step_m = float(forward_step_m)
        self.turn_radians = math.radians(float(turn_degrees))
        self.node_radius_m = float(node_radius_m)
        self.x = 0.0
        self.z = 0.0
        self.heading = 0.0
        self.nodes = [TopologyNode(0.0, 0.0, None)]
        self.edges: set[tuple[int, int]] = set()
        self.current_node = 0
        self.decision_count = 0
        self.primitive_count = 0
        self.forward_attempts = 0
        self.blocked_forwards = 0
        self.consecutive_blocked_forwards = 0
        self.last_forward_succeeded = 0.0
        self.turn_count = 0

    @staticmethod
    def _distance(a_x: float, a_z: float, b_x: float, b_z: float) -> float:
        return math.hypot(a_x - b_x, a_z - b_z)

    def _nearest_node(self) -> tuple[int, float]:
        distances = [
            self._distance(self.x, self.z, node.x, node.z) for node in self.nodes
        ]
        index = min(range(len(distances)), key=distances.__getitem__)
        return index, distances[index]

    def _update_graph(self) -> None:
        nearest, distance = self._nearest_node()
        previous = self.current_node
        if distance >= self.node_radius_m:
            self.nodes.append(TopologyNode(self.x, self.z, previous))
            self.current_node = len(self.nodes) - 1
            self.edges.add(tuple(sorted((previous, self.current_node))))
        else:
            self.current_node = nearest
            self.nodes[nearest].visits += 1
            if previous != nearest:
                self.edges.add(tuple(sorted((previous, nearest))))

    def update(
        self,
        actions: Sequence[int],
        collisions: Sequence[bool],
        *,
        realized_path_m: Sequence[float] | None = None,
        blocked_forwards: Sequence[bool] | None = None,
    ) -> None:
        if not actions or len(actions) != len(collisions):
            raise ValueError("topology update needs one collision flag per executed action")
        if len(actions) > 4:
            raise ValueError("topology update accepts at most four primitive actions")
        if (realized_path_m is None) != (blocked_forwards is None):
            raise ValueError("realized distance and blocked feedback must be supplied together")
        if realized_path_m is not None and (
            len(realized_path_m) != len(actions)
            or len(blocked_forwards) != len(actions)
        ):
            raise ValueError("causal execution feedback must align with executed actions")
        for index, (action, collision) in enumerate(zip(actions, collisions)):
            if type(action) is not int or not 0 <= action <= 3:
                raise ValueError("topology action must be an integer in 0..3")
            if type(collision) is not bool:
                raise ValueError("topology collision flags must be booleans")
            if realized_path_m is None:
                realized = self.forward_step_m if action == FORWARD_ACTION and not collision else 0.0
                blocked = bool(action == FORWARD_ACTION and collision)
            else:
                realized = float(realized_path_m[index])
                blocked = blocked_forwards[index]
                if not math.isfinite(realized) or realized < 0.0:
                    raise ValueError("realized path increments must be finite and nonnegative")
                if type(blocked) is not bool or blocked and action != FORWARD_ACTION:
                    raise ValueError("blocked feedback applies only to forward primitives")
            self.primitive_count += 1
            if action == STOP_ACTION:
                continue
            if action == LEFT_ACTION:
                self.heading += self.turn_radians
                self.turn_count += 1
                continue
            if action == RIGHT_ACTION:
                self.heading -= self.turn_radians
                self.turn_count += 1
                continue
            self.forward_attempts += 1
            if blocked:
                self.blocked_forwards += 1
                self.consecutive_blocked_forwards += 1
                self.last_forward_succeeded = 0.0
            else:
                self.consecutive_blocked_forwards = 0
                self.last_forward_succeeded = 1.0
            # G7.16 uses the causal post-action path increment.  It is still
            # dead reckoning because Habitat does not expose lateral odometry
            # to the high-level policy, but collision no longer implies zero
            # displacement when the agent slid or partially advanced.
            if realized > 0.0:
                self.x += math.sin(self.heading) * realized
                self.z += math.cos(self.heading) * realized
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        self.decision_count += 1
        self._update_graph()

    def update_from_feedback(self, raw_feedback: dict) -> None:
        """Update from the validated history-only G7.16 executor response."""

        if __package__:
            from .streamvln_executor_contract import feedback_from_dict
        else:  # Explicit direct-file mode for local contract tests only.
            from streamvln_executor_contract import feedback_from_dict

        feedback = feedback_from_dict(raw_feedback)
        self.update(
            list(feedback.executed_actions),
            [row.collision for row in feedback.primitives],
            realized_path_m=[row.realized_path_m for row in feedback.primitives],
            blocked_forwards=[row.blocked for row in feedback.primitives],
        )

    def backtrack_waypoint(self) -> tuple[float, float] | None:
        parent = self.nodes[self.current_node].parent
        if parent is None:
            return None
        target = self.nodes[parent]
        dx, dz = target.x - self.x, target.z - self.z
        forward = dx * math.sin(self.heading) + dz * math.cos(self.heading)
        lateral = dx * math.cos(self.heading) - dz * math.sin(self.heading)
        return float(forward), float(lateral)

    def feature_vector(self) -> tuple[float, ...]:
        node = self.nodes[self.current_node]
        backtrack = self.backtrack_waypoint()
        backtrack_distance = 0.0 if backtrack is None else math.hypot(*backtrack)
        blocked_ratio = self.blocked_forwards / max(1, self.forward_attempts)
        turn_fraction = self.turn_count / max(1, self.primitive_count)
        values = (
            self.x / 10.0,
            self.z / 10.0,
            math.sin(self.heading),
            math.cos(self.heading),
            len(self.nodes) / 20.0,
            len(self.edges) / 20.0,
            node.visits / 10.0,
            min(self.consecutive_blocked_forwards, 4) / 4.0,
            blocked_ratio,
            min(backtrack_distance, 5.0) / 5.0,
            float(node.parent is not None),
            min(self.decision_count, self.max_decisions) / self.max_decisions,
            self.last_forward_succeeded,
            turn_fraction,
        )
        if len(values) != len(FEATURE_NAMES) or any(not math.isfinite(v) for v in values):
            raise RuntimeError("topology feature contract failed")
        return values

    def summary(self) -> dict:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "current_node": self.current_node,
            "decision_count": self.decision_count,
            "primitive_count": self.primitive_count,
            "blocked_forwards": self.blocked_forwards,
            "forward_attempts": self.forward_attempts,
            "dead_reckoned_pose": [self.x, self.z, self.heading],
        }


def causal_topology_payload(
    topology: OnlineTopology,
    *,
    max_nodes: int,
) -> CausalTopologyPayload:
    """Build the history-only graph geometry in the current agent frame.

    This is the single geometry implementation used by the tensor snapshot
    adapter.  Its signature deliberately cannot accept goal, demonstration,
    geodesic, reference-path, or future simulator state.
    """

    if not isinstance(topology, OnlineTopology):
        raise TypeError("causal payload requires OnlineTopology")
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("causal topology max_nodes must be positive")
    node_count = len(topology.nodes)
    if not 1 <= node_count <= max_nodes:
        raise RuntimeError("causal topology exceeds the fixed policy node extent")
    if not 0 <= topology.current_node < node_count:
        raise RuntimeError("causal topology current node is invalid")

    sin_heading = math.sin(topology.heading)
    cos_heading = math.cos(topology.heading)
    relative_geometry: list[tuple[float, float]] = []
    node_features: list[tuple[float, ...]] = []
    current_parent = topology.nodes[topology.current_node].parent
    for index, node in enumerate(topology.nodes):
        dx = float(node.x) - float(topology.x)
        dz = float(node.z) - float(topology.z)
        forward = dx * sin_heading + dz * cos_heading
        lateral = dx * cos_heading - dz * sin_heading
        distance = math.hypot(forward, lateral)
        bearing = math.atan2(lateral, forward) if distance > 0.0 else 0.0
        relative_geometry.append((forward, lateral))
        node_features.append(
            (
                forward,
                lateral,
                distance,
                math.sin(bearing),
                math.cos(bearing),
                min(int(node.visits), 10) / 10.0,
                float(index == topology.current_node),
                float(index == current_parent),
            )
        )

    adjacency = [[False] * node_count for _ in range(node_count)]
    edge_features = [
        [[0.0] * CAUSAL_EDGE_FEATURE_DIM for _ in range(node_count)]
        for _ in range(node_count)
    ]
    for raw_left, raw_right in topology.edges:
        left, right = int(raw_left), int(raw_right)
        if not (0 <= left < node_count and 0 <= right < node_count and left != right):
            raise RuntimeError("causal topology contains an invalid edge")
        adjacency[left][right] = adjacency[right][left] = True
        for source, target in ((left, right), (right, left)):
            forward = relative_geometry[target][0] - relative_geometry[source][0]
            lateral = relative_geometry[target][1] - relative_geometry[source][1]
            is_tree_edge = float(
                topology.nodes[target].parent == source
                or topology.nodes[source].parent == target
            )
            edge_features[source][target] = [
                forward,
                lateral,
                math.hypot(forward, lateral),
                is_tree_edge,
            ]

    current = topology.nodes[topology.current_node]
    if current_parent is None:
        parent_forward = parent_lateral = parent_distance = 0.0
    else:
        parent_forward, parent_lateral = relative_geometry[current_parent]
        parent_distance = math.hypot(parent_forward, parent_lateral)
    node_distances = [math.hypot(*geometry) for geometry in relative_geometry]
    global_features = (
        node_count / max_nodes,
        len(topology.edges) / max(1, max_nodes - 1),
        min(int(current.visits), 10) / 10.0,
        min(int(topology.consecutive_blocked_forwards), 4) / 4.0,
        topology.blocked_forwards / max(1, topology.forward_attempts),
        max(-1.0, min(1.0, parent_forward / 5.0)),
        max(-1.0, min(1.0, parent_lateral / 5.0)),
        min(parent_distance, 5.0) / 5.0,
        float(current_parent is not None),
        min(topology.decision_count, topology.max_decisions) / topology.max_decisions,
        float(topology.last_forward_succeeded),
        topology.turn_count / max(1, topology.primitive_count),
        min(sum(node_distances) / len(node_distances), 5.0) / 5.0,
        min(max(node_distances), 5.0) / 5.0,
    )
    if any(len(row) != CAUSAL_NODE_FEATURE_DIM for row in node_features):
        raise RuntimeError("causal node feature width drifted")
    if len(global_features) != len(CAUSAL_GLOBAL_FEATURE_NAMES) or any(
        not math.isfinite(value) for value in global_features
    ):
        raise RuntimeError("causal global feature contract drifted")
    return CausalTopologyPayload(
        node_features=tuple(tuple(float(value) for value in row) for row in node_features),
        adjacency=tuple(tuple(row) for row in adjacency),
        edge_features=tuple(
            tuple(tuple(float(value) for value in feature) for feature in row)
            for row in edge_features
        ),
        current_index=int(topology.current_node),
        parent_index=tuple(
            -1 if node.parent is None else int(node.parent) for node in topology.nodes
        ),
        global_features=tuple(float(value) for value in global_features),
    )


def executed_prefix(actions: Sequence[int]) -> list[int]:
    result = []
    for action in actions:
        result.append(int(action))
        if action == STOP_ACTION:
            break
    return result


def compile_waypoint(
    waypoint_local_m: Sequence[float],
    mode: int,
    topology: OnlineTopology,
) -> list[int]:
    """Compile a high-level choice to exactly four proposed primitive actions."""
    if mode not in (NAVIGATE_MODE, STOP_MODE, RECOVER_MODE):
        raise ValueError("unknown high-level navigation mode")
    if mode == STOP_MODE:
        return [STOP_ACTION] * 4
    if len(waypoint_local_m) != 2:
        raise ValueError("local waypoint must contain forward and lateral meters")
    forward, lateral = (float(value) for value in waypoint_local_m)
    if not math.isfinite(forward) or not math.isfinite(lateral):
        raise ValueError("local waypoint must be finite")
    if mode == RECOVER_MODE:
        recovery = topology.backtrack_waypoint()
        if recovery is not None:
            forward, lateral = recovery
        else:
            forward, lateral = -0.5, 0.0

    bearing = math.atan2(lateral, forward)
    turn_steps = min(4, int(round(abs(bearing) / topology.turn_radians)))
    if turn_steps:
        turn_action = LEFT_ACTION if bearing > 0 else RIGHT_ACTION
        actions = [turn_action] * turn_steps
    else:
        actions = []
    actions.extend([FORWARD_ACTION] * (4 - len(actions)))
    if len(actions) != 4 or STOP_ACTION in actions:
        raise RuntimeError("nonterminal waypoint compilation must produce four actions")
    return actions
