"""SIM-only causal root-goal inputs for the frozen motion planner.

This changes model inputs only. It is not a body trajectory, motion permit,
learned policy, or proof of a dynamically reachable target.
"""

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


def _hash(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("explicit SHA256 source identity required")


@dataclass(frozen=True)
class RootWaypointIntent:
    world_target_xyz_m: tuple[float, float, float]
    world_target_yaw_rad: float
    measured_observation_hash: str
    goal_basis_hash: str

    def __post_init__(self) -> None:
        for h in (self.measured_observation_hash, self.goal_basis_hash):
            _hash(h)
        if (
            type(self.world_target_xyz_m) is not tuple
            or len(self.world_target_xyz_m) != 3
            or any(type(x) is not float or not math.isfinite(x) for x in self.world_target_xyz_m)
            or max(abs(x) for x in self.world_target_xyz_m[:2]) > 20
            or not 0.65 <= self.world_target_xyz_m[2] <= 0.9
            or type(self.world_target_yaw_rad) is not float
            or not math.isfinite(self.world_target_yaw_rad)
            or not -math.pi <= self.world_target_yaw_rad <= math.pi
        ):
            raise ValueError(
                "finite bounded world root goal with current-state provenance required"
            )

    def report(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.planner_root_waypoint_intent.v1",
            **asdict(self),
            four_waypoints="same_known_goal_not_future_measured_body_positions",
            dynamically_qualified=False,
            activation_ceiling="SIM_ONLY",
            hardware_authorized=False,
        )

    @property
    def contract_hash(self) -> str:
        return hash_json(self.report())


def apply_waypoint_intent(
    feed: dict[str, np.ndarray], intent: RootWaypointIntent, *, activate: bool
) -> dict[str, np.ndarray]:
    """Owned copies; no mutation of caller reference/history or measured body."""
    intent.__post_init__()
    if type(activate) is not bool or type(feed) is not dict:
        raise ValueError("explicit boolean planner feature activation required")
    c = feed.get("context_mujoco_qpos")
    if (
        type(c) is not np.ndarray
        or c.shape != (1, 4, 36)
        or c.dtype != np.float32
        or not np.isfinite(c).all()
        or np.any(abs(np.linalg.norm(c[0, :, 3:7], axis=1) - 1) > 1e-5)
        or np.linalg.norm(c[0, -1, :2] - intent.world_target_xyz_m[:2]) > 0.5
    ):
        raise ValueError("finite measured context and local half-meter goal required")
    expected = dict(
        has_specific_target=((1, 1), np.int64),
        specific_target_positions=((1, 4, 3), np.float32),
        specific_target_headings=((1, 4), np.float32),
    )
    for k, (shape, dtype) in expected.items():
        a = feed.get(k)
        if (
            type(a) is not np.ndarray
            or a.shape != shape
            or a.dtype != dtype
            or not np.isfinite(a).all()
        ):
            raise ValueError("exact ONNX waypoint tensor contract required: " + k)
    out = {k: v.copy() for k, v in feed.items()}
    out["has_specific_target"][:] = int(activate)
    out["specific_target_positions"][:] = intent.world_target_xyz_m
    out["specific_target_headings"][:] = intent.world_target_yaw_rad
    return out


def past_context_at_30hz(canonical_qpos: np.ndarray, frame: int) -> np.ndarray:
    """Resample past measured 50Hz history only, not future physical states.

    Quaternion normalization is solely NN input preparation, never a correction
    to any simulator state or historical artifact.
    """
    if (
        type(canonical_qpos) is not np.ndarray
        or canonical_qpos.ndim != 2
        or canonical_qpos.shape[1] != 43
        or canonical_qpos.dtype != np.float64
        or type(frame) is not int
        or not 5 <= frame < len(canonical_qpos)
    ):
        raise ValueError("actual canonical50Hz history and bounded current frame required")
    rows = []
    for t in frame - np.array([5.0, 10.0 / 3.0, 5.0 / 3.0, 0.0]):
        lo, hi = int(np.floor(t)), int(np.ceil(t))
        a, b = canonical_qpos[lo, :36].copy(), canonical_qpos[hi, :36].copy()
        if (
            not np.isfinite(a).all()
            or not np.isfinite(b).all()
            or abs(np.linalg.norm(a[3:7]) - 1) > 1e-5
            or abs(np.linalg.norm(b[3:7]) - 1) > 1e-5
        ):
            raise ValueError("finite unit measured quaternion history required")
        if np.dot(a[3:7], b[3:7]) < 0:
            b[3:7] *= -1
        value = (1 - (t - lo)) * a + (t - lo) * b
        value[3:7] /= np.linalg.norm(value[3:7])
        rows.append(value)
    return np.asarray(rows, dtype=np.float32)[None]


def validate_planner_output(
    output: np.ndarray, count: np.ndarray, joint_limits: np.ndarray
) -> dict[str, Any]:
    if (
        type(output) is not np.ndarray
        or output.ndim != 3
        or output.shape[0] != 1
        or output.shape[2] != 36
        or type(count) is not np.ndarray
        or count.shape != (1,)
        or count.dtype.kind not in "iu"
        or not 4 <= int(count[0]) <= 64
        or output.shape[1] < int(count[0])
        or not np.isfinite(output).all()
        or not np.issubdtype(output.dtype, np.floating)
        or joint_limits.shape != (29, 2)
        or not np.isfinite(joint_limits).all()
        or np.any(joint_limits[:, 0] >= joint_limits[:, 1])
    ):
        raise ValueError("bounded finite actual planner output and original joint limits required")
    poses = output[0, : int(count[0])].astype(np.float64)
    if np.any(abs(np.linalg.norm(poses[:, 3:7], axis=1) - 1) > 0.01):
        raise ValueError("unit planner output quaternions required; no output repair")
    violation = np.maximum(joint_limits[:, 0] - poses[:, 7:], poses[:, 7:] - joint_limits[:, 1])
    max_violation = float(max(0.0, violation.max()))
    return dict(
        output_frames=int(count[0]),
        final_root_xyz_m=poses[-1, :3].tolist(),
        maximum_native_joint_limit_violation_rad=max_violation,
        reference_within_native_joint_limits=max_violation <= 1e-6,
        dynamic_feasibility_proven=False,
    )
