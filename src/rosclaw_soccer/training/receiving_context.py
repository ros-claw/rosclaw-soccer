"""Causal incoming-ball features for physical-to-tactical evidence transfer.

This extracts observations, not a success probability or a skill transition.
The caller chooses a fixed observation frame before inspecting outcomes. No
future contact time, nominal launch speed, or future pelvis pose is used.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReceivingContext:
    time_sec: float
    ball_forward_m: float
    ball_lateral_m: float
    ball_height_m: float
    ball_speed_mps: float
    relative_forward_velocity_mps: float
    relative_lateral_velocity_mps: float
    player_speed_mps: float

    def __post_init__(self) -> None:
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("finite typed measured receiving context required")
        if min(self.time_sec, self.ball_height_m, self.ball_speed_mps, self.player_speed_mps) < 0:
            raise ValueError("nonnegative time, height and speed required")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def receiving_context(trace: Mapping[str, Any], *, agent_id: str, frame: int) -> ReceivingContext:
    """Read current/prior rows only; require 50 Hz and no recorded contact yet.

    Pelvis speed is a backward difference. Ball speed is measured horizontal
    velocity, not nominal course launch speed. Coordinates are pelvis-yaw local.
    No recorded contact is a 50 Hz observation claim, not a microstep proof.
    Later rows may be absent or invalid without affecting this causal query.
    """
    if (
        type(agent_id) is not str
        or re.fullmatch(r"(?:red|blue)\.(?:defender|finisher|goalkeeper|playmaker)", agent_id)
        is None
        or type(frame) is not int
        or frame < 1
    ):
        raise ValueError("named player and a current frame with measured predecessor required")
    key = agent_id.replace(".", "_")

    def prefix(name: str, tail: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(trace[name])
        if value.ndim != len(tail) + 1 or value.shape[1:] != tail or len(value) <= frame:
            raise ValueError("aligned measured observation prefix required")
        value = value[: frame + 1]
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("finite measured observation prefix required")
        return value

    time = prefix("time", ())
    if time[0] < 0 or not np.allclose(np.diff(time), 0.02, rtol=0, atol=1e-8):
        raise ValueError("contiguous measured 50 Hz prefix required")
    for code_key, force_key in (
        ("ball_contact_agent_code", "ball_contact_force_n"),
        ("ball_nonfoot_contact_agent_code", "ball_nonfoot_contact_force_n"),
    ):
        codes, forces = prefix(code_key, ()), prefix(force_key, ())
        if (
            np.any(codes < 0)
            or np.any(codes > 8)
            or np.any(codes != np.floor(codes))
            or np.any(forces < 0)
            or np.any((codes > 0) & (forces > 0))
        ):
            raise ValueError("fixed precontact observation requires no recorded contact yet")
    body = prefix(key + "_pelvis_pose", (7,))
    ball = prefix("ball_pose", (7,))
    velocity = prefix("ball_velocity", (6,))
    if not np.allclose(np.linalg.norm(body[:, 3:], axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("normalized measured pelvis orientation required")
    w, x, y, z = body[frame, 3:]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, s], [-s, c]])
    position = rotation @ (ball[frame, :2] - body[frame, :2])
    player_velocity = (body[frame, :2] - body[frame - 1, :2]) / (time[frame] - time[frame - 1])
    relative_velocity = rotation @ (velocity[frame, :2] - player_velocity)
    return ReceivingContext(
        time_sec=float(time[frame]),
        ball_forward_m=float(position[0]),
        ball_lateral_m=float(position[1]),
        ball_height_m=float(ball[frame, 2]),
        ball_speed_mps=float(np.linalg.norm(velocity[frame, :2])),
        relative_forward_velocity_mps=float(relative_velocity[0]),
        relative_lateral_velocity_mps=float(relative_velocity[1]),
        player_speed_mps=float(np.linalg.norm(player_velocity)),
    )
