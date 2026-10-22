"""Explore TRAIN successor boundaries, never regrade a rejected foot capture.

A controlled ball with a leg touch is not the same task as clean forefoot
trapping. These windows are candidate inputs for ACTUAL next-pass experiments,
not skill successes, promotions, referee decisions or executable robot actions.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HandoffWindow:
    first_control: int
    last_control: int
    earliest_next_control: int


def body_contact_kind(name: str, *, prefix: str) -> str:
    if not name.startswith(prefix):
        raise ValueError("own player contact-body witness required")
    short = name.removeprefix(prefix)
    if short in ("left_ankle_roll_link", "right_ankle_roll_link"):
        return "foot"
    if short in ("pelvis", "torso_link", "head_link") or short.startswith("waist_"):
        return "trunk_or_head"
    if any(
        short.startswith(side + part)
        for side in ("left_", "right_")
        for part in ("hip_", "knee_", "ankle_pitch")
    ):
        return "leg"
    # Whole shoulder/arm links are conservatively excluded: their geometry
    # does not encode the anatomical armpit boundary of IFAB Law12.
    return "excluded_or_unknown"


def handoff_windows(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    body_failed: np.ndarray,
    root_qposadr: int,
    root_dofadr: int,
    ball_qposadr: int,
    ball_dofadr: int,
    contacts: dict[str, dict[str, Any]],
    prefix: str,
) -> tuple[tuple[HandoffWindow, ...], tuple[str, ...]]:
    n = len(qpos)
    if (
        not 1 <= n <= 300
        or qpos.shape != (n, 10, 79)
        or qvel.shape != (n, 10, 76)
        or qpos.dtype != np.float64
        or qvel.dtype != np.float64
        or body_failed.shape != (n, 2)
        or body_failed.dtype != np.bool_
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or any(type(x) is not int for x in (root_qposadr, root_dofadr, ball_qposadr, ball_dofadr))
        or not 0 <= root_qposadr <= 72
        or not 0 <= ball_qposadr <= 72
        or not 0 <= root_dofadr <= 70
        or not 0 <= ball_dofadr <= 70
        or set(range(root_qposadr, root_qposadr + 7)) & set(range(ball_qposadr, ball_qposadr + 7))
        or set(range(root_dofadr, root_dofadr + 6)) & set(range(ball_dofadr, ball_dofadr + 6))
    ):
        raise ValueError(
            "exact finite native two-player H300 archive and disjoint free-body addresses required"
        )
    own = {name: row for name, row in contacts.items() if name.startswith(prefix)}
    for row in own.values():
        if (
            type(row.get("first_substep")) is not int
            or not 0 <= row["first_substep"] < n * 10
            or not np.isfinite(row.get("peak_force_n", float("nan")))
            or row["peak_force_n"] <= 1
        ):
            raise ValueError("actual >1N own-replay contact witnesses required")
    excluded = tuple(
        sorted(
            name for name in own if body_contact_kind(name, prefix=prefix) == "excluded_or_unknown"
        )
    )
    if excluded or not own or body_failed.any():
        return (), excluded
    first_touch = min(row["first_substep"] for row in own.values()) // 10
    q, v = qpos[:, -1], qvel[:, -1]
    root, ball = q[:, root_qposadr : root_qposadr + 7], q[:, ball_qposadr : ball_qposadr + 7]
    if not np.allclose(
        np.linalg.norm(root[:, 3:7], axis=1), 1, rtol=0, atol=1e-5
    ) or not np.allclose(np.linalg.norm(ball[:, 3:7], axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError("actual unit free-body orientations required")
    delta = ball[:, :3] - root[:, :3]
    w, x, y, z = root[:, 3:7].T
    forward = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), axis=1)
    front = (forward * delta).sum(axis=1)
    eligible = (
        (np.linalg.norm(delta[:, :2], axis=1) <= 0.65)
        & (
            np.linalg.norm(
                v[:, ball_dofadr : ball_dofadr + 2] - v[:, root_dofadr : root_dofadr + 2], axis=1
            )
            <= 0.6
        )
        & (ball[:, 2] <= 0.3)
        & (front >= 0.05)
        & (root[:, 2] >= 0.65)
        & (1 - 2 * (root[:, 4] ** 2 + root[:, 5] ** 2) >= 0.9)
    )
    eligible[:first_touch] = False
    windows = []
    start = None
    for index, good in enumerate([*eligible, False]):
        if good and start is None:
            start = index
        elif not good and start is not None:
            if index - start >= 15:
                windows.append(HandoffWindow(start, index - 1, start + 15))
            start = None
    return tuple(windows), excluded
