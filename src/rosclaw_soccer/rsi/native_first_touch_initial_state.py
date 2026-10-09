"""Independent initial-state checks for the native CPU first-touch experiment.

The caller must obtain the course and canonical joint defaults from its pinned
exam declaration and frozen foundation source, NOT from the trace being checked.
This checker neither opens an exam nor proves course freshness or admission.
Its contract deliberately contains hashes only, never private coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _vector(value: Any, width: int) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.shape != (width,)
        or value.dtype != np.dtype(np.float64)
        or not np.isfinite(value).all()
    ):
        raise ValueError("finite canonical float64 initial-state vector required")
    return value


def _course(value: Any, container: type) -> tuple[float, float, float]:
    if type(value) is not container:
        raise ValueError("finite ordinary three-coordinate declared course required")
    values = cast(list[Any] | tuple[Any, ...], value)
    if len(values) != 3 or any(type(v) not in (int, float) for v in values):
        raise ValueError("finite ordinary three-coordinate declared course required")
    try:
        coordinates = (float(values[0]), float(values[1]), float(values[2]))
    except OverflowError:
        raise ValueError("finite ordinary three-coordinate declared course required") from None
    if not all(math.isfinite(v) for v in coordinates):
        raise ValueError("finite ordinary three-coordinate declared course required")
    return coordinates


@dataclass(frozen=True, init=False)
class NativeFirstTouchInitialState:
    """Owned exact initial state for the existing 43q/41v native protocol.

    Fixed root height .793 m, ball height .13 m, ball radius .11 m and
    positive y-axis rolling velocity preserve the existing experiment.
    No numeric tolerance, pose correction, clipping or domain expansion occurs.
    Initial state is canonical HG joint order, not raw MuJoCo joint order.
    """

    _course_value: tuple[float, float, float] = field(repr=False)
    _qpos_bytes: bytes = field(repr=False)
    _qvel_bytes: bytes = field(repr=False)

    def __init__(
        self, course: tuple[float, float, float], canonical_joint_defaults: np.ndarray
    ) -> None:
        coordinates = _course(course, tuple)
        joints = _vector(canonical_joint_defaults, 29)
        x, y, vx = coordinates
        qpos, qvel = np.zeros(43, dtype=np.float64), np.zeros(41, dtype=np.float64)
        qpos[:7] = (0, 0, 0.793, 1, 0, 0, 0)
        qpos[7:36] = joints
        qpos[36:43] = (x, y, 0.13, 1, 0, 0, 0)
        qvel[35:41] = (vx, 0, 0, 0, vx / 0.11, 0)
        if not np.isfinite(qvel).all():
            raise ValueError("finite native initial rolling velocity required")
        object.__setattr__(self, "_course_value", coordinates)
        object.__setattr__(self, "_qpos_bytes", qpos.tobytes())
        object.__setattr__(self, "_qvel_bytes", qvel.tobytes())

    def verify(self, declared_course: Any, canonical_qpos: Any, canonical_qvel: Any) -> None:
        """Reject mismatched declarations or initial state before replay steps."""
        if _course(declared_course, list) != self._course_value:
            raise ValueError("initial course differs from independent declaration")
        qpos, qvel = _vector(canonical_qpos, 43), _vector(canonical_qvel, 41)
        if not np.array_equal(qpos, np.frombuffer(self._qpos_bytes, dtype=np.float64)):
            raise ValueError("initial canonical position differs from independent declaration")
        if not np.array_equal(qvel, np.frombuffer(self._qvel_bytes, dtype=np.float64)):
            raise ValueError("initial canonical velocity differs from independent declaration")

    def contract(self) -> dict[str, Any]:
        """Public hash-only evidence; no private initial-state values or seeds."""
        return {
            "schema": "soccer.rsi.native_first_touch_initial_state.v1",
            "course_hash": hash_json(list(self._course_value)),
            "canonical_qpos_float64_hash": hash_bytes(self._qpos_bytes),
            "canonical_qvel_float64_hash": hash_bytes(self._qvel_bytes),
            "canonical_joint_names_hash": hash_json(list(G1_DDS_JOINT_NAMES)),
            "initial_state_checked_exactly": True,
            "course_freshness_certified": False,
            "exam_admission_authorized": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
