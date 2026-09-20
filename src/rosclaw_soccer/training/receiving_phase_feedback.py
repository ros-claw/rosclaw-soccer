"""Measured-state phase alignment of an offline prior, not an E2 certificate.

No simulator handles or future live observations enter this interface. A
reference can contain privileged offline data and must be labelled accordingly.
This small diagnostic tests timing feedback, not a new motor authority.
"""

import re
from dataclasses import asdict, dataclass

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


def receiving_phase_features(
    pelvis_pose: np.ndarray,
    leg_position: np.ndarray,
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
) -> tuple[float, ...]:
    """Current ball translation/velocity in pelvis-yaw axes plus 12 leg angles.

    Velocity is absolute ball velocity rotated to body axes, not relative to
    pelvis velocity. Source and live features must use exactly this convention.
    Units are normalized by 0.2 m, 0.5 m/s and 0.3 rad respectively.
    """
    arrays = tuple(np.asarray(x) for x in (pelvis_pose, leg_position, ball_position, ball_velocity))
    if any(
        a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all()
        for a, shape in zip(arrays, ((7,), (12,), (3,), (3,)), strict=True)
    ):
        raise ValueError("finite current pose, leg state and ball state required")
    pelvis, joints, ball, velocity = arrays
    if abs(np.linalg.norm(pelvis[3:]) - 1) > 1e-5:
        raise ValueError("unit pelvis quaternion required")
    w, x, y, z = pelvis[3:]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    rotate = np.asarray(((c, s, 0), (-s, c, 0), (0, 0, 1)))
    features = np.r_[rotate @ (ball - pelvis[:3]) / 0.2, rotate @ velocity / 0.5, joints / 0.3]
    if not np.isfinite(features).all() or np.any(abs(features) > 1000):
        raise ValueError("phase features outside diagnostic envelope")
    return tuple(float(v) for v in features)


@dataclass(frozen=True)
class ReceivingPhaseReference:
    schedule_hash: str
    source_evidence_hash: str
    start_frame: int
    features: tuple[tuple[float, ...], ...]
    feedback_enabled: bool = True

    def __post_init__(self) -> None:
        if (
            any(
                type(h) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", h) is None
                for h in (self.schedule_hash, self.source_evidence_hash)
            )
            or type(self.start_frame) is not int
            or not 0 <= self.start_frame < 1000
            or type(self.feedback_enabled) is not bool
            or type(self.features) is not tuple
            or not 1 <= len(self.features) <= 1000 - self.start_frame
        ):
            raise ValueError("bounded source-bound phase reference required")
        for row in self.features:
            if (
                type(row) is not tuple
                or len(row) != 18
                or any(
                    type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 1000 for v in row
                )
            ):
                raise ValueError("finite immutable 18-dimensional phase features required")

    @property
    def contract_hash(self) -> str:
        return str(
            hash_json(
                dict(
                    schema="soccer.receiving_phase_reference.v1",
                    **asdict(self),
                    feature_contract="pelvis_yaw_ball_translation_abs_velocity_leg_angles.v1",
                    max_phase_advance_frames=4,
                    max_clock_deviation_frames=15,
                    temporal_tiebreak_weight=0.01,
                    physical_step_sec=0.02,
                    qualified_teacher=False,
                )
            )
        )


class ReceivingPhaseCursor:
    def __init__(self, reference: ReceivingPhaseReference) -> None:
        reference.__post_init__()
        self.reference = reference
        self._features = np.asarray(reference.features)
        self.next_frame = 0
        self.phase = reference.start_frame
        self.faulted = False

    def step(self, frame: int, features: tuple[float, ...]) -> int | None:
        if self.faulted:
            raise ValueError("phase feedback fault is latched")
        try:
            if (
                type(frame) is not int
                or frame != self.next_frame
                or frame >= 1000
                or type(features) is not tuple
                or len(features) != 18
                or any(
                    type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 1000
                    for v in features
                )
            ):
                raise ValueError(
                    "consecutive frames and finite immutable current features required"
                )
            self.next_frame += 1
            start = self.reference.start_frame
            end = start + len(self.reference.features) - 1
            if frame < start:
                return None
            if not self.reference.feedback_enabled:
                self.phase = min(frame, end)
                return self.phase
            # First controlled frame is common to clock and feedback. Later
            # phases only move forward, at most four reference frames per tick.
            low = min(end, max(self.phase, frame - 15, start))
            high = min(end, self.phase + (0 if frame == start else 4), frame + 15)
            if low > high:
                raise ValueError("phase reference exhausted outside timing envelope")
            candidates = np.arange(low, high + 1)
            difference = self._features[candidates - start] - np.asarray(features)
            cost = np.mean(difference**2, axis=1)
            cost += 0.01 * ((candidates - (self.phase + 1)) / 4) ** 2
            self.phase = int(candidates[int(np.argmin(cost))])
            return self.phase
        except (ValueError, TypeError, FloatingPointError):
            self.faulted = True
            raise
