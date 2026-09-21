"""Bounded open-loop oracle proposals for action-authority experiments only.

An optimizer may choose knots using privileged simulation feedback. This is
not a learned policy, not an online teacher and not promotion evidence.
"""

import re
from dataclasses import asdict, dataclass

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class ReceivingOracleSchedule:
    agent_id: str
    substrate: str
    start_frame: int
    knot_frames: int
    knots: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        dimensions = {"A0_leg12": 12, "A1_body29": 29, "A3_sonic_residual": 29}
        if (
            type(self.agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or self.substrate not in dimensions
            or type(self.start_frame) is not int
            or not 0 <= self.start_frame <= 1000
            or type(self.knot_frames) is not int
            or not 1 <= self.knot_frames <= 100
            or type(self.knots) is not tuple
            or not 1 <= len(self.knots) <= 32
        ):
            raise ValueError("bounded explicit oracle schedule required")
        for row in self.knots:
            if (
                type(row) is not tuple
                or len(row) != dimensions[self.substrate]
                or any(type(x) not in (int, float) or not np.isfinite(x) or abs(x) > 1 for x in row)
            ):
                raise ValueError("finite normalized oracle knots required")

    @property
    def contract_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema": "soccer.receiving_oracle_schedule.v1",
                    **asdict(self),
                    "offset_rad": 0.1,
                    "smoothing": 0.25,
                    "step_rad": 0.02,
                }
            )
        )


class ReceivingOracleCursor:
    """Private per-rollout filter with the original leg residual's envelope."""

    def __init__(self, schedule: ReceivingOracleSchedule) -> None:
        schedule.__post_init__()
        self.schedule = schedule
        self.next_frame = 0
        self.previous: np.ndarray | None = None
        self.faulted = False

    def step(
        self,
        frame: int,
        *,
        active: bool,
        predecessor: np.ndarray,
        reference_frame: int | None = None,
        desired_override_rad: tuple[float, ...] | None = None,
    ) -> np.ndarray | None:
        if self.faulted:
            raise ValueError("oracle cursor fault is latched")
        try:
            if type(frame) is not int or frame != self.next_frame or type(active) is not bool:
                raise ValueError("consecutive oracle frames and explicit admission required")
            if reference_frame is not None and (
                type(reference_frame) is not int
                or not self.schedule.start_frame <= reference_frame < 1000
                or frame < self.schedule.start_frame
            ):
                raise ValueError("explicit bounded post-entry reference frame required")
            if desired_override_rad is not None and (
                self.schedule.substrate not in ("A0_leg12", "A1_body29")
                or reference_frame is not None
                or frame < self.schedule.start_frame
                or type(desired_override_rad) is not tuple
                or len(desired_override_rad) != len(self.schedule.knots[0])
                or any(
                    type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 0.1
                    for v in desired_override_rad
                )
            ):
                raise ValueError("bounded post-entry feedback cannot mix with phase override")
            old = np.asarray(predecessor)
            if old.shape != (12,) or not np.isfinite(old).all() or np.any(abs(old) > 0.100000001):
                raise ValueError("bounded actual predecessor required")
            self.next_frame += 1
            if frame < self.schedule.start_frame:
                return None
            dimension = len(self.schedule.knots[0])
            if self.previous is None:
                self.previous = np.zeros(dimension)
                # SONIC has no preceding leg residual; do not invent one.
                if self.schedule.substrate != "A3_sonic_residual":
                    self.previous[:12] = old
            selected_frame = frame if reference_frame is None else reference_frame
            offset = (selected_frame - self.schedule.start_frame) / self.schedule.knot_frames
            left = min(int(offset), len(self.schedule.knots) - 1)
            right = min(left + 1, len(self.schedule.knots) - 1)
            fraction = min(offset - left, 1.0)
            knots = np.asarray(self.schedule.knots)
            desired = 0.1 * ((1 - fraction) * knots[left] + fraction * knots[right])
            if desired_override_rad is not None:
                desired = np.asarray(desired_override_rad, dtype=np.float64)
            if not active:
                desired[:] = 0
            self.previous += np.clip(0.25 * (desired - self.previous), -0.02, 0.02)
            return self.previous.copy()
        except (ValueError, TypeError, FloatingPointError):
            self.faulted = True
            raise
