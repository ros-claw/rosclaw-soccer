"""Explicit, bounded SIM motor authority for a future CFtrack physics teacher.

Unlike contact reference plans this modifies freshly decoded NN joint targets.
It never executes them, raises torque caps, clips an invalid target, or grants
hardware authority. Installation must separately bind the donor state hash.
"""

import re
from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.rsi.contact_reference_plan import _c2_offsets


@dataclass(frozen=True)
class CFtrackMotorResidualPlan:
    origin_counter: int
    start_control: int
    peak_control: int
    delta_hg: tuple[float, ...]
    donor_policy_hash: str

    def require_donor(self, *, policy_hash: str, counter: int) -> None:
        self.offsets()
        if (
            type(counter) is not int
            or counter != self.origin_counter
            or policy_hash != self.donor_policy_hash
        ):
            raise ValueError("motor plan must be installed at its exact donor policy boundary")

    def offsets(self) -> tuple[np.ndarray, np.ndarray]:
        if (
            any(type(v) is not int or v < 0 for v in (self.origin_counter, self.start_control))
            or self.start_control < self.origin_counter
            or type(self.peak_control) is not int
            or not 1 <= self.peak_control < 48
            or type(self.delta_hg) is not tuple
            or len(self.delta_hg) != 29
            or any(
                type(v) is not float or not np.isfinite(v) or abs(v) > 0.25 for v in self.delta_hg
            )
            or type(self.donor_policy_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.donor_policy_hash) is None
        ):
            raise ValueError("immutable current-origin HG29 .25rad SIM motor plan required")
        delta = np.array(self.delta_hg, np.float64)
        if 1.875 * abs(delta).max() / min(self.peak_control, 48 - self.peak_control) / 0.02 > 4.0:
            raise ValueError("motor residual exceeds continuous additional4rad/s envelope")
        return _c2_offsets(delta, tuple(range(29)), self.peak_control)

    def apply(self, target: np.ndarray, *, counter: int, limits_hg: np.ndarray) -> np.ndarray:
        pos, _ = self.offsets()
        if (
            type(counter) is not int
            or counter < self.origin_counter
            or type(target) is not np.ndarray
            or target.shape != (29,)
            or target.dtype != np.float64
            or not np.isfinite(target).all()
            or type(limits_hg) is not np.ndarray
            or limits_hg.shape != (29, 2)
            or limits_hg.dtype != np.float64
            or not np.isfinite(limits_hg).all()
            or (limits_hg[:, 0] >= limits_hg[:, 1]).any()
        ):
            raise ValueError("finite fresh HG29 target, current clock and physical limits required")
        relative = counter - self.start_control
        offset = pos[relative] if 0 <= relative < 49 else np.zeros(29)
        modified = offset != 0
        result = target.copy()
        result[modified] += offset[modified]
        # Zero branch must preserve the original NN, even if an unchanged
        # original coordinate is beyond a joint range. Never silently clip it.
        if (
            np.any(modified & ((result < limits_hg[:, 0]) | (result > limits_hg[:, 1])))
            or not np.isfinite(result).all()
        ):
            raise ValueError("modified motor target outside physical joint limits; no clipping")
        return result
