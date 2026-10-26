"""Explicit SIM-only residual-torque intention, not a relaxed position plan.

The frozen parent retains its PD setpoints. A C2, <=25%-of-effort envelope
adds torque AFTER the parent's PD saturation. Total effort is saturated to the
UNCHANGED native caps; intervention is suppressed outwards at measured stops.
This is not algebraically equivalent to adding a PD-position offset.
"""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_reference_plan import _c2_offsets


@dataclass(frozen=True)
class CFtrackTorqueIntentPlan:
    start_control: int
    peak_control: int
    fraction_effort_hg: tuple[float, ...]
    body_hash: str
    parent_policy_hash: str

    def offsets(self) -> np.ndarray:
        if (
            type(self.start_control) is not int
            or not 0 <= self.start_control <= 300
            or type(self.peak_control) is not int
            or not 1 <= self.peak_control < 48
            or type(self.fraction_effort_hg) is not tuple
            or len(self.fraction_effort_hg) != 29
            or any(
                type(v) is not float or not np.isfinite(v) or abs(v) > 0.25
                for v in self.fraction_effort_hg
            )
            or any(
                type(h) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", h) is None
                for h in (self.body_hash, self.parent_policy_hash)
            )
        ):
            raise ValueError("immutable body/parent-bound HG29 SIM torque intention required")
        delta = np.array(self.fraction_effort_hg)
        rate = 1.875 * abs(delta).max() / min(self.peak_control, 48 - self.peak_control) / 0.02
        if rate > 2.0:
            raise ValueError("additional continuous torque envelope exceeds 2 effort-fractions/s")
        return _c2_offsets(delta, tuple(range(29)), self.peak_control)[0]


class CFtrackTorqueIntentPolicy:
    """Pure numeric command/actual-feedback protocol; no Runtime or transport.

    Only cold, explicitly qualified use through SharedNativeTorqueLayer is
    admitted. This policy has no optimizer and is NOT a trained neural actor.
    """

    def __init__(self, plan: CFtrackTorqueIntentPlan, effort: np.ndarray):
        if type(plan) is not CFtrackTorqueIntentPlan:
            raise ValueError("exact torque intention plan required")
        self._offsets = plan.offsets()
        if (
            type(effort) is not np.ndarray
            or effort.shape != (29,)
            or effort.dtype != np.float64
            or not np.isfinite(effort).all()
            or (effort <= 0).any()
        ):
            raise ValueError("positive actual float64 HG29 effort caps required")
        self.plan = plan
        self.effort = effort.copy()
        self.effort.setflags(write=False)
        self.body_hash = plan.body_hash
        self.parent_policy_hash = plan.parent_policy_hash
        self.reset()

    def reset(self) -> None:
        if getattr(self, "_soccer_native_layer_owner", None) is not None:
            raise ValueError("bound native torque intention cannot reset its physical clock")
        self.steps = self.saturation_steps = self.stop_guard_steps = 0
        self._pending: np.ndarray | None = None
        self.failed = False

    def command(self, frame: Any, parent: np.ndarray) -> np.ndarray:
        if self.failed or self._pending is not None:
            raise ValueError("consecutive command and actual feedback required")
        q, lo, hi = (
            np.asarray(getattr(frame, k))
            for k in ("joint_position", "joint_lower_limits", "joint_upper_limits")
        )
        if any(v.shape != (29,) or not np.isfinite(v).all() for v in (q, lo, hi, parent)) or (
            parent.dtype != np.float64 or (abs(parent) > self.effort).any() or (lo >= hi).any()
        ):
            self.failed = True
            raise ValueError("finite measured joints and bounded native parent torque required")
        relative = self.steps / 10 - self.plan.start_control
        offset = np.zeros(29)
        if 0 <= relative < 48:
            index = int(relative)
            alpha = relative - index
            # Fixed 500 Hz interpolation of the declared 50 Hz C2 samples.
            offset = (1 - alpha) * self._offsets[index] + alpha * self._offsets[index + 1]
        residual = offset * self.effort
        outward = ((q <= lo) & (residual < 0)) | ((q >= hi) & (residual > 0))
        self.stop_guard_steps += int(outward.any())
        residual[outward] = 0.0
        proposed = parent.copy()
        changed = residual != 0
        proposed[changed] += residual[changed]
        applied: np.ndarray = np.clip(proposed, -self.effort, self.effort)
        self.saturation_steps += int(not np.array_equal(applied, proposed))
        # Preserve untouched parent coordinates (including signed zero).
        applied[~changed] = parent[~changed]
        self._pending = applied.copy()
        return applied.copy()

    def note_applied(self, actual: np.ndarray) -> None:
        if (
            self.failed
            or self._pending is None
            or type(actual) is not np.ndarray
            or actual.dtype != np.float64
            or actual.shape != (29,)
            or not np.array_equal(actual, self._pending)
        ):
            self.failed = True
            raise ValueError("exact actual native force feedback required")
        self._pending = None
        self.steps += 1
