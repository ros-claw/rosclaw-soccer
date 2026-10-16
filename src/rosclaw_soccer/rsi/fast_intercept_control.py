"""Single-factor SIM-only precontact speed authority, not a learned policy."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True, slots=True)
class FastInterceptControl(BoundedFirstTouchControl):
    precontact_forward_cap_m_s: float = 2.0

    def __post_init__(self) -> None:
        BoundedFirstTouchControl.__post_init__(self)
        base = BoundedFirstTouchControl()
        if (
            type(self.precontact_forward_cap_m_s) is not float
            or self.precontact_forward_cap_m_s != 2.0
            or any(getattr(self, k) != v for k, v in asdict(base).items())
        ):
            raise ValueError(
                "fixed single-factor 2.0 precontact / 1.4 postcontact SIM probe required"
            )

    def navigation(
        self, *, gap_x: float, gap_y: float, ball_vx: float, before_contact: bool
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        raw, projected = BoundedFirstTouchControl.navigation(
            self, gap_x=gap_x, gap_y=gap_y, ball_vx=ball_vx, before_contact=before_contact
        )
        if before_contact:
            projected = (min(2.0, max(0.0, raw[0])), projected[1], projected[2])
        return raw, projected

    def contract(self) -> dict[str, Any]:
        self.__post_init__()
        value = dict(
            schema="soccer.rsi.fast_intercept_control.v1",
            base_control=BoundedFirstTouchControl().contract(),
            precontact_forward_cap_m_s=2.0,
            postcontact_forward_cap_m_s=1.4,
            maximum_navigation_speed_m_s=2.1,
            transition_trigger="first previously measured any-body ball contact > 1N",
            source_hash=hash_bytes(Path(__file__).read_bytes()),
            current_measured_state_only=True,
            future_state_read=False,
            physical_joint_and_torque_limits_unchanged=True,
            trained_policy=False,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            hardware_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))


def require_fast_navigation_trace(trace: dict[str, Any]) -> None:
    """Independent scalar reconstruction, not a producer navigation() call."""
    shapes = {
        "root_pose_xyzw_m": (300, 1, 7),
        "ball_position_before_step_m": (300, 1, 3),
        "ball_linear_velocity_before_step_m_s": (300, 1, 3),
        "navigation_command": (300, 1, 3),
        "approach_requested_navigation_command": (300, 1, 3),
        "force_n": (300, 1, 6),
    }
    for key, shape in shapes.items():
        raw = np.asarray(trace.get(key))
        if raw.shape != shape or raw.dtype != np.float64 or not np.isfinite(raw).all():
            raise ValueError("whole finite float64 fast-intercept measurements required")
    contacts = np.flatnonzero(np.any(trace["force_n"][:, 0] > 1.0, axis=1))
    if np.any(trace["force_n"] < 0):
        raise ValueError("nonnegative measured contact forces required")
    first = int(contacts[0]) if len(contacts) else None
    for frame in range(300):
        before = first is None or frame <= first
        gap = (
            trace["ball_position_before_step_m"][frame, 0] - trace["root_pose_xyzw_m"][frame, 0, :3]
        )
        if before:
            forward = float(trace["ball_linear_velocity_before_step_m_s"][frame, 0, 0]) + 1.2 * (
                float(gap[0]) - 0.35
            )
            lateral = (
                1.2 * float(gap[1]) * min(1.0, max(0.0, (float(gap[0]) - 0.35) / (0.95 - 0.35)))
            )
        else:
            forward, lateral = 1.4, 0.0
        expected_requested = (forward, lateral, 0.0)
        actual = (min(2.0 if before else 1.4, max(0.0, forward)), min(0.2, max(-0.2, lateral)), 0.0)
        if not np.allclose(
            expected_requested,
            trace["approach_requested_navigation_command"][frame, 0],
            atol=1e-12,
            rtol=0,
        ) or not np.allclose(actual, trace["navigation_command"][frame, 0], atol=1e-12, rtol=0):
            raise ValueError("fast-intercept requested/projected commands differ from scalar rule")
