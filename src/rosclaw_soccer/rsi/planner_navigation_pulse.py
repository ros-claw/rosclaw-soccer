"""One fixed pre-planner lateral authority probe; not learning or deployment.

An independent future SIM runner must bind this to a zero leg-reference model,
the original parent and a 20-frame SONIC planner. This values-only adapter
does not own a world, Foundation, actuator, Runtime, or execution permission.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True, slots=True)
class PlannerNavigationPulse:
    lateral_delta_m_s: float = 0.0
    start_frame: int = 100
    end_frame: int = 160

    def __post_init__(self) -> None:
        if (
            type(self.lateral_delta_m_s) is not float
            or self.lateral_delta_m_s not in (0.0, -0.08, 0.08)
            or type(self.start_frame) is not int
            or self.start_frame != 100
            or type(self.end_frame) is not int
            or self.end_frame != 160
        ):
            raise ValueError("fixed zero or signed 0.08m/s pulse on [100,160) only required")

    def compose(
        self, frame: int, baseline: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Requested residual and projected command; no stateful implicit clock.

        Hold the same residual between the 100/120/140 planning events and
        reset at 160. Actual reference/encoder/body onset must be measured,
        not inferred from either this request or the future write start.
        """
        self.__post_init__()
        if (
            type(frame) is not int
            or not 0 <= frame < 300
            or type(baseline) is not tuple
            or len(baseline) != 3
            or any(type(v) is not float or not math.isfinite(v) for v in baseline)
            or not 0.0 <= baseline[0] <= 1.48
            or abs(baseline[1]) > 0.2
            or baseline[2] != 0.0
        ):
            raise ValueError(
                "original finite bounded non-turning navigation and H300 clock required"
            )
        delta = self.lateral_delta_m_s if self.start_frame <= frame < self.end_frame else 0.0
        requested = (0.0, delta, 0.0)
        # Preserve all bits, including signed zero, on the entire OFF path and
        # every inactive frame. Do not manufacture baseline + zero arithmetic.
        if delta == 0.0:
            return requested, baseline
        command = (baseline[0], min(0.2, max(-0.2, baseline[1] + delta)), baseline[2])
        if math.hypot(*command[:2]) > 1.5:
            raise ValueError("original 1.5m/s command envelope must not be widened")
        return requested, command

    def contract(self) -> dict[str, Any]:
        self.__post_init__()
        value = dict(
            schema=(
                "soccer.rsi.pre_planner_navigation_positive_pulse.v2"
                if self.lateral_delta_m_s > 0
                else "soccer.rsi.pre_planner_navigation_pulse.v1"
            ),
            parameters=asdict(self),
            source_hash=hash_bytes(Path(__file__).read_bytes()),
            official_horizon_frames=300,
            control_dt_s=0.02,
            planner_event_stride_frames=20,
            active_planner_event_frames=[100, 120, 140],
            reset_planner_event_frame=160,
            forward_command_unchanged=True,
            yaw_command_unchanged=True,
            original_lateral_cap_m_s=0.2,
            original_total_speed_cap_m_s=1.5,
            zero_leg_reference_and_original_parent_required=True,
            reference_lookahead_unchanged=True,
            reference_and_encoder_onset_must_be_measured=True,
            no_claim_of_immediate_physical_command_tracking=True,
            no_command_event_replanning=True,
            activation_ceiling="SIM_ONLY",
            trained_policy=False,
            physics_qualified=False,
            runtime_execution_authorized=False,
            training_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))
