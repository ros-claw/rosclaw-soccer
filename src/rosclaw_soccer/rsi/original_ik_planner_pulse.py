"""Values-only prototype for a separately qualified approach + foot-IK study.

The existing zero-reference native interface intentionally rejects this type.
Integrating it requires a new explicit request, scalar review, CPU/foundation
qualification and original-teacher parity; this contract does not supply those.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi import planner_navigation_pulse as composition
from rosclaw_soccer.rsi.planner_navigation_pulse import PlannerNavigationPulse
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True, slots=True)
class OriginalIkPlannerPulse(PlannerNavigationPulse):
    """Same projected navigation arithmetic, explicitly different leg intent."""

    def __post_init__(self) -> None:
        PlannerNavigationPulse.__post_init__(self)
        if type(self) is not OriginalIkPlannerPulse or self.lateral_delta_m_s not in (0.0, 0.08):
            raise ValueError("exact original-IK zero or positive fixed pulse prototype required")

    def contract(self) -> dict[str, Any]:
        value = PlannerNavigationPulse.contract(self)
        value.pop("contract_hash")
        value.pop("zero_leg_reference_and_original_parent_required")
        value.update(
            schema="soccer.rsi.original_bounded_ik_pre_planner_pulse.v1",
            source_hash=hash_bytes(Path(__file__).read_bytes()),
            composition_source_hash=hash_bytes(Path(composition.__file__).read_bytes()),
            leg_reference_kind="ORIGINAL_BOUNDED_IK",
            original_parent_required=True,
            original_bounded_ik_path_required=True,
            learned_reference_model_forbidden=True,
            zero_reference_model_forbidden=True,
            native_request_and_independent_qualification_required=True,
            original_teacher_off_all300_arrays_required=True,
            active_teacher_first100_arrays_required=True,
            navigation_gain_does_not_qualify_contact_direction=True,
        )
        return dict(value, contract_hash=hash_json(value))
