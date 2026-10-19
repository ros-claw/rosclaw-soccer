"""Values-only proposal; no native acceptance or physical improvement claim."""

from dataclasses import FrozenInstanceError

import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.rsi.planner_navigation_pulse import PlannerNavigationPulse
from rosclaw_soccer.rsi.planner_navigation_pulse_evidence import require_navigation_pulse_request
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_planner_navigation_pulse_evidence import zero_model


@pytest.mark.parametrize("delta", [0.0, 0.08])
def test_reuses_original_numeric_composition_but_not_zero_reference_intent(delta):
    pulse = OriginalIkPlannerPulse(delta)
    original = PlannerNavigationPulse(delta)
    baseline = (1.4, -0.0, 0.0)
    for frame in range(300):
        assert pulse.compose(frame, baseline) == original.compose(frame, baseline)
        if delta == 0.0 or not 100 <= frame < 160:
            assert pulse.compose(frame, baseline)[1] is baseline
    c = pulse.contract()
    assert c["contract_hash"] == hash_json({k: v for k, v in c.items() if k != "contract_hash"})
    assert c["schema"] != original.contract()["schema"]
    assert c["leg_reference_kind"] == "ORIGINAL_BOUNDED_IK"
    assert c["original_parent_required"] and c["original_bounded_ik_path_required"]
    assert c["learned_reference_model_forbidden"] and c["zero_reference_model_forbidden"]
    assert c["source_hash"] != c["composition_source_hash"]
    assert c["native_request_and_independent_qualification_required"]
    for key in (
        "trained_policy",
        "physics_qualified",
        "runtime_execution_authorized",
        "training_authorized",
        "hardware_authorized",
        "promotion_authorized",
    ):
        assert c[key] is False


@pytest.mark.parametrize("model", [None, "zero"])
def test_old_zero_reference_native_entry_cannot_implicitly_enable_original_ik(model):
    with pytest.raises(ValueError):
        require_navigation_pulse_request(
            OriginalIkPlannerPulse(0.08),
            zero_model() if model == "zero" else None,
            profile=BoundedFirstTouchControl(),
            record_planner_events=True,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lateral_delta_m_s=-0.08),
        dict(lateral_delta_m_s=0.081),
        dict(lateral_delta_m_s=True),
        dict(start_frame=80),
        dict(end_frame=240),
    ],
)
def test_proposal_does_not_open_arbitrary_parameter_or_timing_search(kwargs):
    with pytest.raises(ValueError):
        OriginalIkPlannerPulse(**kwargs)


def test_prototype_is_frozen_and_untrusted_subclasses_are_not_executable_requests():
    pulse = OriginalIkPlannerPulse(0.08)
    with pytest.raises(FrozenInstanceError):
        pulse.lateral_delta_m_s = 0.1

    class Subclass(OriginalIkPlannerPulse):
        pass

    with pytest.raises(ValueError):
        Subclass(0.08)
