"""Synthetic contact state/actual toy integration, not genuine G1 contact proof."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.born_navigation_recovery import BornNavigationRecovery
from rosclaw_soccer.rsi.causal_impact_replay import CausalImpactReplay
from rosclaw_soccer.rsi.full_task_branch import FullTaskBranch, integration_hash
from tests.rsi.test_full_task_branch import (  # noqa: F401
    build,
    candidate,
    current,
    imitation_parent,
    model,
    smooth_parent,
)


@pytest.mark.parametrize("name,cap", [("nav_brake_0_7", 0.7), ("nav_brake_0", 0.0)])
def test_only_existing_forward_request_is_lowered(name, cap):
    option = BornNavigationRecovery(186, name)
    assert option.apply(
        (1.4, 0.1, 0.0), (1.4, 0.1, 0.0), frame=186, after_measured_contact=True
    ) == ((cap, 0.1, 0.0), (cap, 0.1, 0.0))
    assert (
        option.apply((0.2, -0.1, 0.0), (0.2, -0.1, 0.0), frame=299, after_measured_contact=True)[1][
            0
        ]
        <= 0.2
    )
    assert option.contract()["hardware_authorized"] is False
    for frame, measured in [(185, True), (300, True), (186, False)]:
        with pytest.raises(ValueError):
            option.apply(
                (1.4, 0.0, 0.0), (1.4, 0.0, 0.0), frame=frame, after_measured_contact=measured
            )


@pytest.mark.parametrize(
    "start,name",
    [(30, "nav_brake_0"), (300, "nav_brake_0"), (True, "nav_brake_0"), (31, "torque_override")],
)
def test_unknown_action_or_forged_clock_rejected(start, name):
    with pytest.raises(ValueError):
        BornNavigationRecovery(start, name)


@pytest.mark.parametrize(
    "constant,recovery", [(False, False), (True, False), (False, True), (True, True)]
)
def test_review_does_not_confuse_precontact_law_with_postcontact_lease(constant, recovery):
    # Metadata-only fixture, not proof that these counter values were executed.
    actor = object.__new__(CausalImpactReplay)
    actor.failed = False
    actor.next_frame = 300
    actor.integrated_steps = actor.attempted_steps = 3000
    actor.teacher = actor.release_control = None
    actor.initial_reference = SimpleNamespace(contract=lambda: {})
    actor.constant_forward_approach = constant
    actor.navigation_recovery = BornNavigationRecovery(186, "nav_brake_0") if recovery else None
    actor.postcontact_exit_enabled = False
    review = actor.review()
    assert ("original_precontact_forward_law_unchanged" in review) is constant
    assert ("born_navigation_recovery" in review) is recovery
    if constant:
        assert review["original_precontact_forward_law_unchanged"] is False


def test_nonzero_owned_prediction_matches_actual_same_born_lease(monkeypatch, request):
    parent, wh = build(monkeypatch, request.getfixturevalue("imitation_parent"))
    for f in range(31):
        parent.step(f)
    lease = BornNavigationRecovery(31, "nav_brake_0_7")
    with pytest.raises(ValueError, match="measured-foot"):
        parent.install_navigation_recovery(lease)
    # Explicitly synthetic measured-contact fixture. Do not count this as a
    # successful actual foot contact, M0 exam or official model qualification.
    parent.ik.contact = 30
    parent.previous_forces[0] = 2.0
    donor = integration_hash(parent.model, parent.data)
    predicted = FullTaskBranch(parent, world_hash=wh, action_name="nav_brake_0_7")
    assert parent.navigation_recovery is None
    assert integration_hash(parent.model, parent.data) == donor
    frames = [predicted.step() for _ in range(269)]
    review = predicted.review()
    assert review["action_name"] == "nav_brake_0_7"
    assert review["born_navigation_recovery"] == lease.contract()
    assert review["new_native_integrated_steps"] == 2690
    assert review["new_planner_onnx_calls"] == 13
    parent.install_navigation_recovery(lease)
    with pytest.raises(ValueError, match="measured-foot"):
        parent.install_navigation_recovery(lease)
    for f, expected in enumerate(frames, 31):
        actual = parent.step(f)
        assert actual.arrays.keys() == expected.arrays.keys()
        for k in actual.arrays:
            assert actual.arrays[k].tobytes() == expected.arrays[k].tobytes(), k
        assert actual.arrays["ik_planner_pulse_baseline_command"][0, 0] == 0.7
    assert parent.review()["born_navigation_recovery"] == lease.contract()
    np.testing.assert_array_equal(parent.torque_limits, predicted._owned.torque_limits)
