import pytest

from rosclaw_soccer.training.sim_contact_impulse_probe import (
    SimContactImpulseProbe,
    guard_refusal_masks,
)


@pytest.mark.parametrize("step", range(10))
@pytest.mark.parametrize("projector_ok", [False, True])
@pytest.mark.parametrize("torque_ok", [False, True])
@pytest.mark.parametrize("actuator_ok", [False, True])
def test_refusal_masks_encode_each_guard_independently(step, projector_ok, torque_ok, actuator_ok):
    mask = 1 << step
    assert guard_refusal_masks(
        step, projector_ok=projector_ok, torque_ok=torque_ok, actuator_ok=actuator_ok
    ) == tuple(0 if ok else mask for ok in (projector_ok, torque_ok, actuator_ok))


@pytest.mark.parametrize("step", [-1, 10, 1.0, True])
def test_refusal_masks_reject_out_of_range_or_invalid_step(step):
    with pytest.raises(ValueError):
        guard_refusal_masks(step, projector_ok=True, torque_ok=True, actuator_ok=True)


def test_refusal_masks_reject_non_boolean_guard_results():
    with pytest.raises(ValueError):
        guard_refusal_masks(0, projector_ok=1, torque_ok=True, actuator_ok=True)


def test_bounded_probe_is_immutable_and_sim_only():
    p = SimContactImpulseProbe("red.defender", 177, "left_ankle_pitch_joint", 5.0)
    assert p.activation_ceiling == "SIM_ONLY"
    assert not p.hardware_authorized
    assert p.duration_substeps == 1
    with pytest.raises(AttributeError):
        p.frame = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("frame", "joint", "delta", "ceiling", "hardware"),
    [
        (-1, "left_ankle_pitch_joint", 5.0, "SIM_ONLY", False),
        (500, "left_ankle_pitch_joint", 5.0, "SIM_ONLY", False),
        (0, "waist_yaw_joint", 5.0, "SIM_ONLY", False),
        (0, "left_ankle_pitch_joint", 5.1, "SIM_ONLY", False),
        (0, "left_ankle_pitch_joint", 5.0, "REAL", False),
        (0, "left_ankle_pitch_joint", 5.0, "SIM_ONLY", True),
    ],
)
def test_rejects_unbounded_or_non_sim_probe(frame, joint, delta, ceiling, hardware):
    with pytest.raises(ValueError):
        SimContactImpulseProbe("red.defender", frame, joint, delta, ceiling, hardware)


@pytest.mark.parametrize("duration", [0, 2, 11, 100, True, 1.0])
def test_rejects_invalid_duration(duration):
    with pytest.raises(ValueError):
        SimContactImpulseProbe(
            "red.defender", 177, "left_ankle_pitch_joint", 5.0, duration_substeps=duration
        )


@pytest.mark.parametrize("duration", [1, 5, 10])
def test_allows_only_bounded_substep_durations(duration):
    assert (
        SimContactImpulseProbe(
            "red.defender", 177, "left_ankle_pitch_joint", 5.0, duration_substeps=duration
        ).duration_substeps
        == duration
    )
