import pytest

from rosclaw_soccer.training.sim_contact_impulse_probe import SimContactImpulseProbe


def test_bounded_probe_is_immutable_and_sim_only():
    p = SimContactImpulseProbe("red.defender", 177, "left_ankle_pitch_joint", 5.0)
    assert p.activation_ceiling == "SIM_ONLY"
    assert not p.hardware_authorized
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
