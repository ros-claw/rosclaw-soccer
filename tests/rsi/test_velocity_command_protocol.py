import pytest

from rosclaw_soccer.rsi.velocity_command_protocol import (
    ball_aware_checkpoint_protocol,
    command_protocol,
    require_same_command_protocol,
)


@pytest.mark.parametrize("point", ["torso", "pelvis"])
@pytest.mark.parametrize("gain", [0.0, 1.0])
def test_protocol_bound_to_point_and_gain(point, gain):
    report = command_protocol(point, gain)
    require_same_command_protocol(report, point, gain)
    with pytest.raises(ValueError, match="same command protocol"):
        require_same_command_protocol(report, point, 1 - gain)
    with pytest.raises(ValueError, match="same command protocol"):
        require_same_command_protocol(report, "pelvis" if point == "torso" else "torso", gain)


def test_legacy_qualification_is_not_pelvis_qualification():
    require_same_command_protocol({}, "torso", 0)
    with pytest.raises(ValueError, match="same command protocol"):
        require_same_command_protocol({}, "pelvis", 1)


@pytest.mark.parametrize(
    "point,gain",
    [
        (None, 0),
        ("root", 0),
        ("pelvis", True),
        ("pelvis", 0.5),
        ("torso", float("nan")),
        ("torso", float("inf")),
    ],
)
def test_invalid_contract_rejected(point, gain):
    with pytest.raises(ValueError):
        command_protocol(point, gain)


def test_first_ball_aware_report_omission_uses_bound_checkpoint_not_legacy_defaults():
    from rosclaw_soccer.rsi.ball_aware_inputs import BALL_AWARE_ABI

    report = dict(ball_aware_abi=BALL_AWARE_ABI, actor_observation_dim=177)
    checkpoint = dict(report, **command_protocol("pelvis", 1.0))
    assert ball_aware_checkpoint_protocol(report, checkpoint) == command_protocol("pelvis", 1.0)
    for key, value in command_protocol("torso", 0.0).items():
        with pytest.raises(ValueError, match="mismatch"):
            ball_aware_checkpoint_protocol(dict(report, **{key: value}), checkpoint)
    with pytest.raises(ValueError):
        ball_aware_checkpoint_protocol(report, report)
    with pytest.raises(ValueError):
        ball_aware_checkpoint_protocol(dict(report, actor_observation_dim=168), checkpoint)
