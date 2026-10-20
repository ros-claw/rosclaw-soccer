import pytest

from rosclaw_soccer.rsi.velocity_command_protocol import (
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
