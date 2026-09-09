from dataclasses import replace

import pytest

from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorReadiness,
    motor_ball_action_ready,
)


class Provider:
    def __init__(self, value):
        self.value = value

    def readiness(self, *, frame, time_sec):
        return self.value


@pytest.mark.parametrize("ready", [False, True])
def test_current_readiness_is_only_a_commitment_gate(ready):
    status = TeamMotorReadiness("blue.playmaker", 410, 8.2, ready)
    assert (
        motor_ball_action_ready(
            Provider(status), agent_id="blue.playmaker", frame=410, time_sec=8.2
        )
        is ready
    )


@pytest.mark.parametrize(
    "changes",
    [{"agent_id": "red.playmaker"}, {"frame": 409}, {"time_sec": 8.0}],
)
def test_foreign_or_stale_readiness_cannot_release_a_commitment(changes):
    status = replace(TeamMotorReadiness("blue.playmaker", 410, 8.2, True), **changes)
    with pytest.raises(ValueError, match="another player, frame or clock"):
        motor_ball_action_ready(
            Provider(status), agent_id="blue.playmaker", frame=410, time_sec=8.2
        )


@pytest.mark.parametrize("value", [None, True, {"ball_action_ready": True}])
def test_untyped_readiness_is_rejected(value):
    with pytest.raises(ValueError):
        motor_ball_action_ready(Provider(value), agent_id="blue.playmaker", frame=410, time_sec=8.2)


@pytest.mark.parametrize(
    "field,value",
    [("frame", True), ("time_sec", float("nan")), ("ball_action_ready", 1)],
)
def test_readiness_fields_are_finite_and_strictly_typed(field, value):
    with pytest.raises(ValueError):
        replace(TeamMotorReadiness("blue.playmaker", 410, 8.2, True), **{field: value})
