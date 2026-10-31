from dataclasses import replace

import pytest

from rosclaw_soccer.skills.team.motor_option import (
    TeamActuatorEvidence,
    TeamMotorPhysicsObservation,
)
from rosclaw_soccer.skills.team.physics_evidence import PhysicsEvidenceSlot
from rosclaw_soccer.training.dense_team_contact_recorder import DenseTeamContactRecorder


def row(time=0.002):
    return TeamMotorPhysicsObservation(
        time,
        (0.0,) * 43,
        (0.0,) * 41,
        True,
        0.0,
        0.0,
        observer_agent_id="red.finisher",
        contacts_complete=True,
    )


def test_complete_stream_is_observation_only_and_exported():
    recorder = DenseTeamContactRecorder("red.finisher", maximum_substeps=2)
    slot = PhysicsEvidenceSlot(recorder)
    slot.observe_physics(row())
    slot.observe_physics(row(0.004))
    assert not slot.faulted
    assert not hasattr(recorder, "propose")
    assert recorder.observations() == (row(), row(0.004))
    assert recorder.to_dict()["steps"] == 2
    assert recorder.to_dict()["promotion_authorized"] is False
    assert "actuator_evidence" not in recorder.to_dict()["rows"][0]


def test_opt_in_actuator_evidence_is_player_bound_and_complete():
    recorder = DenseTeamContactRecorder("red.finisher", actuator_evidence=True)
    slot = PhysicsEvidenceSlot(recorder)
    actuator = TeamActuatorEvidence((1.0,) * 29, (2.0,) * 29, (3.0,) * 29)
    slot.observe_physics(replace(row(), actuator_evidence=actuator))
    assert not slot.faulted
    exported = recorder.to_dict()["rows"][0]["actuator_evidence"]
    assert exported["control"] == (1.0,) * 29
    assert exported["actuator_force"] == (2.0,) * 29
    assert exported["generalized_force"] == (3.0,) * 29
    assert recorder.contract_hash != DenseTeamContactRecorder("red.finisher").contract_hash


def test_actuator_subscription_cannot_change_or_be_missing():
    recorder = DenseTeamContactRecorder("red.finisher", actuator_evidence=True)
    slot = PhysicsEvidenceSlot(recorder)
    slot.observe_physics(row())
    assert slot.faulted
    assert not recorder.observations()
    recorder = DenseTeamContactRecorder("red.finisher", actuator_evidence=True)
    slot = PhysicsEvidenceSlot(recorder)
    recorder.needs_actuator_evidence = False
    slot.observe_physics(row())
    assert slot.faulted


@pytest.mark.parametrize("bad", [(0.0,) * 28, (float("nan"),) * 29])
def test_reject_invalid_actuator_evidence(bad):
    with pytest.raises(ValueError, match="simulated actuator evidence"):
        TeamActuatorEvidence(bad, (0.0,) * 29, (0.0,) * 29)


@pytest.mark.parametrize(
    "bad",
    [
        row(0.004),
        replace(row(), contacts_complete=False, observer_agent_id=None),
        replace(row(), observer_agent_id="blue.finisher"),
    ],
)
def test_missing_or_foreign_observation_rejected(bad):
    recorder = DenseTeamContactRecorder("red.finisher")
    with pytest.raises(ValueError):
        recorder.observe_physics(bad)
    assert not recorder.observations()


def test_capacity_fail_closed():
    recorder = DenseTeamContactRecorder("red.finisher", maximum_substeps=1)
    recorder.observe_physics(row())
    with pytest.raises(ValueError):
        recorder.observe_physics(row(0.004))
    assert len(recorder.observations()) == 1


def test_early_terminal_frame_is_complete_without_padding():
    recorder = DenseTeamContactRecorder("red.finisher", maximum_substeps=50)
    for index in range(20):
        recorder.observe_physics(row((index + 1) * 0.002))
    result = recorder.verified_observations(control_frame_count=2, terminal_time_sec=0.04)
    assert len(result) == 20
    assert result[-1].time_sec == pytest.approx(0.04)
    # A caller cannot claim the requested five frames were recorded.
    with pytest.raises(ValueError, match="authenticated control-frame"):
        recorder.verified_observations(control_frame_count=5, terminal_time_sec=0.10)


@pytest.mark.parametrize(
    "frames,terminal", [(1, 0.03), (2, 0.02), (0, 0.0), (True, 0.02), (1, float("nan"))]
)
def test_bad_terminal_identity_is_rejected(frames, terminal):
    recorder = DenseTeamContactRecorder("red.finisher", maximum_substeps=20)
    for index in range(10):
        recorder.observe_physics(row((index + 1) * 0.002))
    with pytest.raises(ValueError, match="authenticated control-frame"):
        recorder.verified_observations(control_frame_count=frames, terminal_time_sec=terminal)


def test_incomplete_terminal_frame_is_rejected():
    recorder = DenseTeamContactRecorder("red.finisher", maximum_substeps=20)
    for index in range(19):
        recorder.observe_physics(row((index + 1) * 0.002))
    with pytest.raises(ValueError, match="authenticated control-frame"):
        recorder.verified_observations(control_frame_count=2, terminal_time_sec=0.04)
