import numpy as np
import pytest
from test_receiving_rollout import contact, trace

from rosclaw_soccer.training.in_play_substep_capture_audit import audit_in_play_substep_capture

IDS = ("blue.other", "red.receiver")


def fixture():
    value = trace()
    contact(value)
    n = len(value["time"])
    value["pitch_boundary_geometry"] = np.tile([-1.5, 7.5, 3.0, 0.115, 3.0, 2.0], (n, 1))
    value["contact_normal_substeps"] = np.zeros((n, len(IDS)), dtype=int)
    value["contact_normal_substeps"][20, 1] = 1
    value["contact_nonfoot_substeps"] = np.zeros(n, dtype=int)
    value["contact_normal_world_safe"] = np.ones(n, dtype=bool)
    return value


def audit(value):
    return audit_in_play_substep_capture(
        value, agent_ids=IDS, agent_id="red.receiver", start=1, frames=100
    )


def test_sampled_in_play_does_not_authorize_training_or_promotion():
    result = audit(fixture())
    assert result["sampled_in_play_capture_without_recorded_substep_veto"]
    assert not result["continuous_boundary_verified"]
    assert not result["team_intent_verified"]
    assert not result["training_authorized"]
    assert not result["promotion_eligible"]


@pytest.mark.parametrize("frame", [0, 10, 20, 50, 100])
def test_exit_before_or_within_window_cannot_be_rescued_by_reentry(frame):
    value = fixture()
    value["ball_pose"][frame, 1] = 3.2
    result = audit(value)
    assert result["boundary"]["first_exit_frame"] == frame
    assert not result["sampled_in_play_through_window"]
    assert not result["sampled_in_play_capture_without_recorded_substep_veto"]


def test_exit_after_window_is_context_not_retroactive_veto():
    value = fixture()
    value = {key: np.concatenate([v, v[-1:]]) for key, v in value.items()}
    value["time"][-1] += 0.02
    value["ball_pose"][-1, 1] = 3.2
    result = audit(value)
    assert result["boundary"]["first_exit_frame"] == 101
    assert result["sampled_in_play_through_window"]


@pytest.mark.parametrize("missing", ["pitch_boundary_geometry", "contact_nonfoot_substeps"])
def test_missing_evidence_cannot_be_a_passing_result(missing):
    value = fixture()
    del value[missing]
    with pytest.raises((ValueError, KeyError)):
        audit(value)


def test_substep_nonfoot_still_vetoes_an_in_play_sampled_capture():
    value = fixture()
    value["contact_nonfoot_substeps"][21] = 1
    result = audit(value)
    assert result["sampled_in_play_through_window"]
    assert result["capture"]["substep_status"] == "NONFOOT_CONTACT"
    assert not result["sampled_in_play_capture_without_recorded_substep_veto"]


def test_clean_in_play_evidence_cannot_upgrade_a_missing_touch():
    value = fixture()
    value["ball_contact_agent_code"][:] = 0
    value["ball_contact_effector_code"][:] = 0
    value["ball_contact_force_n"][:] = 0
    result = audit(value)
    assert result["sampled_in_play_through_window"]
    assert result["capture"]["substep_status"] == "NO_SAMPLED_TOUCH"
    assert not result["sampled_in_play_capture_without_recorded_substep_veto"]


def test_boundary_schema_drift_fails_closed():
    value = fixture()
    value["pitch_boundary_geometry"][42, 3] = 0.1
    with pytest.raises(ValueError, match="constant match boundary"):
        audit(value)
