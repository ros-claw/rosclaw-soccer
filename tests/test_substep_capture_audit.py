import numpy as np
import pytest
from test_receiving_rollout import contact, trace

from rosclaw_soccer.training.substep_capture_audit import audit_substep_capture

IDS = ("blue.other", "red.receiver")


def setup():
    value = trace()
    contact(value)
    value["contact_normal_substeps"] = np.zeros((101, 2), dtype=int)
    value["contact_normal_substeps"][20, 1] = 1
    value["contact_nonfoot_substeps"] = np.zeros(101, dtype=int)
    value["contact_normal_world_safe"] = np.ones(101, dtype=bool)
    return value


def assess(value):
    return audit_substep_capture(value, agent_ids=IDS, agent_id=IDS[1], start=1, frames=100)


def test_clean_sampled_success_never_promotes():
    result = assess(setup())
    assert result["sampled"]["controlled_reception"]
    assert result["capture_without_recorded_substep_veto"]
    assert not result["promotion_eligible"]


@pytest.mark.parametrize("kind", ["body", "other_foot", "unsafe"])
def test_unsampled_substep_interruptions_veto_sampled_success(kind):
    value = setup()
    if kind == "body":
        value["contact_nonfoot_substeps"][25] = 1
    elif kind == "other_foot":
        value["contact_normal_substeps"][25, 0] = 1
    else:
        value["contact_normal_world_safe"][25] = False
    result = assess(value)
    assert result["sampled"]["controlled_reception"]
    assert not result["capture_without_recorded_substep_veto"]
    assert result["substep_status"] != "NO_SUBSTEP_VETO"


def test_mixed_first_bin_is_conservatively_rejected():
    value = setup()
    value["contact_nonfoot_substeps"][20] = 1
    assert assess(value)["substep_status"] == "NONFOOT_CONTACT"


def test_prior_interruption_does_not_contaminate_later_capture():
    value = setup()
    value["contact_nonfoot_substeps"][19] = 1
    assert assess(value)["capture_without_recorded_substep_veto"]


def test_incomplete_or_mismatched_substeps_fail_closed():
    value = setup()
    value["contact_normal_substeps"][20, 1] = 0
    with pytest.raises(ValueError, match="inconsistent"):
        assess(value)
    value = setup()
    del value["contact_nonfoot_substeps"]
    with pytest.raises(KeyError):
        assess(value)
    value = setup()
    value["contact_nonfoot_substeps"][25] = 11
    with pytest.raises(ValueError, match="coverage"):
        assess(value)


def test_no_sampled_touch_cannot_be_upgraded():
    value = setup()
    value["ball_contact_force_n"][20] = 0
    result = assess(value)
    assert result["substep_status"] == "NO_SAMPLED_TOUCH"
    assert not result["capture_without_recorded_substep_veto"]
