import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.factual_response_pair import extract_pair, wrench_wish
from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS, PREFIX
from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES
from tests.rsi.test_foot_heading_parity import baseline


def pair():
    b = baseline()
    b.update({PREFIX + k: np.zeros((300, 1, width)) for k, width in FIELDS.items()})
    b["force_n"][72, 0, 0] = 2
    a = copy.deepcopy(b)
    a["postcontact_release_requested_offset_nm"] = np.zeros((300, 1, 29))
    a["postcontact_release_requested_offset_nm"][73:76, 0, 0] = 1
    a["postcontact_release_applied_offset_nm"] = np.zeros((300, 1, 10, 29))
    a["postcontact_release_forecast_steps"] = np.zeros((300, 1, 1), dtype=np.int64)
    a["ball_linear_velocity_before_step_m_s"][77, 0] = [1, 2, 3]
    return b, a


def test_complete_original_prefix_and_actual_future_labels_only():
    b, a = pair()
    result = extract_pair(b, a, name="forward_plus", trigger_frame=73)
    assert result["status"] == "FACTUAL_PAIR"
    np.testing.assert_array_equal(result["action_delta_velocity"], [1, 2, 3])
    np.testing.assert_array_equal(result["baseline_delta_velocity"], [0, 0, 0])
    assert result["action_features"].shape == (17,)
    assert result["source_and_physics_authenticated_here"] is False
    original_features = result["action_features"].copy()
    a["ball_linear_velocity_before_step_m_s"][77, 0] *= 2
    changed = extract_pair(b, a, name="forward_plus", trigger_frame=73)
    np.testing.assert_array_equal(changed["action_features"], original_features)
    np.testing.assert_array_equal(changed["action_delta_velocity"], [2, 4, 6])


def test_every_original_named_wish_is_bounded():
    wishes = [wrench_wish(name) for name in ACTION_NAMES]
    assert len({x.tobytes() for x in wishes}) == 13
    assert all(x.shape == (4,) and np.max(abs(x)) <= 1 for x in wishes)


@pytest.mark.parametrize("frame", [None, 299, 30])
def test_missing_or_censored_tail_is_not_fabricated_training_zero(frame):
    b, a = pair()
    result = extract_pair(b, a, name="forward_plus", trigger_frame=frame)
    assert result["status"] == "SKIPPED" and "action_delta_velocity" not in result


def test_zero_alias_is_not_nonzero_supervision():
    b, a = pair()
    a["postcontact_release_requested_offset_nm"].fill(0)
    assert extract_pair(b, a, name="forward_plus", trigger_frame=73)["status"] == "SKIPPED"


def test_prefix_or_current_state_mismatch_refused():
    b, a = pair()
    a["recurrent_residual_hidden_state"][50, 0, 0] = 1
    with pytest.raises(ValueError, match="prefix"):
        extract_pair(b, a, name="forward_plus", trigger_frame=73)
    b, a = pair()
    a["joint_position_rad"][73, 0, 4] = 1
    with pytest.raises(ValueError, match="current before-action state"):
        extract_pair(b, a, name="forward_plus", trigger_frame=73)
