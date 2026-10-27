import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_motion_effect import paired_receiver_motion_effect


def trace(n=3):
    context = np.zeros((n, 14), np.float32)
    context[:, 0] = [2, 1, 0.5][:n]
    context[:, 3] = 0.75
    flags = np.zeros((n, 2), np.bool_)
    flags[-1, 0] = True
    context[:, 12:14] = flags
    return dict(
        control=np.arange(n),
        context14=context,
        action=np.zeros((n, 29), np.float32),
        realized_delta_radians=np.zeros((n, 29), np.float64),
        contact_flags_before=flags,
    )


def test_paired_measurement_reports_causal_limit_and_common_length():
    a, b = trace(), trace()
    b["action"][1, 0] = 0.3
    b["realized_delta_radians"][1, 0] = 0.06
    b["context14"][1, 0] = 1.5
    result = paired_receiver_motion_effect(a, b)
    assert result["baseline"]["min_root_ball_distance_3d_m"] == pytest.approx(0.5)
    assert result["candidate"]["first_preaction_past_contact_control"] == 2
    assert result["max_executed_action_difference"] == pytest.approx(0.3)
    assert result["candidate"]["max_realized_joint_delta_rad"] == pytest.approx(0.06)
    assert result["causal_attribution_proven"] is False


def test_inconsistent_contact_history_fails_closed():
    a, b = trace(), trace()
    b["contact_flags_before"][0, 0] = True
    with pytest.raises(ValueError, match="causal persistent"):
        paired_receiver_motion_effect(a, b)
