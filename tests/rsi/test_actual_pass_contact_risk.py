import numpy as np
import pytest

from rosclaw_soccer.rsi.actual_pass_contact_risk import contact_risk_data


def test_risk_uses_before_state_and_groups_shared_contexts():
    n = 32
    data = dict(
        qpos_before=np.zeros((n, 79)),
        qvel_before=np.zeros((n, 76)),
        receiver_observation=np.zeros((n, 98)),
        receiver_action_applied=np.zeros((n, 29)),
        ball_force_max_after=np.zeros((n, 2, 3)),
        course_index=np.arange(n),
    )
    data["ball_force_max_after"][7, 1, 2] = 2.0
    courses = [dict(index=i, course=dict(context=(1, 4, 20, 21)[i % 4])) for i in range(32)]
    x, y, groups = contact_risk_data(data, courses)
    assert x.shape == (32, 282)
    assert y.sum() == 1 and y[7] == 1
    assert groups[7] == groups[3] == 21
    del data["qvel_before"]
    with pytest.raises(ValueError, match="complete causal"):
        contact_risk_data(data, courses)
