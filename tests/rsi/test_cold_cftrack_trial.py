import numpy as np
import pytest
from test_shared_cftrack_forecast import setup

from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash


def test_new_episodes_have_private_body_history_and_fk_but_do_not_reset_donor(tmp_path):
    template, _ = setup(tmp_path, physical_range=1.0, cold=True)
    before = cftrack_state_hash(template.policy)
    q = template.world.data.qpos.copy()
    left = cold_cftrack_trial(template, ball_xy=(1.0, -0.02))
    right = cold_cftrack_trial(template, ball_xy=(1.0, 0.02))
    assert left.world.model is right.world.model is template.world.model
    assert left.world.data is not right.world.data
    assert left.policy.ort_session is template.policy.ort_session
    assert left.policy.kinematics.data is not template.policy.kinematics.data
    assert left.policy.obs_history_buffer is not right.policy.obs_history_buffer
    np.testing.assert_array_equal(
        left.world.data.qpos[left.layout.bq : left.layout.bq + 2], [1.0, -0.02]
    )
    target, _ = left.propose()
    left.world.advance({left.prefix: target})
    left.commit()
    assert right.world.native_steps == template.world.native_steps == 0
    np.testing.assert_array_equal(template.world.data.qpos, q)
    assert cftrack_state_hash(template.policy) == before


@pytest.mark.parametrize(
    "ball", [(0.89, 0.0), (1.0, 0.11), (1.0, float("nan")), [1.0, 0.0], (True, 0.0)]
)
def test_undeclared_initial_ball_rejected_before_new_world(tmp_path, ball):
    template, _ = setup(tmp_path, cold=True)
    with pytest.raises(ValueError):
        cold_cftrack_trial(template, ball_xy=ball)
    assert template.world.native_steps == 0


def test_executed_template_cannot_be_reset_or_reused_for_cold_trials(tmp_path):
    template, _ = setup(tmp_path, cold=True)
    target, _ = template.propose()
    template.world.advance({template.prefix: target})
    template.commit()
    q = template.world.data.qpos.copy()
    with pytest.raises(ValueError, match="never-executed"):
        cold_cftrack_trial(template, ball_xy=(1.0, 0.0))
    assert template.world.native_steps == 10
    np.testing.assert_array_equal(template.world.data.qpos, q)
