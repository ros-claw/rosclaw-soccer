from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.aim_goal_command import AimGoalCommand, condition_aim_goal


def policy():
    return SimpleNamespace(
        runtime_mode="sim",
        use_body_frame_ball=False,
        control_dt=0.02,
        time_step=20,
        WARMUP_STEPS=0,
        target_pos_w=np.array([8.0, 0.0, 0.11], np.float32),
        previous_target_history=np.ones((5, 3)),
        measured_ball=np.array([1.0, 0.0, 0.11]),
        motion_joint_pos=np.ones((100, 29)),
    )


def test_command_changes_future_goal_only_not_measurements_references_or_history():
    p = policy()
    before = vars(p).copy()
    receipt = condition_aim_goal(p, AimGoalCommand((8.0, -0.5, 0.11)))
    assert receipt["applied"]
    np.testing.assert_array_equal(p.target_pos_w, np.array([8.0, -0.5, 0.11], np.float32))
    for key in ("previous_target_history", "measured_ball", "motion_joint_pos"):
        assert vars(p)[key] is before[key]
    assert not receipt["past_history_modified"] and not receipt["task_scoring_goal_modified"]


def test_zero_command_preserves_array_identity():
    p = policy()
    target = p.target_pos_w
    assert not condition_aim_goal(p, AimGoalCommand((8.0, 0.0, 0.11)))["applied"]
    assert p.target_pos_w is target


@pytest.mark.parametrize(
    "point", [(7.0, 0.0, 0.11), (8.0, 3.0, 0.11), (8.0, 0.0, -0.1), (8.0, float("nan"), 0.11)]
)
def test_out_of_domain_command_is_rejected_without_modification(point):
    p = policy()
    target = p.target_pos_w
    with pytest.raises(ValueError, match="domain"):
        condition_aim_goal(p, AimGoalCommand(point))
    assert p.target_pos_w is target


def test_real_policy_is_not_an_aim_command_path():
    p = policy()
    p.runtime_mode = "real"
    with pytest.raises(ValueError, match="SIM"):
        condition_aim_goal(p, AimGoalCommand((8.0, -0.5, 0.11)))


def test_nonzero_forecast_command_copies_private_goal_without_mutating_donor(tmp_path):
    from test_shared_nn_forecast import setup

    from rosclaw_soccer.rsi.shared_nn_forecast import policy_state_hash

    p, f = setup(tmp_path, budget=80)
    p.world.advance({p.prefix: p.pending})
    p.commit()
    p.policy.control_dt = 0.02
    before = policy_state_hash(p.policy)
    command = AimGoalCommand((8.0, -0.5, 0.11))
    result = f.forecast(controls=4, before_proposal=True, aim_goal=command)
    assert result["aim_command_conditioned"] and not result["reference_conditioned"]
    assert policy_state_hash(p.policy) == before
    np.testing.assert_array_equal(p.goal, [5.0, 0.0, np.float32(0.11)])
    condition_aim_goal(p.policy, command)
    p.goal = p.policy.target_pos_w.copy()
    records, targets = [], []
    for _ in range(4):
        target, _ = p.propose()
        targets.append(target)
        records.append(p.world.advance({p.prefix: target}))
        p.commit()
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(result[key], np.concatenate([r[key] for r in records]))
    np.testing.assert_array_equal(result["joint_targets"], np.stack(targets))
