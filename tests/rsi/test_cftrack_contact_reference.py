from dataclasses import replace

import numpy as np
import pytest
from test_shared_cftrack_forecast import setup

from rosclaw_soccer.rsi.cftrack_contact_reference import (
    CFtrackContactPlan,
    condition_cftrack_contact_reference,
)
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash


def fixture(tmp_path):
    p, f = setup(tmp_path)
    for side in ("left", "right"):
        setattr(
            p.policy,
            "ref_" + side + "_ankle_future_pos",
            np.tile(np.array([0, 0, 0.1], np.float32), (100, 1)),
        )
        setattr(
            p.policy,
            "ref_" + side + "_ankle_future_quat",
            np.tile(np.array([1, 0, 0, 0], np.float32), (100, 1)),
        )
    p.policy.ref_torso_future_pos = np.tile(np.array([0, 0, 1], np.float32), (100, 1))
    plan = CFtrackContactPlan(
        0,
        0,
        2,
        "right",
        (0.02, 0.0, 0.1),
        (float(np.cos(0.01)), 0.0, 0.0, float(np.sin(0.01))),
        cftrack_state_hash(p.policy),
    )
    return p, f, plan


def test_absolute_pose_reference_atomic_and_nonactive_history_past_unchanged(tmp_path):
    p, _, plan = fixture(tmp_path)
    q = p.policy.ref_right_ankle_future_pos.copy()
    other = p.policy.ref_left_ankle_future_pos.copy()
    history = p.policy.obs_history_buffer.copy()
    world = p.world.data.qpos.copy()
    receipt = condition_cftrack_contact_reference(p.policy, plan)
    assert receipt["endpoint_position_error_m"] < 1e-6
    assert receipt["endpoint_rotation_error_rad"] < 1e-5
    np.testing.assert_allclose(
        p.policy.ref_right_ankle_future_pos[2], plan.position_world, atol=1e-7
    )
    np.testing.assert_array_equal(p.policy.ref_right_ankle_future_pos[0], q[0])
    np.testing.assert_array_equal(p.policy.ref_right_ankle_future_pos[48:], q[48:])
    np.testing.assert_array_equal(p.policy.ref_left_ankle_future_pos, other)
    np.testing.assert_array_equal(p.policy.obs_history_buffer, history)
    np.testing.assert_array_equal(p.world.data.qpos, world)
    assert p.policy.counter_step == p.world.control_steps == 0


@pytest.mark.parametrize(
    "defect", ["hash", "position_speed", "angular_speed", "quaternion", "past"]
)
def test_invalid_plan_does_not_partially_change_refs(tmp_path, defect):
    p, _, plan = fixture(tmp_path)
    if defect == "hash":
        plan = replace(plan, donor_policy_hash="sha256:" + "0" * 64)
    elif defect == "position_speed":
        plan = replace(plan, position_world=(0.3, 0.0, 0.1))
    elif defect == "angular_speed":
        plan = replace(plan, quaternion_world=(float(np.cos(0.5)), 0.0, 0.0, float(np.sin(0.5))))
    elif defect == "quaternion":
        plan = replace(plan, quaternion_world=(2.0, 0.0, 0.0, 0.0))
    else:
        plan = replace(plan, origin_counter=1)
    before = cftrack_state_hash(p.policy)
    with pytest.raises(ValueError):
        condition_cftrack_contact_reference(p.policy, plan)
    assert cftrack_state_hash(p.policy) == before


def test_double_cover_identity_does_not_introduce_quaternion_spin(tmp_path):
    p, _, plan = fixture(tmp_path)
    plan = replace(plan, position_world=(0.0, 0.0, 0.1), quaternion_world=(-1.0, 0.0, 0.0, 0.0))
    old = p.policy.ref_right_ankle_future_quat.copy()
    receipt = condition_cftrack_contact_reference(p.policy, plan)
    np.testing.assert_array_equal(p.policy.ref_right_ankle_future_quat, old)
    assert receipt["maximum_sampled_additional_reference_angular_speed_rad_s"] < 1e-6


@pytest.mark.parametrize("bounded", [False, True])
def test_conditioned_private_future_exact_native_actual_and_donor_unmodified(tmp_path, bounded):
    p, f, plan = fixture(tmp_path)
    plan = replace(plan, bounded_position_path=bounded)
    before = cftrack_state_hash(p.policy)
    predicted = f.forecast(controls=4, contact_plan=plan)
    assert cftrack_state_hash(p.policy) == before
    receipt = condition_cftrack_contact_reference(p.policy, plan)
    assert receipt == predicted["reference_conditioning"]
    for i in range(4):
        target, _ = p.propose()
        np.testing.assert_array_equal(target, predicted["joint_targets"][i])
        row = p.world.advance({p.prefix: target})
        p.commit()
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            np.testing.assert_array_equal(row[key], predicted[key][i * 10 : (i + 1) * 10])


def test_bounded_path_still_rejects_impossible_velocity_atomically(tmp_path):
    p, _, plan = fixture(tmp_path)
    plan = replace(plan, bounded_position_path=True, position_world=(0.5, 0.0, 0.1))
    before = cftrack_state_hash(p.policy)
    with pytest.raises(ValueError):
        condition_cftrack_contact_reference(p.policy, plan)
    assert cftrack_state_hash(p.policy) == before
