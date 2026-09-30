import copy

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi import contact_motor_strike as strike
from rosclaw_soccer.rsi.contact_motor_evidence import audit_motor_arrays
from rosclaw_soccer.rsi.contact_motor_primitive import JOINT_NAMES, make_policy, motor_delta
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES
from rosclaw_soccer.sim.contracts import hash_json


def evidence(*, strike_profile=False):
    order = list(G1_DDS_JOINT_NAMES)
    ids = [order.index(n) for n in JOINT_NAMES]
    policy = (strike.make_policy if strike_profile else make_policy)(
        np.full((3, 12), 0.08), hash_json({"train": 303})
    )
    limits = np.tile([-1.0, 1.0], (1, 29, 1))
    baseline = np.zeros((300, 1, 29))
    force = np.zeros((300, 1, 6))
    force[70, 0, 0] = 2
    root = np.zeros((300, 1, 7))
    root[:, :, 6] = 1
    ball = np.zeros((300, 1, 3))
    ball[:, :, 0] = 0.8
    swing = {
        "pre_step_foot_link_position_w": np.zeros((300, 1, 2, 3)),
        "pre_step_foot_linear_jacobian_w": np.zeros((300, 1, 2, 3, 6)),
        "taskspace_selected_side": np.full((300, 1), -1),
        "applied_taskspace_joint_delta_rad": baseline.copy(),
        "baseline_taskspace_joint_target_rad": baseline.copy(),
        "executed_taskspace_joint_target_rad": baseline.copy(),
        "taskspace_joint_limits_rad": limits,
        "pre_step_ball_position_local_m": ball,
        "observed_ball_body_contact_force_peak_n": force,
        "predicted_baseline_joint_target_rad": baseline.copy(),
    }
    report = {
        "contact_motor_policy": policy,
        "contact_motor_policy_hash": policy["policy_hash"],
        "frames": 300,
        "environments": [{}],
        "taskspace_joint_order": order,
        "taskspace_leg_joint_names": [list(x) for x in LEG_NAMES],
        "taskspace_forward_m": 0.08,
        "selected_taskspace_mask": [False],
        "late_swing_actor_hash": "bound-by-outer-auditor",
    }
    deltas = np.zeros((300, 1, 12))
    previous = np.zeros(12)
    contact = previous.copy()
    for frame in range(300):
        delta = (strike.motor_delta if strike_profile else motor_delta)(
            np.asarray(policy["knots_rad"]),
            0.8,
            np.zeros(12),
            limits[0, ids],
            previous,
            contact,
            frame - 70 if frame > 70 else None,
        )
        previous = delta.astype(np.float32).astype(float)
        deltas[frame, 0] = previous
        if frame == 70:
            contact = previous.copy()
    target = baseline.copy()
    target[:, :, ids] += deltas
    motor = {
        "baseline_joint_target_rad": baseline[:, :, ids],
        "applied_joint_delta_rad": deltas,
        "joint_limits_rad": limits[:, ids],
    }
    body = {
        "root_pose_xyzw_m": root,
        "ball_position_before_step_m": ball,
        "joint_target_rad": target,
    }
    return motor, body, swing, {"ball_body_contact_force_peak_n": force.copy()}, report


def test_every_motor_target_and_contact_release_is_reconstructed():
    result = audit_motor_arrays(*evidence())
    assert result["contact_motor_action_audited"]
    assert result["contact_motor_active_frames"] > 70


def test_explicit_strike_profile_is_reconstructed_without_legacy_reinterpretation():
    result = audit_motor_arrays(*evidence(strike_profile=True))
    assert result["contact_motor_action_audited"]
    assert result["contact_motor_active_frames"] > 70


@pytest.mark.parametrize(
    "which,key",
    [
        (0, "applied_joint_delta_rad"),
        (1, "joint_target_rad"),
        (1, "ball_position_before_step_m"),
        (3, "ball_body_contact_force_peak_n"),
    ],
)
def test_tampered_action_observation_or_contact_is_rejected(which, key):
    arrays = copy.deepcopy(evidence())
    arrays[which][key][40, 0, 0] += 0.03
    with pytest.raises(ValueError):
        audit_motor_arrays(*arrays)
