"""Synthetic reference-stitch fixtures, never native task evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_root_waypoint_inputs import RootWaypointIntent
from rosclaw_soccer.rsi.root_waypoint_recorded_review import (
    encoder_root_channel_audit,
    review_waypoint_events,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def fixture():
    q = np.zeros((300, 43))
    q[:, 2] = 0.793
    q[:, 3] = 1.0
    q[:, 36:39] = [0.4, -0.1, 0.13]
    q[:, 39] = 1.0
    v = np.zeros((300, 41))
    cmd = np.zeros((300, 3))
    limits = np.tile([-3.2, 3.2], (29, 1))
    pose = q[0, :36].copy()
    segment = np.tile(pose, (32, 1))
    raw = dict(
        canonical_qpos=q,
        canonical_qvel=v,
        entry_command=cmd,
        encoder_reference_poses=np.tile(pose, (300, 10, 1)),
    )
    binding = "sha256:" + "b" * 64
    row = dict(
        entry_ball_offset_xy_m=[0.37, -0.06],
        motion_entry_binding_hash=binding,
        report=dict(handoff_frame=30),
    )
    target = q[0, 36:38] - np.array(row["entry_ball_offset_xy_m"])
    events = []
    for index, frame in enumerate((0, 20)):
        oh = hash_json(dict(frame=frame, qpos=q[frame].tolist(), qvel=v[frame].tolist()))
        intent = RootWaypointIntent((float(target[0]), float(target[1]), 0.793), 0.0, oh, binding)
        feed = dict(
            context_mujoco_qpos=np.tile(pose, (1, 4, 1)).astype(np.float32),
            target_vel=np.array([0.0], dtype=np.float32),
            mode=np.array([0], dtype=np.int64),
            movement_direction=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            facing_direction=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            random_seed=np.array([30300 + index], dtype=np.int64),
            has_specific_target=np.array([[1]], dtype=np.int64),
            specific_target_positions=np.tile(
                np.array([*target, 0.793], dtype=np.float32), (1, 4, 1)
            ),
            specific_target_headings=np.zeros((1, 4), dtype=np.float32),
            allowed_pred_num_tokens=np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64),
            height=np.array([-1.0], dtype=np.float32),
        )
        events.append(
            dict(
                frame=frame,
                intent=intent.report(),
                intent_hash=intent.contract_hash,
                observation_hash=oh,
                fixed_goal_expired=False,
                actual_planner_inputs={k: a.tolist() for k, a in feed.items()},
                actual_planner_input_dtypes={k: str(a.dtype) for k, a in feed.items()},
                raw_planner_segment=segment.tolist(),
                maximum_native_joint_limit_violation_rad=0.0,
                original_planner_record=dict(
                    frame=frame,
                    command=[0.0, 0.0, 0.0],
                    reference_hash=hash_bytes(segment.tobytes()),
                    planner_mode=0,
                ),
            )
        )
    a = dict(
        fixed_goal=dict(
            goal_basis_hash=binding,
            goal_deadline_frame=90,
            native_joint_limits=limits.tolist(),
            fixed_world_goal_xy_m=target.tolist(),
        ),
        events=events,
        inputs_are_model_goals_not_measured_future_body=True,
    )
    return raw, row, a, limits


def test_current_state_goal_and_all_actual_encoder_reference_frames_reconstructed():
    raw, row, a, limits = fixture()
    result = review_waypoint_events(raw, row, a, limits)
    assert result["planner_calls_reviewed"] == 2 and result["navigation_frames_reviewed"] == 30
    assert result["original_reference_stitch_math_reproduced"]
    assert result["new_native_steps"] == result["new_neural_calls"] == 0


def test_root_only_reference_error_is_unobservable_but_joint_positive_control_is_not():
    raw, _, _, _ = fixture()
    before = copy.deepcopy(raw)
    d = encoder_root_channel_audit(raw)
    assert len(d["rows"]) == 3 and d["reference_xyz_not_encoded_in_this_G1_mode"]
    assert d["not_exclusive_cause_of_native_failure"]
    for r in d["rows"]:
        assert r["original_encoder_input_hash"] == r["root_reference_only_shift_input_hash"]
        assert r["original_encoder_input_hash"] != r["joint_reference_positive_control_hash"]
    for k in raw:
        np.testing.assert_array_equal(raw[k], before[k])


@pytest.mark.parametrize(
    "change", ["goal", "observation", "reference", "output", "deadline", "cadence"]
)
def test_corrupted_goal_reference_or_timing_is_rejected(change):
    raw, row, a, limits = fixture()
    before = copy.deepcopy(a)
    if change == "goal":
        a["events"][0]["actual_planner_inputs"]["specific_target_positions"][0][0][0] += 0.01
    elif change == "observation":
        raw["canonical_qpos"][20, 0] += 0.01
    elif change == "reference":
        raw["encoder_reference_poses"][5, 0, 0] += 0.01
    elif change == "output":
        a["events"][0]["raw_planner_segment"][5][7] = 4.0
    elif change == "deadline":
        a["fixed_goal"]["goal_deadline_frame"] = 91
    else:
        a["events"].pop()
    with pytest.raises(ValueError):
        review_waypoint_events(raw, row, a, limits)
    assert before["fixed_goal"]["goal_deadline_frame"] == 90
