"""Rebuild actual waypoint feeds and planned reference stitches, no inference/step."""

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_navigation import future_reference_context
from rosclaw_soccer.providers.g1.sonic_root_waypoint_inputs import RootWaypointIntent
from rosclaw_soccer.providers.g1.sonic_runup import _resample_segments_30_to_50
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def review_waypoint_events(
    raw: dict[str, np.ndarray],
    row: dict[str, Any],
    artifact: dict[str, Any],
    native_joint_limits: np.ndarray,
) -> dict[str, Any]:
    fixed = artifact["fixed_goal"]
    if (
        fixed["goal_basis_hash"] != row["motion_entry_binding_hash"]
        or fixed["goal_deadline_frame"] != 90
        or not np.array_equal(native_joint_limits, fixed["native_joint_limits"])
    ):
        raise ValueError("fixed goal/entry/world/native-limit binding differs")
    target = raw["canonical_qpos"][0, 36:38] - np.array(row["entry_ball_offset_xy_m"])
    if not np.array_equal(target, fixed["fixed_world_goal_xy_m"]):
        raise ValueError("fixed goal must follow declared initial stationary ball/entry offset")
    handoff = row["report"]["handoff_frame"]
    stop = 300 if handoff is None else handoff
    events = artifact["events"]
    if [e["frame"] for e in events] != list(range(0, stop, 20)) or artifact[
        "inputs_are_model_goals_not_measured_future_body"
    ] is not True:
        raise ValueError("exact original planning cadence and source handoff boundary required")
    reference: np.ndarray | None = None
    count = 0
    for frame in range(stop):
        if frame % 20 == 0:
            e = events[count]
            q, v = raw["canonical_qpos"][frame], raw["canonical_qvel"][frame]
            obs_hash = hash_json(dict(frame=frame, qpos=q.tolist(), qvel=v.tolist()))
            if e["observation_hash"] != obs_hash or e["fixed_goal_expired"] is not (frame > 90):
                raise ValueError("waypoint intent must bind this current actual body observation")
            if frame <= 90:
                intent = RootWaypointIntent(
                    (float(target[0]), float(target[1]), float(q[2])),
                    0.0,
                    obs_hash,
                    row["motion_entry_binding_hash"],
                )
                if (
                    hash_json(e["intent"]) != hash_json(intent.report())
                    or e["intent_hash"] != intent.contract_hash
                ):
                    raise ValueError(
                        "NN target intent differs from independently derived current goal"
                    )
            elif e["intent"] is not None or e["intent_hash"] is not None:
                raise ValueError("waypoint must expire without extending original entry deadline")
            if frame == 0:
                context = np.repeat(q[None, :36], 4, axis=0)
            else:
                assert reference is not None
                context = future_reference_context(reference, frame + 10)
            cmd = raw["entry_command"][frame]
            speed = math.hypot(*cmd[:2])
            facing = math.atan2(
                2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6])
            )
            direction = (
                (cmd[0] / speed, cmd[1] / speed)
                if speed > 1e-8
                else (math.cos(facing), math.sin(facing))
            )
            heading = facing + cmd[2] * 0.4
            expected: dict[str, np.ndarray] = dict(
                context_mujoco_qpos=context[None].astype(np.float32),
                target_vel=np.array([speed], dtype=np.float32),
                mode=np.array([0 if speed < 0.02 else 1], dtype=np.int64),
                movement_direction=np.array([[*direction, 0.0]], dtype=np.float32),
                facing_direction=np.array(
                    [[math.cos(heading), math.sin(heading), 0.0]], dtype=np.float32
                ),
                random_seed=np.array([30300 + count], dtype=np.int64),
                has_specific_target=np.array([[int(frame <= 90)]], dtype=np.int64),
                specific_target_positions=np.tile(
                    np.array([*target, q[2]], dtype=np.float32), (1, 4, 1)
                )
                if frame <= 90
                else np.zeros((1, 4, 3), dtype=np.float32),
                specific_target_headings=np.zeros((1, 4), dtype=np.float32),
                allowed_pred_num_tokens=np.array(
                    [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64
                ),
                height=np.array([-1.0], dtype=np.float32),
            )
            if set(expected) != set(e["actual_planner_inputs"]) or set(expected) != set(
                e["actual_planner_input_dtypes"]
            ):
                raise ValueError("all twelve actual model input tensors required")
            for k, gold in expected.items():
                actual = np.array(
                    e["actual_planner_inputs"][k], dtype=e["actual_planner_input_dtypes"][k]
                )
                if actual.dtype != gold.dtype or not np.array_equal(actual, gold):
                    raise ValueError(
                        "planner input differs from state/reference reconstruction: " + k
                    )
            segment = np.asarray(e["raw_planner_segment"], dtype=np.float64)
            if (
                segment.ndim != 2
                or segment.shape[1] != 36
                or not 4 <= len(segment) <= 64
                or not np.isfinite(segment).all()
                or np.any(abs(np.linalg.norm(segment[:, 3:7], axis=1) - 1) > 0.01)
            ):
                raise ValueError("complete finite raw planner segment required")
            violation = float(
                max(
                    0.0,
                    np.maximum(
                        native_joint_limits[:, 0] - segment[:, 7:],
                        segment[:, 7:] - native_joint_limits[:, 1],
                    ).max(),
                )
            )
            original = e["original_planner_record"]
            if (
                violation > 1e-6
                or violation != e["maximum_native_joint_limit_violation_rad"]
                or original["frame"] != frame
                or original["reference_hash"] != hash_bytes(segment.tobytes())
                or not np.array_equal(original["command"], cmd)
                or original["planner_mode"] != int(expected["mode"][0])
            ):
                raise ValueError("native range/raw output reference commitment differs")
            generated = _resample_segments_30_to_50([segment], 0.02)
            if frame == 0:
                reference = np.concatenate(
                    (generated, np.repeat(generated[-1:], max(0, 410 - len(generated)), axis=0))
                )
            else:
                assert reference is not None
                start = frame + 10
                n = min(len(generated), len(reference) - start)
                reference[start : start + n] = generated[:n]
                reference[start + n :] = generated[n - 1]
            count += 1
        assert reference is not None
        if not np.array_equal(raw["encoder_reference_poses"][frame], reference[frame : frame + 10]):
            raise ValueError(
                "encoder reference differs from the recorded current-goal planner stitches"
            )
    return dict(
        planner_calls_reviewed=count,
        navigation_frames_reviewed=stop,
        actual_goal_inputs_reconstructed=True,
        planned_future_context_not_measured_future=True,
        original_reference_stitch_math_reproduced=True,
        new_neural_calls=0,
        new_native_steps=0,
    )


def encoder_root_channel_audit(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    """Input-map counterfactual only; no sessions, simulator state or rollout.

    A missing absolute-root-reference channel does not imply that joint-motion
    references cannot cause locomotion. It limits direct waypoint tracking.
    """
    from rosclaw_soccer.providers.g1.sonic_runup import (
        _VARIANTS,
        G1SonicRunupConfig,
        G1SonicRunupController,
    )

    controller = object.__new__(G1SonicRunupController)
    controller.config = G1SonicRunupConfig(model_variant="low_latency")
    controller._variant = _VARIANTS["low_latency"]
    rows = []
    for frame in (0, 60, 90):
        state = SimpleNamespace(
            qpos=raw["canonical_qpos"][frame].copy(), qvel=raw["canonical_qvel"][frame].copy()
        )
        original = raw["encoder_reference_poses"][frame].copy()
        controller.reference = original.copy()
        baseline = controller._encoder_observation(state, 0)
        controller.reference = original.copy()
        controller.reference[:, :3] += np.array([0.1, -0.1, 0.05])
        root_shift = controller._encoder_observation(state, 0)
        controller.reference = original.copy()
        controller.reference[:, 7] += 0.01
        joint_shift = controller._encoder_observation(state, 0)
        if not np.array_equal(baseline, root_shift) or np.array_equal(baseline, joint_shift):
            raise ValueError("expected frozen G1 mode0 root/joint input-channel contract changed")
        rows.append(
            dict(
                frame=frame,
                original_encoder_input_hash=hash_bytes(baseline.tobytes()),
                root_reference_only_shift_input_hash=hash_bytes(root_shift.tobytes()),
                joint_reference_positive_control_hash=hash_bytes(joint_shift.tobytes()),
                root_shift_input_exactly_identical=True,
                joint_positive_control_changes_input=True,
            )
        )
    return dict(
        rows=rows,
        reference_xyz_not_encoded_in_this_G1_mode=True,
        actual_body_and_ball_unchanged=True,
        new_neural_calls=0,
        new_native_steps=0,
        not_proof_that_joint_motion_cannot_locomote=True,
        not_exclusive_cause_of_native_failure=True,
    )
