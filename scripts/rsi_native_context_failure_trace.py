"""Finite guard-failure trace validation; no stepping, reset or retry authority."""

import json


def validate_failure_trace(folder, model, expected_reason):
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_measured_context import measured_joint_context
    from rosclaw_soccer.providers.g1.sonic_navigation import future_reference_context
    from rosclaw_soccer.providers.g1.sonic_reference_blend import blend_planner_segment
    from rosclaw_soccer.providers.g1.sonic_runup import _resample_segments_30_to_50
    from scripts.rsi_native_context_recovery_common import POLICY_REASONS

    if expected_reason not in POLICY_REASONS:
        raise ValueError("unknown exception is not a resumable policy rejection")
    failed = json.loads((folder / "failure.json").read_text())
    ticks = failed["completed_control_ticks"]
    if (
        type(ticks) is not int
        or not 20 <= ticks < 300
        or ticks % 20
        or failed["attempted_native_steps"] != 10 * ticks
        or failed["integrated_native_steps"] != 10 * ticks
        or failed["trace_persistence_error"] is not None
        or failed["attempted_source_planner_calls"] != ticks // 20
        or failed["returned_source_planner_calls"] != ticks // 20
    ):
        raise ValueError(
            "guard rejection requires all complete finite control ticks and model calls"
        )
    with np.load(folder / "observed-control-ticks.npz", allow_pickle=False) as a:
        raw = {k: a[k] for k in a.files}
    with np.load(folder / "last-native-state.npz", allow_pickle=False) as a:
        last = {k: a[k] for k in a.files}
    with np.load(folder / "planner-calls.npz", allow_pickle=False) as a:
        calls = {k: a[k] for k in a.files}
    shapes = dict(
        canonical_qpos=(ticks, 43),
        canonical_qvel=(ticks, 41),
        joint_target_rad=(ticks, 29),
        pd_requested_torque_nm=(ticks, 10, 29),
        torque_nm=(ticks, 10, 29),
        actual_actuator_force_nm=(ticks, 10, 29),
        pelvis_z_per_substep_m=(ticks, 10),
        force_n=(ticks, 6),
        ball_position_after_step_m=(ticks, 3),
        target_body_canonical_qpos_after_substep=(ticks, 10, 43),
        target_body_canonical_qvel_after_substep=(ticks, 10, 41),
        ball_contact_force_by_body_n=(ticks, 10, model.nbody),
        neural_action_isaac=(ticks, 29),
        encoder_reference_poses=(ticks, 10, 36),
        entry_stage=(ticks,),
        entry_command=(ticks, 3),
        entry_target_root_xy_m=(ticks, 2),
        entry_error_m=(ticks,),
    )
    if (
        (model.nq, model.nv, model.nu) != (43, 41, 29)
        or set(raw) != set(shapes)
        or any(
            a.shape != shapes[k]
            or a.dtype
            != (
                np.int64
                if k == "entry_stage"
                else np.float32
                if k == "neural_action_isaac"
                else np.float64
            )
            or not np.isfinite(a).all()
            for k, a in raw.items()
        )
        or set(last) != {"qpos", "qvel", "time_sec"}
        or last["qpos"].shape != (43,)
        or last["qvel"].shape != (41,)
        or any(not np.isfinite(a).all() for a in last.values())
        or any(not np.isfinite(a).all() for a in calls.values())
    ):
        raise ValueError("nonfinite or incomplete trace cannot be resumed around")
    joints = [model.joint(n).id for n in G1_DDS_JOINT_NAMES]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    ball_joint = int(model.body("ball").jntadr[0])
    bq, bv = int(model.jnt_qposadr[ball_joint]), int(model.jnt_dofadr[ball_joint])
    q = np.concatenate((last["qpos"][:7], last["qpos"][qi], last["qpos"][bq : bq + 7]))
    v = np.concatenate((last["qvel"][:6], last["qvel"][vi], last["qvel"][bv : bv + 6]))
    np.testing.assert_array_equal(q, raw["target_body_canonical_qpos_after_substep"][-1, -1])
    np.testing.assert_array_equal(v, raw["target_body_canonical_qvel_after_substep"][-1, -1])
    if last["time_sec"].shape != (1,) or abs(float(last["time_sec"][0]) - ticks * 0.02) > 1e-8:
        raise ValueError("last native state clock differs")
    limits = model.jnt_range[joints].copy()
    events = json.loads((folder / "planning-events.json").read_text())["events"]
    if [e["frame"] for e in events] != list(range(0, ticks, 20)):
        raise ValueError("every source planning event required")
    reference = None
    for index, event in enumerate(events):
        frame = index * 20
        count = calls[f"call{index}__raw_count"]
        prediction = calls[f"call{index}__raw_output"]
        if (
            count.shape != (1,)
            or count.dtype.kind not in "iu"
            or not 4 <= int(count[0]) <= 64
            or prediction.ndim != 3
            or prediction.shape[0] != 1
            or prediction.shape[2] != 36
            or prediction.shape[1] < int(count[0])
        ):
            raise ValueError("complete valid source raw model output required")
        segment = _resample_segments_30_to_50(
            [prediction[0, : int(count[0])].astype(np.float64)], 0.02
        )
        if index == 0:
            reference = np.concatenate(
                (segment, np.repeat(segment[-1:], 410 - len(segment), axis=0))
            )
        else:
            measured = raw["canonical_qpos"][frame]
            start = frame + 10
            n = min(len(segment), 410 - start)
            original = future_reference_context(reference, start)
            corrected, delta = measured_joint_context(original, measured, reference[frame], limits)
            old = reference[start : start + n].copy()
            blended = blend_planner_segment(old, segment[:n])
            stitch = event["reference_stitch"]
            for key, expected in dict(
                measured_qpos=measured,
                current_reference=reference[frame],
                original_context=original,
                used_context=corrected,
                joint_context_delta=delta,
                old_future=old,
                generated_future=segment[:n],
                blended_future=blended,
            ).items():
                np.testing.assert_array_equal(np.asarray(stitch[key]), expected)
            reference[start : start + n] = blended
            reference[start + n :] = blended[-1]
        for f in range(frame, min(frame + 20, ticks)):
            np.testing.assert_array_equal(reference[f : f + 10], raw["encoder_reference_poses"][f])
    context = future_reference_context(reference, ticks + 10)
    try:
        measured_joint_context(context, q, reference[ticks], limits)
    except ValueError as error:
        if str(error) != expected_reason:
            raise ValueError(
                "saved state does not reproduce the actual rejection reason"
            ) from error
    else:
        raise ValueError("saved source state does not reproduce a guard rejection")
    return dict(
        completed_control_ticks=ticks,
        native_steps=10 * ticks,
        reason=expected_reason,
        finite_complete_trace_and_clock_verified=True,
        reference_guard_reproduced=True,
        independent_physics_replayed_by_this_validator=False,
        model_math_replayed_by_this_validator=False,
    )
