"""SIM motor student execution and independent NN+native replay, no reference feed."""

PHYSICAL_FIELDS = (
    "canonical_qpos",
    "canonical_qvel",
    "joint_target_rad",
    "pd_requested_torque_nm",
    "torque_nm",
    "actual_actuator_force_nm",
    "pelvis_z_per_substep_m",
    "force_n",
    "ball_position_after_step_m",
    "target_body_canonical_qpos_after_substep",
    "target_body_canonical_qvel_after_substep",
    "ball_contact_force_by_body_n",
)


def exact_array(actual, expected):
    import numpy as np

    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or np.ascontiguousarray(actual).tobytes() != np.ascontiguousarray(expected).tobytes()
    ):
        raise AssertionError("student NN/native array bytes differ")


def action_target(action):
    import numpy as np

    from rosclaw_soccer.providers.g1.sonic_runup import (
        ISAACLAB_TO_MUJOCO,
        G1SonicRunupController,
        _sonic_control_parameters,
    )

    if action.shape != (29,) or action.dtype != np.float32 or not np.isfinite(action).all():
        raise ValueError("finite29 float32 student/consumed action required")
    _, _, scale = _sonic_control_parameters(1.0, (1.0,) * 29)
    return G1SonicRunupController.default_angles + action[ISAACLAB_TO_MUJOCO] * scale


def prediction(model, q, v, previous, frame, hidden):
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.neural_motor_student import motor_student_features

    feature = motor_student_features(q, v, previous, frame=frame)
    with torch.no_grad():
        a, value, next_hidden = model(torch.from_numpy(feature)[None, None, :], hidden)
    return (
        feature,
        a[0, 0].cpu().numpy().copy(),
        np.float32(value[0, 0].cpu().item()),
        next_hidden,
        next_hidden[0, 0].cpu().numpy().copy(),
    )


def rollout_student(model, body, prefix, *, progress):
    import numpy as np

    from rosclaw_soccer.rsi.recorded_entry_review import validate_recording

    validate_recording(prefix, frames=20, nbody=body.model.nbody)
    if body.next_frame != 20 or body.integrated_steps != 200:
        raise ValueError("actual unmodified own20 prefix required")
    np.testing.assert_array_equal(
        body.canonical()[0], prefix["target_body_canonical_qpos_after_substep"][-1, -1]
    )
    rows = []
    hidden = None
    previous = np.zeros(29, dtype=np.float32)
    progress["rows"] = rows
    for frame in range(300):
        if frame < 20:
            q, v = prefix["canonical_qpos"][frame], prefix["canonical_qvel"][frame]
        else:
            q, v = body.canonical()
        feature, proposed, value, hidden, h = prediction(model, q, v, previous, frame, hidden)
        consumed = prefix["neural_action_isaac"][frame].copy() if frame < 20 else proposed.copy()
        target = action_target(consumed)
        progress["last_proposal"] = dict(
            frame=frame, feature=feature, proposed=proposed, consumed=consumed, target=target
        )
        if frame < 20:
            np.testing.assert_array_equal(target, prefix["joint_target_rad"][frame])
            row = {k: prefix[k][frame : frame + 1].copy() for k in PHYSICAL_FIELDS}
        else:
            row = body.step(frame, target)
        row.update(
            student_features=feature[None],
            student_proposed_action_isaac=proposed[None],
            consumed_action_isaac=consumed[None],
            value_estimate=np.array([value], dtype=np.float32),
            hidden_after=h[None],
            owner_index=np.array([int(frame >= 20)], dtype=np.int64),
        )
        rows.append(row)
        previous = consumed
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def replay_student(model, body, trace):
    import numpy as np

    hidden = None
    previous = np.zeros(29, dtype=np.float32)
    validate_student_recording(trace, nbody=body.model.nbody)
    if body.next_frame != 0:
        raise ValueError("new independently initialized native body required")
    for frame in range(300):
        q, v = body.canonical()
        feature, proposed, value, hidden, h = prediction(model, q, v, previous, frame, hidden)
        for actual, expected in (
            (feature, trace["student_features"][frame]),
            (proposed, trace["student_proposed_action_isaac"][frame]),
            (h, trace["hidden_after"][frame]),
        ):
            exact_array(actual, expected)
        if value != trace["value_estimate"][frame] or trace["owner_index"][frame] != int(
            frame >= 20
        ):
            raise ValueError("student critic/memory/ownership trace differs")
        consumed = trace["consumed_action_isaac"][frame]
        if frame >= 20:
            exact_array(proposed, consumed)
        target = action_target(consumed)
        exact_array(target, trace["joint_target_rad"][frame])
        physical = body.step(frame, target)
        for k in PHYSICAL_FIELDS:
            exact_array(physical[k][0], trace[k][frame])
        previous = consumed.copy()
    return dict(
        additional_native_steps=body.integrated_steps,
        student_nn_calls=300,
        student_actions_consumed=280,
        teacher_actions_consumed_after_boundary=0,
        student_neural_math_and_native_physics_bit_exact=True,
        prefix_foundation_neural_math_replayed=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def validate_student_recording(trace, *, nbody):
    import numpy as np

    shapes = dict(
        canonical_qpos=(300, 43),
        canonical_qvel=(300, 41),
        joint_target_rad=(300, 29),
        pd_requested_torque_nm=(300, 10, 29),
        torque_nm=(300, 10, 29),
        actual_actuator_force_nm=(300, 10, 29),
        pelvis_z_per_substep_m=(300, 10),
        force_n=(300, 6),
        ball_position_after_step_m=(300, 3),
        target_body_canonical_qpos_after_substep=(300, 10, 43),
        target_body_canonical_qvel_after_substep=(300, 10, 41),
        ball_contact_force_by_body_n=(300, 10, nbody),
        student_features=(300, 108),
        student_proposed_action_isaac=(300, 29),
        consumed_action_isaac=(300, 29),
        value_estimate=(300,),
        hidden_after=(300, 128),
        owner_index=(300,),
    )
    if set(trace) != set(shapes):
        raise ValueError("complete NN/native student trace required")
    for name, shape in shapes.items():
        expected = (
            np.int64
            if name == "owner_index"
            else np.float64
            if name in PHYSICAL_FIELDS
            else np.float32
        )
        value = trace[name]
        if (
            type(value) is not np.ndarray
            or value.shape != shape
            or value.dtype != expected
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite typed complete student trace required: " + name)
    if np.any(trace["ball_contact_force_by_body_n"] < 0) or not np.array_equal(
        trace["owner_index"], np.array([0] * 20 + [1] * 280, dtype=np.int64)
    ):
        raise ValueError("native nonnegative contact and exact controller boundary required")
    np.testing.assert_array_equal(
        trace["student_proposed_action_isaac"][20:], trace["consumed_action_isaac"][20:]
    )
