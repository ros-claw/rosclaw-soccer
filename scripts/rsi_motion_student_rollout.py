"""Own-state64 intention -> frozen muscle NN -> native PD, plus independent replay."""

from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, action_target, exact_array


class ForbiddenTeacherNetwork:
    def run(self, *args, **kwargs):
        raise RuntimeError("teacher encoder/planner forbidden in motion student execution")

    def get_inputs(self):
        raise RuntimeError("teacher encoder/planner forbidden in motion student execution")


def disable_teacher_networks(codec):
    codec._backend._encoder = ForbiddenTeacherNetwork()
    codec._backend._planner = ForbiddenTeacherNetwork()


def proprioception(codec):
    import numpy as np

    history = list(codec._history)
    if len(history) != 10 or not codec._observed:
        raise ValueError("pending own ten-state muscle history required")
    return np.concatenate(
        [
            np.asarray([entry[k] for entry in history], dtype=np.float32).reshape(-1)
            for k in range(5)
        ]
    )


def prediction(model, q, v, previous, frame, hidden):
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.neural_motor_student import motor_student_features

    feature = motor_student_features(q, v, previous, frame=frame)
    with torch.no_grad():
        token, value, next_hidden = model(torch.from_numpy(feature)[None, None], hidden)
    proposed = token[0, 0].cpu().numpy().copy()
    if (
        proposed.shape != (64,)
        or proposed.dtype != np.float32
        or not np.isfinite(proposed).all()
        or not np.array_equal(proposed * 16, np.round(proposed * 16))
        or proposed.min() < -1
        or proposed.max() > 0.9375
    ):
        raise ValueError("finite original FSQ32 motion intent required")
    return (
        feature,
        proposed,
        np.float32(value[0, 0].item()),
        next_hidden,
        next_hidden[0, 0].cpu().numpy().copy(),
    )


def rollout_motion_student(model, codec, body, prefix, *, progress):
    import numpy as np

    from rosclaw_soccer.rsi.recorded_entry_review import validate_recording

    validate_recording(prefix, frames=20, nbody=body.model.nbody)
    if body.next_frame != 20 or body.integrated_steps != 200 or codec._next_frame != 0:
        raise ValueError("actual own20 prefix and unused muscle history required")
    exact_array(body.canonical()[0], prefix["target_body_canonical_qpos_after_substep"][-1, -1])
    disable_teacher_networks(codec)
    hidden, previous = None, np.zeros(29, dtype=np.float32)
    rows = []
    progress["rows"] = rows
    for frame in range(300):
        q, v = (
            (prefix["canonical_qpos"][frame], prefix["canonical_qvel"][frame])
            if frame < 20
            else body.canonical()
        )
        feature, token, value, hidden, h = prediction(model, q, v, previous, frame, hidden)
        codec.observe(q, v, previous, frame=frame)
        proprio = proprioception(codec)
        if frame < 20:
            consumed = prefix["neural_action_isaac"][frame].copy()
            target = action_target(consumed)
            exact_array(target, prefix["joint_target_rad"][frame])
            row = {k: prefix[k][frame : frame + 1].copy() for k in PHYSICAL_FIELDS}
            codec.commit_consumed(consumed, foundation_prefix=True)
        else:
            consumed = codec.decode_motion(token)
            target = action_target(consumed)
            progress["last_proposal"] = dict(
                frame=frame, feature=feature, token=token, consumed=consumed, target=target
            )
            row = body.step(frame, target)
            codec.commit_consumed(consumed)
        row.update(
            student_features=feature[None],
            student_motion_token=token[None],
            muscle_proprioception=proprio[None],
            consumed_action_isaac=consumed[None],
            value_estimate=np.asarray([value], dtype=np.float32),
            hidden_after=h[None],
            owner_index=np.asarray([int(frame >= 20)], dtype=np.int64),
        )
        rows.append(row)
        previous = consumed.copy()
    result = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    validate_motion_recording(result, nbody=body.model.nbody)
    return result


def replay_motion_student(model, codec, body, trace):
    import numpy as np

    validate_motion_recording(trace, nbody=body.model.nbody)
    if body.next_frame != 0 or codec._next_frame != 0:
        raise ValueError("independently initialized native body and muscle memory required")
    disable_teacher_networks(codec)
    hidden, previous = None, np.zeros(29, dtype=np.float32)
    for frame in range(300):
        q, v = body.canonical()
        feature, token, value, hidden, h = prediction(model, q, v, previous, frame, hidden)
        codec.observe(q, v, previous, frame=frame)
        for actual, expected in (
            (feature, trace["student_features"][frame]),
            (token, trace["student_motion_token"][frame]),
            (h, trace["hidden_after"][frame]),
            (proprioception(codec), trace["muscle_proprioception"][frame]),
            (np.asarray(value), trace["value_estimate"][frame : frame + 1].reshape(())),
        ):
            exact_array(actual, expected)
        consumed = (
            trace["consumed_action_isaac"][frame].copy()
            if frame < 20
            else codec.decode_motion(token)
        )
        exact_array(consumed, trace["consumed_action_isaac"][frame])
        target = action_target(consumed)
        exact_array(target, trace["joint_target_rad"][frame])
        row = body.step(frame, target)
        codec.commit_consumed(consumed, foundation_prefix=frame < 20)
        for name in PHYSICAL_FIELDS:
            exact_array(row[name], trace[name][frame : frame + 1])
        previous = consumed.copy()
    counts = dict(intention_network_calls=300)
    method = getattr(model, "network_call_counts", None)
    if callable(method):
        measured = method()
        if (
            type(measured) is not dict
            or not set(measured)
            <= {"intention_network_calls", "policy_invocations", "router_network_calls"}
            or measured.get("intention_network_calls", 0) < 300
            or any(type(v) is not int or not 0 <= v <= 10000 for v in measured.values())
        ):
            raise ValueError("bounded explicit neural execution counts required")
        counts.update(measured)
    return dict(
        additional_native_steps=body.integrated_steps,
        **counts,
        muscle_decoder_calls=280,
        teacher_encoder_calls=0,
        teacher_planner_calls=0,
        student_neural_and_muscle_native_bit_exact=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def validate_motion_recording(trace, *, nbody):
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
        student_motion_token=(300, 64),
        muscle_proprioception=(300, 930),
        consumed_action_isaac=(300, 29),
        value_estimate=(300,),
        hidden_after=(300, 128),
        owner_index=(300,),
    )
    if set(trace) != set(shapes):
        raise ValueError("complete distinct intention/muscle/native fields required")
    for key, shape in shapes.items():
        expected = (
            np.int64
            if key == "owner_index"
            else np.float64
            if key in PHYSICAL_FIELDS
            else np.float32
        )
        value = trace[key]
        if (
            type(value) is not np.ndarray
            or value.shape != shape
            or value.dtype != expected
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite full motion/muscle trace required: " + key)
    token = trace["student_motion_token"]
    if (
        not np.array_equal(token * 16, np.round(token * 16))
        or token.min() < -1
        or token.max() > 0.9375
        or np.any(trace["ball_contact_force_by_body_n"] < 0)
        or not np.array_equal(
            trace["owner_index"], np.asarray([0] * 20 + [1] * 280, dtype=np.int64)
        )
    ):
        raise ValueError("original FSQ32, nonnegative contacts and exact prefix boundary required")
