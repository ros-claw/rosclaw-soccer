"""Recompute actual categorical-student30 and warm a teacher on consumed actions.

No state transplantation or physical clock rewrites. Foundation actions0..19
are explicitly replayed; student NN/own muscle history recompute0..29; student
muscle outputs20..29 physically execute. Teacher advice never controls physics.
"""


def warm_student30(model, codec, teacher, body, affordance, expected, *, progress):
    import numpy as np

    from rosclaw_soccer.rsi.ball_entry_teacher import ball_entry_decision
    from rosclaw_soccer.rsi.full_entry_forecast import _observation
    from scripts.rsi_motion_student_rollout import (
        disable_teacher_networks,
        prediction,
        proprioception,
        validate_motion_recording,
    )
    from scripts.rsi_motion_teacher_tap import MotionTeacherTap
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, action_target, exact_array

    validate_motion_recording(expected, nbody=body.model.nbody)
    if (
        body.next_frame != 0
        or body.integrated_steps != 0
        or body.attempted_steps != 0
        or codec._next_frame != 0
        or teacher._next_frame != 0
        or teacher._origin_frame != 0
        or teacher.config.pose_reference is not None
        or teacher.config.latent_schedule is not None
        or codec._backend is teacher.backend
        or model.calls != 0
    ):
        raise ValueError("fresh independent student/muscle/teacher and native body required")
    disable_teacher_networks(codec)
    tap = MotionTeacherTap(teacher.backend)
    hidden, previous = None, np.zeros(29, dtype=np.float32)
    rows, labels = [], []
    progress["student_rows"], progress["teacher_labels"] = rows, labels
    try:
        for frame in range(30):
            q, v = body.canonical()
            exact_array(q, expected["canonical_qpos"][frame])
            exact_array(v, expected["canonical_qvel"][frame])
            feature, token, value, hidden, h = prediction(model, q, v, previous, frame, hidden)
            codec.observe(q, v, previous, frame=frame)
            muscle = proprioception(codec)
            for actual, key in (
                (feature, "student_features"),
                (token, "student_motion_token"),
                (h, "hidden_after"),
                (muscle, "muscle_proprioception"),
                (np.asarray(value), "value_estimate"),
            ):
                exact_array(actual, np.asarray(expected[key][frame]))
            consumed = (
                expected["consumed_action_isaac"][frame].copy()
                if frame < 20
                else codec.decode_motion(token)
            )
            exact_array(consumed, expected["consumed_action_isaac"][frame])
            decision = ball_entry_decision(q, v, frame=frame, affordance=affordance)
            obs = _observation(teacher, body, frame, decision.command)
            advice = teacher.prepare_shadow_action(obs)
            label = tap.label(muscle)
            exact_array(advice, label["teacher_advice_action"])
            if frame < 20:
                # Original shared Foundation prefix, not student motor authority.
                exact_array(advice, consumed)
            target = teacher.commit_batched_action(consumed)
            requested = action_target(consumed)
            exact_array(np.asarray(target.target_rad), requested)
            exact_array(np.asarray(target.kp), body.kp)
            exact_array(np.asarray(target.kd), body.kd)
            row = body.step(frame, requested)
            codec.commit_consumed(consumed, foundation_prefix=frame < 20)
            for key in PHYSICAL_FIELDS:
                exact_array(row[key], expected[key][frame : frame + 1])
            row.update(
                student_features=feature[None],
                student_motion_token=token[None],
                muscle_proprioception=muscle[None],
                consumed_action_isaac=consumed[None],
                value_estimate=np.asarray([value], dtype=np.float32),
                hidden_after=h[None],
                owner_index=np.asarray([int(frame >= 20)], dtype=np.int64),
            )
            rows.append(row)
            labels.append(label)
            previous = consumed.copy()
        if (
            body.next_frame != 30
            or body.integrated_steps != 300
            or body.attempted_steps != 300
            or teacher._next_frame != 30
            or codec._next_frame != 30
            or model.calls != 30
            or tap.encoder.calls != 30
            or tap.decoder.calls != 30
        ):
            raise ValueError("actual physical/neural/shadow clocks must all reach30")
        return (
            {k: np.concatenate([r[k] for r in rows]) for k in rows[0]},
            {k: np.stack([r[k] for r in labels]) for k in labels[0]},
        )
    finally:
        # Forecasts must not share mutable instrumentation or its counters.
        teacher.backend._encoder = tap.encoder.session
        teacher.backend._decoder = tap.decoder.session
