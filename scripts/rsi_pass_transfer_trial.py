"""Reusable SIM transfer collector; never used as the audit integrator.

The caller owns immutable source admission, whole-bank preflight and budget.
Normal terminal trials persist all charged native/NN rows before scoring.
No simulator or model is allocated by importing this module.
"""

import json
from types import SimpleNamespace

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
from rosclaw_soccer.rsi.paid_run_pilot_recovery import stand_from_raw
from rosclaw_soccer.rsi.pass_transfer_outcome import pass_transfer_outcome
from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_stroke_search_common import stroke_action


def collect_trial(
    setup,
    actor,
    course,
    *,
    output,
    index,
    counts,
    student=None,
    student_prefix_controls=None,
    emit_course_rows=True,
    feedback_variant=None,
    feedback_anticipation_seconds=0.0,
    sampling_seed=None,
    sampling_profile=None,
    sampling_include_recovery=False,
    receiver_intent_mode="pursuit-v1",
    receiver_role_ablation=False,
):
    if type(receiver_role_ablation) is not bool or (
        receiver_role_ablation
        and (
            sampling_seed is None
            or sampling_include_recovery is not True
            or feedback_variant is not None
            or student is not None
            or course["assisted"] is not False
        )
    ):
        raise ValueError("separate full-stochastic receiver-role diagnostic required")
    if (
        type(receiver_intent_mode) is not str
        or receiver_intent_mode not in ("pursuit-v1", "anchor-plane-v2")
        or (
            receiver_intent_mode != "pursuit-v1"
            and (
                (sampling_seed is not None and not receiver_role_ablation)
                or feedback_variant is not None
                or student is not None
                or course["assisted"] is not False
            )
        )
    ):
        raise ValueError("separate explicit pure-NN receiver-role ablation required")
    if type(sampling_include_recovery) is not bool or (
        sampling_include_recovery and sampling_seed is None
    ):
        raise ValueError("explicit stochastic recovery requires a sampling seed/profile")
    if (sampling_seed is None) != (sampling_profile is None):
        raise ValueError("explicit sampling seed/profile pair required")
    if sampling_seed is not None:
        from rosclaw_soccer.rsi.motor_exploration_profile import exploration_sigma
        from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter

        exploration_sigma(sampling_profile, "box-truncated-ar1")
        if (
            type(sampling_seed) is not int
            or not 0 <= sampling_seed < 2**31
            or type(actor) is not RefinementMotorAdapter
            or student is not None
            or course["assisted"] is not False
            or feedback_variant is not None
            or feedback_anticipation_seconds != 0.0
        ):
            raise ValueError("pure own refinement stochastic rollout only; no stacked teacher")
    if feedback_variant is not None:
        from rosclaw_soccer.rsi.causal_foot_ball_feedback import feedback_parameters

        feedback_parameters(feedback_variant, anticipation_seconds=feedback_anticipation_seconds)
        if student is not None or course["assisted"] is not False:
            raise ValueError("separate unassisted current NN basis required for feedback study")
    elif feedback_anticipation_seconds != 0.0:
        raise ValueError("anticipation requires an explicit feedback study")
    if type(emit_course_rows) is not bool:
        raise ValueError("explicit course log disclosure mode required before physics")
    if (student is None) != (student_prefix_controls is None) or (
        student is not None
        and (type(student_prefix_controls) is not int or student_prefix_controls not in (0, 80))
    ):
        raise ValueError("explicit separate student and preregistered0/80 control prefix required")
    if student is not None and (
        course["assisted"] is not True or "shadow_actor_calls" not in counts
    ):
        raise ValueError(
            "explicit teacher-continuation mode and separate shadow accounting required"
        )

    def store(folder, name, rows):
        path = folder / (name + ".npz")
        np.savez_compressed(path, **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
        return dict(path=str(path.relative_to(output)), hash=digest(path))

    ep, w, forward, home, _ = setup.prepare(
        course["context"],
        course["side"],
        ball_planar_offset_world=None
        if "ball_planar_offset_world" not in course
        else np.array(course["ball_planar_offset_world"], dtype=np.float64),
    )
    initial_q, initial_v = w.data.qpos.copy(), w.data.qvel.copy()
    folder = output / f"trial{index}"
    folder.mkdir()
    init_path = folder / "initial.npz"
    np.savez_compressed(init_path, qpos=initial_q, qvel=initial_v)
    if index % 2:
        with np.load(output / f"trial{index - 1}" / "initial.npz", allow_pickle=False) as z:
            if not np.array_equal(initial_q, z["qpos"]) or not np.array_equal(initial_v, z["qvel"]):
                raise ValueError("paired teacher/control arms must start in identical physics")
    goal = np.array(course["target"])
    sampling, sampling_rows = None, []
    if sampling_seed is not None:
        from rosclaw_soccer.rsi.shared_motor_sampling import SharedMotorSampling

        sampling = SharedMotorSampling(
            ep.scale[np.argsort(ep.order)],
            seed=sampling_seed,
            profile=sampling_profile,
            include_recovery=sampling_include_recovery,
        )
    feedback = None
    feedback_rows = []
    if feedback_variant is not None:
        from rosclaw_soccer.rsi.causal_foot_ball_feedback import CausalFootBallFeedback

        feedback = CausalFootBallFeedback(
            w,
            ep,
            goal=goal,
            variant=feedback_variant,
            anticipation_seconds=feedback_anticipation_seconds,
        )
    lateral = np.array([-forward[1], forward[0]])
    anchor = goal[:2] + 0.25 * forward + 0.12 * lateral
    passer = SharedNativeNeuralPlayer(
        w,
        "a/",
        actor,
        clip=ep.clip,
        torso_index=ep.torso_index,
        build_observation=ep.build_observation,
        goal=goal,
        offset=ep.offset,
        scale=ep.scale,
        order=ep.order,
    )
    pair = None
    if student is not None:
        from rosclaw_soccer.rsi.shared_teacher_pair import SharedTeacherPair

        learner = SharedNativeNeuralPlayer(
            w,
            "a/",
            student,
            clip=ep.clip,
            torso_index=ep.torso_index,
            build_observation=ep.build_observation,
            goal=goal,
            offset=ep.offset,
            scale=ep.scale,
            order=ep.order,
        )
        pair = SharedTeacherPair(learner, passer)
    velocity = ContinuousVelocityStream(
        setup.velocity_path,
        setup.velocity_config,
        expected_model_hash=setup.pins[str(setup.velocity_path)],
        expected_config_hash=setup.pins[str(setup.velocity_config)],
        horizon_controls=400,
    )
    ready = OnlineReceivingBoundary(w, "b/")
    physical, nn_a, nn_b = [], [], []
    launch, stand_controls = False, 0
    for step in range(ep.clip.length + 100):
        mujoco.mj_forward(w.model, w.data)
        controller = passer if pair is None else pair
        if controller.phase == "MOTION_COMPLETE":
            controller.begin_standing()
        standing = passer.phase == "STANDING"
        if pair is None:
            target_a, ta = passer.propose(with_reference_evidence=True)
        else:
            _, trace_s, target_a, ta = pair.propose()
            trace_t = {k: v.copy() for k, v in ta.items()}
        action, delta, clipped = ta["NN_action"].copy(), np.zeros(29), False
        if course["assisted"] and not standing:
            action, delta, clipped = stroke_action(
                SimpleNamespace(frame=passer.frame, order=ep.order, scale=ep.scale),
                action,
                ta["NN_reference_nominal"],
                course["center"],
                np.array(course["latent"]),
                foot=course["foot"],
            )
            target_a = ep.offset + action[ep.order] * ep.scale
        if feedback is not None:
            action, delta, clipped, fb = feedback.apply(
                action, ta["NN_reference_nominal"], frame=step, standing=standing
            )
            target_a = ep.offset + action[ep.order] * ep.scale
            feedback_rows.append(dict(measured_feedback=fb))
        if pair is not None:
            advice, advice_delta, advice_clipped = action.copy(), delta.copy(), clipped
            actual_student = step < student_prefix_controls
            if actual_student:
                action = trace_s["NN_action"].copy()
                target_a = ep.offset + action[ep.order] * ep.scale
                delta, clipped = np.zeros(29), False
                ta = {k: v.copy() for k, v in trace_s.items()}
            ta.update({"STUDENT_" + k: v.copy() for k, v in trace_s.items()})
            ta.update({"TEACHER_" + k: v.copy() for k, v in trace_t.items()})
            ta.update(
                teacher_advice_action=advice,
                teacher_advice_delta_HG=advice_delta,
                teacher_advice_clipped=np.array(advice_clipped),
                student_executed=np.array(actual_student),
            )
        if sampling is not None:
            action, sample_trace = sampling.propose(
                action, ta["NN_reference_nominal"], control=step, standing=standing
            )
            target_a = ep.offset + action[ep.order] * ep.scale
            sampling_rows.append(sample_trace)
        q, v = w.layouts["b/"].canonical(w.data)
        qw, x, y, z = q[3:7]
        yaw = float(np.arctan2(2 * (qw * z + x * y), 1 - 2 * (y * y + z * z)))
        intent = receiving_intent(
            root_xy=q[:2],
            yaw=yaw,
            ball_xy=q[36:38],
            ball_velocity_xy=v[35:37],
            home_xy=home,
            pass_direction=forward,
            launch_seen=launch,
            ready_anchor_xy=anchor,
            mode=receiver_intent_mode,
        )
        launch = intent.launch_seen
        command = np.array(intent.command_body, np.float32)
        target_b = velocity.propose(
            q, v, w.layouts["b/"].angular_velocity_body(w.data), command, frame=step
        )
        tb = dict(
            obs=velocity.last_observation.copy(),
            action=velocity.last_action.copy(),
            command=command,
            phase=np.array(velocity.phase),
            control=np.array(step),
        )
        row = w.advance({"a/": target_a, "b/": target_b})
        counts["native_steps"] += 10
        counts["actor_calls"] += 2
        if pair is None:
            passer.commit(executed_action=action)
        else:
            counts["shadow_actor_calls"] += 1
            pair.commit(action)
        ready.update()
        stand_controls += int(standing)
        ta.update(
            control=np.array(step),
            executed_action=action.copy(),
            teacher_delta_HG=delta,
            teacher_authority_clipped=np.array(clipped),
            standing=np.array(standing),
        )
        nn_a.append(ta)
        nn_b.append(tb)
        physical.append(
            {k: row[k] for k in ("qpos", "qvel", "torque", "ball_forces")}
            | dict(
                body_failed=np.array([row["body_failed"][p] for p in ("a/", "b/")]),
                target_a=target_a.copy(),
                target_b=target_b.copy(),
                time_before=np.array(row["time_before"]),
            )
        )
        if w.failed:
            break
    raw = {k: np.stack([r[k] for r in physical]) for k in physical[0]}
    # Persist every charged native and inference row BEFORE scoring;
    # a scoring defect must not destroy an actual physical execution.
    artifacts = {
        "initial": dict(path=str(init_path.relative_to(output)), hash=digest(init_path)),
        "physical": store(folder, "physical", physical),
        "A": store(folder, "A", nn_a),
        "B": store(folder, "B", nn_b),
    }
    if feedback is not None:
        artifacts["feedback"] = store(folder, "feedback", feedback_rows)
    if sampling is not None:
        artifacts["sampling"] = store(folder, "sampling", sampling_rows)
    al = w.layouts["a/"]
    qids = [*range(al.rq, al.rq + 7), *al.qi, *range(al.bq, al.bq + 7)]
    vids = [*range(al.rv, al.rv + 6), *al.vi, *range(al.bv, al.bv + 6)]
    recovery = None
    if stand_controls:
        recovery = stand_from_raw(
            dict(
                qpos=raw["qpos"][-stand_controls:, :, qids],
                qvel=raw["qvel"][-stand_controls:, :, vids],
                body_failed=raw["body_failed"][-stand_controls:, 0],
            )
        )
    measure = pass_transfer_outcome(
        raw["qpos"][:, :, al.bq : al.bq + 3].reshape(-1, 3),
        raw["qvel"][:, :, al.bv : al.bv + 3].reshape(-1, 3),
        raw["ball_forces"].reshape(-1, 2, 3),
        goal,
        whole_body_healthy=not w.failed,
        passer_recovery_observed=bool(
            stand_controls == 100
            and recovery
            and recovery["healthy"]
            and recovery["stability_observed"]
        ),
        receiver_first_ready_control=ready.first_ready_control,
        receiver_excluded_contact=ready.excluded_contact,
    )
    result = dict(
        index=index,
        course=course,
        controls=w.control_steps,
        native_steps=w.native_steps,
        actor_calls=2 * w.control_steps,
        stand_controls=stand_controls,
        A_recovery=recovery,
        outcome=measure,
        termination="BODY_FAILURE" if w.failed else "FULL_MOTION_PLUS100NN_RECOVERY",
        artifacts=artifacts,
        neural_actor_optimizer_updates=0,
    )
    if feedback is not None:
        result["feedback_variant"] = feedback_variant
        if feedback_anticipation_seconds:
            result["feedback_anticipation_seconds"] = feedback_anticipation_seconds
    if receiver_intent_mode != "pursuit-v1":
        result["receiver_intent_mode"] = receiver_intent_mode
    if sampling is not None:
        result.update(
            sampling.metadata(reference_controls=ep.clip.length, recovery_controls=stand_controls)
        )
        result.update(
            assisted=False,
            external_teacher=False,
            authority_rejected=False,
            all_actor_actions_actually_executed=True,
            natural_terminal_observed=True,
        )
    if pair is not None:
        result.update(
            student_prefix_controls=student_prefix_controls,
            shadow_actor_calls=w.control_steps,
            total_neural_calls=3 * w.control_steps,
            teacher_advice_alone_physically_qualified=False,
            learner_hidden_state_borrowed=False,
            PPO_actor_rows=0,
        )
    if receiver_role_ablation:
        result.update(
            receiver_role_ablation=True,
            controller_kind="receiver-role-stochastic-diagnostic-native-v3",
        )
    write_once(folder / "complete.json", result)
    from rosclaw_soccer.rsi.pass_trial_logging import trial_progress

    print(json.dumps(trial_progress(result, include_course=emit_course_rows)), flush=True)
    return result
