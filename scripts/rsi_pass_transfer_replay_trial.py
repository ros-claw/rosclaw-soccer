"""Reusable own-history replay using ONLY the independent manual native clock.

Shares NN codecs and teacher rule on own states; never imports the producer
trial collector or executes saved actions/hidden state. No import allocation.
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
from rosclaw_soccer.rsi.relay_audit_clock import RelayAuditClock
from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stroke_search_common import stroke_action


def replay_trial(
    setup,
    actor,
    course,
    claim,
    *,
    raw_root,
    output,
    index,
    counts,
    prior_initial=None,
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
    receiver_motor=None,
):
    from rosclaw_soccer.rsi.receiving_initialization_protocol import KIND as RECEIVING_KIND

    if (claim.get("receiver_motor_initialization_only", False) is True) != (
        receiver_motor is not None
    ):
        raise ValueError("exact separate receiving motor replay required")
    if receiver_motor is not None:
        from rosclaw_soccer.rsi.receiving_initialization_protocol import admit_motor

        admit_motor(receiver_motor)
        if (
            receiver_role_ablation
            or receiver_intent_mode != "pursuit-v1"
            or student is not None
            or feedback_variant is not None
            or course["assisted"] is not False
            or sampling_seed is None
            or sampling_include_recovery is not True
        ):
            raise ValueError("separate untrained receiving identity replay required")
    if (
        type(receiver_role_ablation) is not bool
        or (claim.get("receiver_role_ablation", False) is not receiver_role_ablation)
        or (
            receiver_role_ablation
            and (
                sampling_seed is None
                or sampling_include_recovery is not True
                or feedback_variant is not None
                or student is not None
                or course["assisted"] is not False
            )
        )
    ):
        raise ValueError("exact separate full-stochastic receiver-role diagnostic required")
    if (
        type(receiver_intent_mode) is not str
        or receiver_intent_mode not in ("pursuit-v1", "anchor-plane-v2")
        or claim.get("receiver_intent_mode", "pursuit-v1") != receiver_intent_mode
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
        raise ValueError("exact separate receiver-role replay mode required before physics")
    if type(sampling_include_recovery) is not bool or (
        sampling_include_recovery and sampling_seed is None
    ):
        raise ValueError("explicit stochastic recovery requires a sampling seed/profile")
    if (sampling_seed is None) != (sampling_profile is None):
        raise ValueError("explicit own sampling seed/profile pair required")
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
            or claim.get("sampling_seed") != sampling_seed
            or claim.get("sampling_profile") != sampling_profile
            or claim.get("sampling_law") != "box-truncated-ar1"
            or claim.get("controller_kind")
            != (
                RECEIVING_KIND
                if receiver_motor is not None
                else "receiver-role-stochastic-diagnostic-native-v3"
                if receiver_role_ablation
                else (
                    "pure-refinement-full-recovery-stochastic-native-v2"
                    if sampling_include_recovery
                    else "pure-refinement-stochastic-native-v1"
                )
            )
            or claim.get("deterministic_recovery_excluded_from_actor_rows")
            is not (not sampling_include_recovery)
        ):
            raise ValueError("exact pure own refinement sampling configuration required")
    elif "sampling_seed" in claim:
        raise ValueError("stochastic source requires actual independent random-law replay")
    if feedback_variant is not None:
        from rosclaw_soccer.rsi.causal_foot_ball_feedback import feedback_parameters

        feedback_parameters(feedback_variant, anticipation_seconds=feedback_anticipation_seconds)
        if (
            student is not None
            or course["assisted"] is not False
            or claim.get("feedback_variant") != feedback_variant
            or claim.get("feedback_anticipation_seconds", 0.0) != feedback_anticipation_seconds
        ):
            raise ValueError("same explicit unassisted-NN feedback configuration required")
    elif "feedback_variant" in claim:
        raise ValueError("feedback source requires independent causal rule replay")
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
        raise ValueError("explicit continuation mode and separate shadow accounting required")

    def equal(actual, saved, index):
        if set(actual) != set(saved) or any(
            not np.array_equal(v, saved[k][index]) for k, v in actual.items()
        ):
            raise ValueError(f"own neural/native trace differs at control{index}")

    def load(claim, key):
        with np.load(raw_root / claim["artifacts"][key]["path"], allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    ep, initial, forward, home, _ = setup.prepare(
        course["context"],
        course["side"],
        ball_planar_offset_world=None
        if "ball_planar_offset_world" not in course
        else np.array(course["ball_planar_offset_world"], dtype=np.float64),
    )
    w = RelayAuditClock(initial)
    raw, na, nb, ini = [load(claim, k) for k in ("physical", "A", "B", "initial")]
    if (
        set(ini) != {"qpos", "qvel"}
        or not np.array_equal(ini["qpos"], w.data.qpos)
        or not np.array_equal(ini["qvel"], w.data.qvel)
    ):
        raise ValueError("own pre-physics initial state differs")
    if index % 2 and any(not np.array_equal(v, prior_initial[k]) for k, v in ini.items()):
        raise ValueError("paired whole initial states differ")
    goal = np.array(course["target"])
    sampling, sampling_trace = None, None
    if sampling_seed is not None:
        from rosclaw_soccer.rsi.shared_motor_sampling import SharedMotorSampling

        sampling = SharedMotorSampling(
            ep.scale[np.argsort(ep.order)],
            seed=sampling_seed,
            profile=sampling_profile,
            include_recovery=sampling_include_recovery,
        )
        sampling_trace = load(claim, "sampling")
    feedback, feedback_trace = None, None
    if feedback_variant is not None:
        from rosclaw_soccer.rsi.causal_foot_ball_feedback import CausalFootBallFeedback

        feedback = CausalFootBallFeedback(
            w,
            ep,
            goal=goal,
            variant=feedback_variant,
            anticipation_seconds=feedback_anticipation_seconds,
        )
        feedback_trace = load(claim, "feedback")
    anchor = goal[:2] + 0.25 * forward + 0.12 * np.array([-forward[1], forward[0]])
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
    velocity = (
        None
        if receiver_motor is not None
        else ContinuousVelocityStream(
            setup.velocity_path,
            setup.velocity_config,
            expected_model_hash=setup.pins[str(setup.velocity_path)],
            expected_config_hash=setup.pins[str(setup.velocity_config)],
            horizon_controls=400,
        )
    )
    receiving_player = None
    if receiver_motor is not None:
        from rosclaw_soccer.rsi.shared_native_receiving_player import SharedNativeReceivingPlayer

        receiving_player = SharedNativeReceivingPlayer(
            w,
            "b/",
            receiver_motor,
            model=setup.velocity_path,
            config=setup.velocity_config,
            expected_model_hash=setup.pins[str(setup.velocity_path)],
            expected_config_hash=setup.pins[str(setup.velocity_config)],
        )
    ready = OnlineReceivingBoundary(w, "b/")
    physical = []
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
            equal(dict(measured_feedback=fb), feedback_trace, step)
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
            action, sampled = sampling.propose(
                action, ta["NN_reference_nominal"], control=step, standing=standing
            )
            target_a = ep.offset + action[ep.order] * ep.scale
            equal(sampled, sampling_trace, step)
        ta.update(
            control=np.array(step),
            executed_action=action.copy(),
            teacher_delta_HG=delta,
            teacher_authority_clipped=np.array(clipped),
            standing=np.array(standing),
        )
        equal(ta, na, step)
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
        if receiving_player is not None:
            target_b, tb = receiving_player.propose(command, np.r_[anchor, goal[2]], frame=step)
        else:
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
        equal(tb, nb, step)
        row = w.advance({"a/": target_a, "b/": target_b})
        counts["native_steps"] += 10
        counts["actor_calls"] += 4 if receiving_player is not None else 2
        if receiving_player is not None:
            receiving_player.commit()
        if pair is None:
            passer.commit(executed_action=action)
        else:
            counts["shadow_actor_calls"] += 1
            pair.commit(action)
        ready.update()
        stand_controls += int(standing)
        own = {k: row[k] for k in ("qpos", "qvel", "torque", "ball_forces")} | dict(
            body_failed=np.array([row["body_failed"][p] for p in ("a/", "b/")]),
            target_a=target_a.copy(),
            target_b=target_b.copy(),
            time_before=np.array(row["time_before"]),
        )
        equal(own, raw, step)
        physical.append(own)
        if w.failed:
            break
    if any(len(v) != w.control_steps for r in (raw, na, nb) for v in r.values()):
        raise ValueError("all own controls and complete source rows required")
    if feedback_trace is not None and any(
        len(v) != w.control_steps for v in feedback_trace.values()
    ):
        raise ValueError("all own causal feedback observations required")
    if sampling_trace is not None and any(
        len(v) != w.control_steps for v in sampling_trace.values()
    ):
        raise ValueError("complete own stochastic and deterministic recovery rows required")
    ownraw = {k: np.stack([r[k] for r in physical]) for k in physical[0]}
    al = w.layouts["a/"]
    qids = [*range(al.rq, al.rq + 7), *al.qi, *range(al.bq, al.bq + 7)]
    vids = [*range(al.rv, al.rv + 6), *al.vi, *range(al.bv, al.bv + 6)]
    recovery = None
    if stand_controls:
        recovery = stand_from_raw(
            dict(
                qpos=ownraw["qpos"][-stand_controls:, :, qids],
                qvel=ownraw["qvel"][-stand_controls:, :, vids],
                body_failed=ownraw["body_failed"][-stand_controls:, 0],
            )
        )
    measure = pass_transfer_outcome(
        ownraw["qpos"][:, :, al.bq : al.bq + 3].reshape(-1, 3),
        ownraw["qvel"][:, :, al.bv : al.bv + 3].reshape(-1, 3),
        ownraw["ball_forces"].reshape(-1, 2, 3),
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
        actor_calls=(4 if receiving_player is not None else 2) * w.control_steps,
        stand_controls=stand_controls,
        A_recovery=recovery,
        outcome=measure,
        termination="BODY_FAILURE" if w.failed else "FULL_MOTION_PLUS100NN_RECOVERY",
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
    if receiving_player is not None:
        from rosclaw_soccer.rsi.receiving_initialization_protocol import motor_metadata

        result.update(motor_metadata(w.control_steps))
    if result != {k: v for k, v in claim.items() if k != "artifacts"}:
        raise ValueError("own recovery/receiving/delivery/whole counters differ")
    write_once(output / f"trial{index}.json", result)
    from rosclaw_soccer.rsi.pass_trial_logging import trial_progress

    print(json.dumps(trial_progress(result, include_course=emit_course_rows)), flush=True)
    return result, ini
