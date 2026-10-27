"""SIM neural skill-bank relay: actual A kick -> B receive -> B return attempt.

Each preregistered trial runs continuously from one physical initialization.
The role selector is an explicit geometry teacher. Low-level controls are
neural ONLY: existing velocity98 and Dual256; no Cartesian foot overrides.
The optional whole128 reference-teacher ablation keeps zero-bridge controls.
This is a new12s profile, not inherited H300/standing100 or full-game proof.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("frontier-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("frontier-head", "frontier-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--reference-bridge-controls", nargs="+", type=int, default=[0])
    p.add_argument("--successor-alignment-study", action="store_true")
    p.add_argument("--new-context-block", type=int, choices=(0, 1, 2))
    p.add_argument("--balance-teacher-pair", nargs=4, action="append")
    p.add_argument("--balance-expanded", action="store_true")
    p.add_argument("--receiver-follow-study", action="store_true")
    p.add_argument("--receiver-whole-identity", action="store_true")
    p.add_argument("--receiver-exploration", action="store_true")
    p.add_argument("--identity-review", type=Path)
    p.add_argument("--identity-review-hash")
    p.add_argument("--identity-review-head")
    p.add_argument("--balance-pilot-review", type=Path)
    p.add_argument("--balance-pilot-review-hash")
    p.add_argument("--balance-pilot-review-head")
    for key in ("prior-relay-root", "prior-audit-root"):
        p.add_argument("--" + key, type=Path)
    for key in ("prior-relay-hash", "prior-audit-hash"):
        p.add_argument("--" + key)
    a = p.parse_args()
    if a.receiver_exploration and (
        a.new_context_block is None
        or a.balance_teacher_pair is not None
        or a.balance_expanded
        or a.receiver_follow_study
        or a.receiver_whole_identity
        or a.successor_alignment_study
        or a.reference_bridge_controls != [0]
        or not a.identity_review
        or not a.identity_review_hash
        or not a.identity_review_head
    ):
        p.error("receiver exploration requires one isolated block and exact identity review")
    if not a.receiver_exploration and any(
        (a.identity_review, a.identity_review_hash, a.identity_review_head)
    ):
        p.error("identity review inputs belong only to receiver exploration")
    if a.receiver_whole_identity and (
        a.new_context_block is None
        or a.balance_teacher_pair is not None
        or a.balance_expanded
        or a.receiver_follow_study
        or a.successor_alignment_study
        or a.reference_bridge_controls != [0]
    ):
        p.error("whole receiver identity requires one isolated development block")
    if a.receiver_follow_study and (
        a.new_context_block is None
        or a.balance_teacher_pair is not None
        or a.balance_expanded
        or a.successor_alignment_study
        or a.reference_bridge_controls != [0]
    ):
        p.error("post-contact follow requires one isolated development block")
    if a.reference_bridge_controls not in ([0], [0, 5, 15, 30]):
        p.error("entire preregistered reference ablation [0] or [0,5,15,30] required")
    if a.successor_alignment_study and a.reference_bridge_controls != [0]:
        p.error("team-entry ablation must isolate goal/alignment, with reference bridge0")
    if a.new_context_block is not None and (
        a.successor_alignment_study or a.reference_bridge_controls != [0]
    ):
        p.error("development contexts isolate the original neural relay arms")
    if a.balance_teacher_pair is not None and (
        len(a.balance_teacher_pair) != 3 or a.new_context_block is None
    ):
        p.error("balance study requires all three teacher/audit pairs and a development block")
    if (
        a.balance_expanded
        and (
            not a.balance_pilot_review
            or not a.balance_pilot_review_hash
            or not a.balance_pilot_review_head
            or a.balance_teacher_pair is None
        )
    ) or (
        not a.balance_expanded
        and (
            a.balance_pilot_review is not None
            or a.balance_pilot_review_hash is not None
            or a.balance_pilot_review_head is not None
        )
    ):
        p.error("expanded study requires exact closed pilot review and all three teachers")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused relay study root required")
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    if a.receiver_exploration:
        if a.identity_review.is_symlink() or digest(a.identity_review) != a.identity_review_hash:
            raise ValueError("exact independent whole-receiver identity review required")
        identity = json.loads(a.identity_review.read_text())
        if (
            identity.get("status") != "RELAY_WHOLE_RECEIVER_ZERO_IDENTITY_QUALIFIED"
            or identity.get("source_head") != a.identity_review_head
            or identity.get("conditions") != 24
            or len(identity.get("records", [])) != 24
            or identity.get("identity_passed") is not True
            or identity.get("own_physics_and_neural_replay_complete") is not True
            or any(
                identity.get(k) is not False
                for k in ("trained", "promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("all 24 strict zero-identity own-replayed conditions required")
        pins[str(a.identity_review)] = a.identity_review_hash
    if [c["seed"] for c in candidates] != [65023]:
        raise ValueError("same loss-free frozen neural motor required")
    blocks = ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18))
    indices = blocks[a.new_context_block] if a.new_context_block is not None else (1, 4, 20, 21)
    if a.new_context_block is not None:
        frontier = json.loads((a.frontier_root / "complete.json").read_text())
        report = next(r for r in frontier["reports"] if r["seed"] == 65023)
        if not set(indices).issubset(report["achieved_contexts"]):
            raise ValueError("relay contexts must be in the acquired passer TRAIN frontier")
    balance_latents = []
    if a.balance_teacher_pair is not None:
        from rosclaw_soccer.rsi.relay_balance_evidence import admit_relay_balance_teachers

        balance_latents, balance_pins = admit_relay_balance_teachers(a.balance_teacher_pair)
        if set(pins) & set(balance_pins):
            raise ValueError("distinct explicit balance teacher inputs required")
        pins.update(balance_pins)
    if a.balance_expanded:
        review = a.balance_pilot_review
        if review.is_symlink() or digest(review) != a.balance_pilot_review_hash:
            raise ValueError("exact immutable independently replayed pilot review required")
        pilot = json.loads(review.read_text())
        if (
            pilot.get("status") != "RELAY_BALANCE_PAIRED_DEVELOPMENT_NOT_PROMOTED"
            or pilot.get("source_head") != a.balance_pilot_review_head
            or pilot.get("original_baselines_raw_byte_identical") is not True
            or pilot.get("paired_physical_and_private_handoff_prefix_exact") is not True
            or len(pilot.get("records", [])) != 72
            or any(
                pilot.get(k) is not False
                for k in ("promotion_authorized", "hardware_authorized", "independent_fresh")
            )
            or not any(
                row.get("body_failure_rescues", 0) > 0
                for row in pilot.get("summaries", {}).values()
            )
        ):
            raise ValueError("complete paired pilot and at least one measured body rescue required")
        pins[str(review)] = a.balance_pilot_review_hash
    prior = None
    if len(a.reference_bridge_controls) > 1 or a.successor_alignment_study:
        if not all(
            (a.prior_relay_root, a.prior_audit_root, a.prior_relay_hash, a.prior_audit_hash)
        ):
            raise ValueError("reference teacher requires closed independently replayed first relay")
        reports = []
        for root, h in (
            (a.prior_relay_root, a.prior_relay_hash),
            (a.prior_audit_root, a.prior_audit_hash),
        ):
            path = root / "complete.json"
            if path.is_symlink() or (root / "failed.json").exists() or digest(path) != h:
                raise ValueError("exact completed prior relay and own audit required")
            reports.append(json.loads(path.read_text()))
            pins[str(path)] = h
        old, own = reports
        if (
            old["status"] != "NEURAL_RELAY32_COLLECTED_PENDING_OWN_REPLAY"
            or own["status"] != "ALL32_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
            or own["producer_hash"] != a.prior_relay_hash
            or own["counts"] != old["counts"]
            or len(own["records"]) != 32
            or len(old["records"]) != 32
            or any(
                r[k] is not False
                for r in reports
                for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
            )
        ):
            raise ValueError(
                "all32 unpromoted own replay required before expanding reference teacher"
            )
        prior = dict(relay_hash=a.prior_relay_hash, audit_hash=a.prior_audit_hash)
    source_pins(pins)
    check_pins(pins)
    arms = [
        dict(start=s, memory=memory, reference_bridge_controls=bridge)
        for s in (0, 80)
        for memory in ("new_skill_zero", "own_shadow_carry")
        for bridge in a.reference_bridge_controls
    ]
    if a.balance_teacher_pair is not None:
        arms = [
            dict(
                start=80, memory="own_shadow_carry", reference_bridge_controls=0, balance_teacher=i
            )
            for i in (-1, 0, 1, 2)
        ]
    if a.receiver_follow_study:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                receiver_follow_gain=gain,
            )
            for gain in (0.0, 0.25, 0.5, 0.75)
        ]
    if a.receiver_whole_identity:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                receiver_whole=whole,
            )
            for whole in (False, True)
        ]
    if a.receiver_exploration:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                receiver_whole=True,
                exploration_seed=seed,
            )
            for seed in (None, 92117, 92143, 92171)
        ]
    if a.successor_alignment_study:
        arms = [
            arm | dict(assigned_receiver_pass=assigned, align_successor=align)
            for arm in arms
            for assigned in (False, True)
            for align in (False, True)
        ]
    sides = (-0.75, -0.35, 0.35, 0.75) if a.balance_expanded else (-0.55, 0.55)
    courses = [
        dict(context=i, lateral_m=side, arm=arm) for i in indices for side in sides for arm in arms
    ]
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol=(
                "soccer.neural_relay_receiver_exploration.development.v9"
                if a.receiver_exploration
                else "soccer.neural_relay_whole_receiver_identity.development.v8"
                if a.receiver_whole_identity
                else "soccer.neural_relay_post_touch_follow.development.v7"
                if a.receiver_follow_study
                else "soccer.neural_relay_balance_expanded.development.v6"
                if a.balance_expanded
                else "soccer.neural_relay_balance_teacher.development.v5"
                if a.balance_teacher_pair is not None
                else "soccer.neural_relay_development_contexts.v4"
                if a.new_context_block is not None
                else "soccer.neural_relay_team_entry.v3"
                if a.successor_alignment_study
                else "soccer.neural_relay_reference_bridge.v2"
            ),
            source_head=a.expected_head,
            input_file_hashes=pins,
            courses=courses,
            development_context_block=a.new_context_block,
            context_indices=list(indices),
            balance_teacher_pairs=a.balance_teacher_pair,
            balance_pilot_review=str(a.balance_pilot_review) if a.balance_expanded else None,
            balance_pilot_review_hash=a.balance_pilot_review_hash,
            balance_pilot_review_head=a.balance_pilot_review_head,
            identity_review=str(a.identity_review) if a.receiver_exploration else None,
            identity_review_hash=a.identity_review_hash,
            identity_review_head=a.identity_review_head,
            receiver_lateral_offsets_m=list(sides),
            balance_teacher_is_not_learned_actor=a.balance_teacher_pair is not None,
            candidate=candidates[0],
            baseline_training_hash=base_hash,
            max_world_controls=600,
            max_native_steps=6000 * len(courses),
            maximum_actual_actor_calls=1200 * len(courses),
            maximum_shadow_actor_calls=300 * len(courses),
            reference_bridge_controls=a.reference_bridge_controls,
            reference_bridge_is_unlearned_teacher=True,
            successor_geometry_teacher=a.successor_alignment_study,
            assigned_receiver_goal_is_fixed_communicated_forefoot=a.successor_alignment_study,
            aligned_entry_consecutive_measured_controls=10,
            prior_closed_relay=prior,
            joint_targets_remain_actual_neural_outputs=a.balance_teacher_pair is None
            and not a.receiver_exploration,
            bounded_teacher_added_after_neural=a.balance_teacher_pair is not None,
            all_bodies_initialized_once=True,
            no_ball_reposition=True,
            no_future_readiness=True,
            cartesian_teachers_enabled=False,
            geometry_role_selector_is_teacher=True,
            post_touch_velocity_follow_is_unlearned_teacher=a.receiver_follow_study,
            whole_receiver_identity_not_trained=a.receiver_whole_identity,
            receiver_exploration_is_stochastic_not_training=a.receiver_exploration,
            receiver_sampling_profile="contact-v1" if a.receiver_exploration else None,
            receiver_sampling_law="box-truncated-ar1" if a.receiver_exploration else None,
            actual_actor_calls_are_body_proposals_not_auxiliary_forward_passes=True,
            receiver_velocity_original_H300_preserved=True,
            longer_dual_neural_standing_is_NEW_UNQUALIFIED_PROFILE=True,
            return_goal_rule="A CURRENT root+.25forward+.12lateral,z=.11 fixed at handoff",
            outgoing_rule=("native B foot>10N AND toward-goal ball speed>=1m/s after handoff"),
            strict_return_accuracy_m=0.10,
            clean_foot_capture_not_regraded=True,
            optimizer_updates=0,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    counts = dict(native_steps=0, actual_actor_calls=0, shadow_actor_calls=0)
    if a.receiver_whole_identity or a.receiver_exploration:
        counts.update(receiver_onnx_calls=0, receiver_plastic_calls=0, receiver_anchor_calls=0)
    if a.receiver_exploration:
        counts["receiver_stochastic_calls"] = 0
    records = []
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
        from rosclaw_soccer.rsi.neural_relay_incoming import incoming_delivery
        from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
        from rosclaw_soccer.rsi.receiver_exploration_stream import ReceiverExplorationStream
        from rosclaw_soccer.rsi.receiving_handoff_opportunity import body_contact_kind
        from rosclaw_soccer.rsi.receiving_successor_alignment import (
            SuccessorAlignmentGate,
            assigned_receiver_pass_goal,
            successor_alignment,
        )
        from rosclaw_soccer.rsi.relay_balance_teacher import RelayBalanceTeacher
        from rosclaw_soccer.rsi.relay_post_touch_follow import post_touch_follow
        from rosclaw_soccer.rsi.relay_whole_receiver_identity import zero_receiver_actor
        from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters
        from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
        from rosclaw_soccer.rsi.shared_native_receiving_player import SharedNativeReceivingPlayer
        from rosclaw_soccer.rsi.shared_relay_setup import (
            SharedRelaySetup,
            motion_segment,
            receiving_target_for_player,
        )
        from rosclaw_soccer.rsi.shared_shadow_motor import SharedShadowMotor
        from rosclaw_soccer.sim.contracts import hash_json
        from rosclaw_soccer.training.native_pass_courses import NativePassCourse
        from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
        from scripts.rsi_stroke_search_common import open_motor

        bench, base = open_motor(pins, base_hash, {i: contexts[i] for i in indices})
        candidate = candidates[0]
        _, training = receipt(pins, candidate["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            base.parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=Path(candidate["training_root"]),
            expected_receipt_hash=candidate["training_hash"],
            checkpoint_kind=candidate["kind"],
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor, context_indices=indices)
        whole_actor = (
            zero_receiver_actor(setup)
            if a.receiver_whole_identity or a.receiver_exploration
            else None
        )
        (a.output / "scene.xml").write_text(setup.xml)
        typed = []
        for i in indices:
            for side in sides:
                _, w, _, _, _ = setup.prepare(i, side)
                typed.append(
                    NativePassCourse(
                        i,
                        4.0,
                        side,
                        hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())),
                    )
                )
        binding = preflight_receiving_courses(tuple(typed))
        for course in courses:
            ep, w, forward, _, anchor = setup.prepare(course["context"], course["lateral_m"])
            motion_segment(ep.clip, course["arm"]["start"])
            if course["arm"].get("receiver_whole", False):
                if whole_actor is None:
                    raise ValueError("qualified zero whole receiver required before any physics")
                SharedNativeReceivingPlayer(
                    w,
                    "b/",
                    whole_actor,
                    model=setup.velocity_path,
                    config=setup.velocity_config,
                    expected_model_hash=pins[str(setup.velocity_path)],
                    expected_config_hash=pins[str(setup.velocity_config)],
                    horizon_controls=300,
                )
                if course["arm"].get("exploration_seed") is not None:
                    ReceiverExplorationStream(
                        np.asarray(setup.velocity_parameters.action_scale, dtype=np.float32),
                        seed=course["arm"]["exploration_seed"],
                    )
            teacher_index = course["arm"].get("balance_teacher", -1)
            if teacher_index >= 0:
                RelayBalanceTeacher(balance_latents[teacher_index], scale=ep.scale, order=ep.order)
            SuccessorAlignmentGate(enabled=course["arm"].get("align_successor", False))
            if type(course["arm"].get("assigned_receiver_pass", False)) is not bool:
                raise ValueError("explicit assigned-pass experiment arm required before physics")
            if course["arm"].get("assigned_receiver_pass", False):
                assigned_receiver_pass_goal(anchor, forward)
            if w.native_steps:
                raise ValueError("all relay declarations must preflight before physics")
        write_once(
            a.output / "preflight.json",
            dict(
                worlds=len(courses),
                initial_conditions=len(indices) * len(sides),
                course_binding=binding,
                ball_dimensions=setup.ball_dimensions,
                native_steps=0,
            ),
        )

        def store(folder, name, rows):
            if not rows:
                return None
            keys = rows[0].keys()
            path = folder / (name + ".npz")
            np.savez_compressed(path, **{k: np.stack([r[k] for r in rows]) for k in keys})
            return dict(path=str(path.relative_to(a.output)), hash=digest(path))

        for index, course in enumerate(courses):
            ep, w, forward, home, anchor = setup.prepare(course["context"], course["lateral_m"])
            segment = motion_segment(ep.clip, course["arm"]["start"])
            pass_goal = (
                assigned_receiver_pass_goal(anchor, forward)
                if course["arm"].get("assigned_receiver_pass", False)
                else ep.goal
            )
            passer = SharedNativeNeuralPlayer(
                w,
                "a/",
                actor,
                clip=ep.clip,
                torso_index=ep.torso_index,
                build_observation=ep.build_observation,
                goal=pass_goal,
                offset=ep.offset,
                scale=ep.scale,
                order=ep.order,
            )
            velocity = setup.velocity()
            whole_receiver = None
            receiver_sampler = None
            if course["arm"].get("receiver_whole", False):
                if whole_actor is None:
                    raise ValueError("qualified zero whole receiver required")
                whole_receiver = SharedNativeReceivingPlayer(
                    w,
                    "b/",
                    whole_actor,
                    model=setup.velocity_path,
                    config=setup.velocity_config,
                    expected_model_hash=pins[str(setup.velocity_path)],
                    expected_config_hash=pins[str(setup.velocity_config)],
                    horizon_controls=300,
                )
                if course["arm"].get("exploration_seed") is not None:
                    receiver_sampler = ReceiverExplorationStream(
                        np.asarray(setup.velocity_parameters.action_scale, dtype=np.float32),
                        seed=course["arm"]["exploration_seed"],
                    )
            shadow = SharedShadowMotor(
                w,
                "b/",
                actor,
                clip=segment,
                torso_index=ep.torso_index,
                build_observation=ep.build_observation,
                goal=receiving_target_for_player(w, "a/"),
                offset=ep.offset,
                scale=ep.scale,
                order=ep.order,
            )
            readiness = OnlineReceivingBoundary(w, "b/")
            entry = SuccessorAlignmentGate(enabled=course["arm"].get("align_successor", False))
            receiver = None
            balance_teacher = None
            launch = False
            folder = a.output / f"trial{index}"
            folder.mkdir()
            physical, nn_a, nn_b, nn_v, nn_shadow = [], [], [], [], []
            ready = False
            handoff = None
            return_goal = kick_origin = None
            initial_q, initial_v = w.data.qpos.copy(), w.data.qvel.copy()
            excluded_contacts = set()
            termination = "WORLD_HORIZON_COMPLETE"
            previous_target_b = None
            largest_jump = 0.0
            for step in range(600):
                if receiver is None and ready:
                    # This switch is before mj_forward/proposals, at the exact
                    # preceding physical receipt. No mjData field is assigned.
                    profile = w.switch_pd_profile("b/", NativePDParameters(ep.kp, ep.kd, ep.effort))
                    boundary = folder / "handoff.npz"
                    np.savez_compressed(
                        boundary,
                        **{"before_" + k: v for k, v in profile["before"].items()},
                        **{"after_" + k: v for k, v in profile["after"].items()},
                        shadow_h=shadow.state[0].numpy(),
                        shadow_c=shadow.state[1].numpy(),
                        previous_executed_action=shadow.previous_action,
                    )
                    return_goal = receiving_target_for_player(w, "a/")
                    kick_origin = w.data.qpos[w.layouts["b/"].bq : w.layouts["b/"].bq + 3].copy()
                    receiver = shadow.handoff(
                        goal=return_goal,
                        carry_memory=course["arm"]["memory"] == "own_shadow_carry",
                        reference_bridge_controls=course["arm"]["reference_bridge_controls"],
                    )
                    teacher_index = course["arm"].get("balance_teacher", -1)
                    if teacher_index >= 0:
                        balance_teacher = RelayBalanceTeacher(
                            balance_latents[teacher_index], scale=ep.scale, order=ep.order
                        )
                    handoff = dict(
                        control=step,
                        time_s=float(w.data.time),
                        profile={k: v for k, v in profile.items() if k not in ("before", "after")},
                        boundary=dict(
                            path=str(boundary.relative_to(a.output)), hash=digest(boundary)
                        ),
                        return_goal=return_goal.tolist(),
                        kick_origin=kick_origin.tolist(),
                        shadow_observations=shadow.observations,
                        prior_action_carried=True,
                        body_or_ball_reset=False,
                    )
                if receiver is None and step >= 300:
                    termination = "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300"
                    break
                mujoco.mj_forward(w.model, w.data)
                if passer.phase == "MOTION_COMPLETE":
                    passer.begin_standing()
                target_a, trace_a = passer.propose()
                trace_a["control"] = np.array(step)
                counts["actual_actor_calls"] += 1
                if receiver is None:
                    trace_shadow = shadow.observe()
                    trace_shadow["control"] = np.array(step)
                    counts["shadow_actor_calls"] += 1
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
                    )
                    launch = intent.launch_seen
                    command = np.array(intent.command_body, np.float32)
                    if a.receiver_follow_study:
                        command = post_touch_follow(
                            command,
                            root_xy=q[:2],
                            yaw=yaw,
                            ball_xy=q[36:38],
                            ball_velocity_xy=v[35:37],
                            pass_direction=forward,
                            contact_seen=readiness.contact_seen and not readiness.excluded_contact,
                            gain=course["arm"]["receiver_follow_gain"],
                        )
                    if entry.active:
                        command = np.array(
                            successor_alignment(
                                q, v, receiving_target_for_player(w, "a/")
                            ).command_body,
                            np.float32,
                        )
                    if whole_receiver is None:
                        target_b = velocity.propose(
                            q, v, w.layouts["b/"].angular_velocity_body(w.data), command, frame=step
                        )
                        trace_v = dict(
                            control=np.array(step),
                            obs=velocity.last_observation.copy(),
                            action=velocity.last_action.copy(),
                            command=command,
                            phase=np.array(velocity.phase),
                        )
                    else:
                        anchor3 = np.array([anchor[0], anchor[1], 0.11], dtype=np.float64)
                        target_b, trace_v = whole_receiver.propose(command, anchor3, frame=step)
                        if receiver_sampler is not None:
                            from rosclaw_soccer.providers.g1.velocity_policy import target29

                            sampled_action, sample_trace = receiver_sampler.propose(
                                trace_v["action"], trace_v["reference_action"], control=step
                            )
                            target_b = target29(sampled_action, whole_receiver.params)
                            trace_v.update({"SAM_" + k: v for k, v in sample_trace.items()})
                            trace_v["SAM_executed_action"] = sampled_action.copy()
                            counts["receiver_stochastic_calls"] += 1
                    if a.receiver_whole_identity or a.receiver_exploration:
                        counts["receiver_onnx_calls"] += 1
                        if whole_receiver is not None:
                            counts["receiver_plastic_calls"] += 1
                            counts["receiver_anchor_calls"] += 1
                else:
                    if receiver.phase == "MOTION_COMPLETE":
                        receiver.begin_standing()
                    target_b, trace_b = receiver.propose(
                        with_reference_evidence=balance_teacher is not None
                    )
                    if balance_teacher is not None:
                        q, v = w.layouts["b/"].canonical(w.data)
                        action_b, balance_trace = balance_teacher.apply(
                            trace_b["NN_action"], trace_b["NN_reference_nominal"], q, v
                        )
                        target_b = ep.offset + action_b[ep.order] * ep.scale
                        trace_b.update(balance_trace)
                    trace_b["control"] = np.array(step)
                counts["actual_actor_calls"] += 1
                row = w.advance({"a/": target_a, "b/": target_b})
                counts["native_steps"] += 10
                passer.commit()
                if receiver is None:
                    if whole_receiver is not None:
                        whole_receiver.commit(
                            executed_action=sampled_action if receiver_sampler is not None else None
                        )
                        if receiver_sampler is not None:
                            receiver_sampler.commit(sampled_action)
                    shadow.commit_actual()
                    receiving_ready = readiness.update()
                    current_q, current_v = w.layouts["b/"].canonical(w.data)
                    ready = entry.observe(
                        current_q,
                        current_v,
                        receiving_target_for_player(w, "a/"),
                        control=w.control_steps,
                        receiving_ready=receiving_ready,
                        contact_eligible=readiness.contact_seen and not readiness.excluded_contact,
                        body_healthy=not w.failed,
                    )
                    nn_shadow.append(trace_shadow)
                    nn_v.append(trace_v)
                else:
                    receiver.commit(
                        executed_action=action_b if balance_teacher is not None else None
                    )
                    nn_b.append(trace_b)
                nn_a.append(trace_a)
                if previous_target_b is not None:
                    largest_jump = max(
                        largest_jump, float(np.abs(target_b - previous_target_b).max())
                    )
                previous_target_b = target_b.copy()
                for prefix in ("a/", "b/"):
                    for name in w.last_ball_contact_bodies[prefix]:
                        if body_contact_kind(name, prefix=prefix) == "excluded_or_unknown":
                            excluded_contacts.add(name)
                physical.append(
                    {k: row[k] for k in ("qpos", "qvel", "torque", "ball_forces")}
                    | dict(
                        body_failed=np.array([row["body_failed"][p] for p in ("a/", "b/")]),
                        time_before=np.array(row["time_before"]),
                        target_b=np.asarray(target_b, np.float64),
                    )
                )
                if w.failed:
                    termination = "BODY_FAILURE"
                    break
            artifacts = {
                name: store(folder, name, rows)
                for name, rows in (
                    ("physics", physical),
                    ("passer_nn", nn_a),
                    ("receiver_velocity", nn_v),
                    ("receiver_shadow", nn_shadow),
                    ("receiver_return_nn", nn_b),
                )
            }
            initial_path = folder / "initial.npz"
            np.savez_compressed(initial_path, qpos=initial_q, qvel=initial_v)
            artifacts["initial"] = dict(
                path=str(initial_path.relative_to(a.output)), hash=digest(initial_path)
            )
            result = dict(
                handoff_executed=handoff is not None,
                outgoing_foot_strike=False,
                minimum_return_goal_error_3d_m=None,
                return_0p1m_and_healthy=False,
                maximum_B_target_change_rad=largest_jump,
                excluded_contact_bodies=sorted(excluded_contacts),
            )
            if handoff is not None:
                post = physical[handoff["control"] :]
                if post:
                    layout = w.layouts["b/"]
                    q = np.stack([r["qpos"] for r in post]).reshape(-1, w.model.nq)
                    v = np.stack([r["qvel"] for r in post]).reshape(-1, w.model.nv)
                    force = np.stack([r["ball_forces"] for r in post]).reshape(-1, 2, 3)
                    direction = return_goal[:2] - kick_origin[:2]
                    direction /= np.linalg.norm(direction)
                    struck = (force[:, 1, :2].max(axis=1) > 10) & (
                        v[:, layout.bv : layout.bv + 2] @ direction >= 1.0
                    )
                    points = q[:, layout.bq : layout.bq + 3]
                    result["outgoing_foot_strike"] = bool(struck.any())
                    result["minimum_return_goal_error_3d_m"] = float(
                        np.linalg.norm(points - return_goal, axis=1).min()
                    )
                    if struck.any():
                        first = int(np.flatnonzero(struck)[0])
                        error = float(np.linalg.norm(points[first:] - return_goal, axis=1).min())
                        result["minimum_return_goal_error_3d_m"] = error
                        result["return_0p1m_and_healthy"] = bool(
                            error <= 0.1 and not w.failed and not excluded_contacts
                        )
            if a.successor_alignment_study:
                layout = w.layouts["b/"]
                all_q = np.stack([r["qpos"] for r in physical]).reshape(-1, w.model.nq)
                all_forces = np.stack([r["ball_forces"] for r in physical]).reshape(-1, 2, 3)
                result["incoming_delivery"] = incoming_delivery(
                    all_q[:, layout.bq : layout.bq + 3],
                    all_forces,
                    assigned_receiver_pass_goal(anchor, forward),
                )
                result["A_requested_pass_goal"] = pass_goal.tolist()
            record = dict(
                index=index,
                course=course,
                termination=termination,
                controls=w.control_steps,
                native_steps=w.native_steps,
                handoff=handoff,
                result=result,
                artifacts=artifacts,
                original_clean_foot_grade_not_changed=True,
                whole_game_verified=False,
            )
            records.append(record)
            write_once(a.output / f"trial{index}.json", record)
            print(
                json.dumps(
                    dict(
                        index=index,
                        course=course,
                        termination=termination,
                        result=result,
                        counts=counts,
                    )
                ),
                flush=True,
            )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "NEURAL_RELAY_RECEIVER_EXPLORATION32_PENDING_OWN_REPLAY"
                    if a.receiver_exploration
                    else "NEURAL_RELAY_WHOLE_RECEIVER_IDENTITY16_PENDING_OWN_REPLAY"
                    if a.receiver_whole_identity
                    else "NEURAL_RELAY_POST_TOUCH_FOLLOW32_COLLECTED_PENDING_OWN_REPLAY"
                    if a.receiver_follow_study
                    else "NEURAL_RELAY_BALANCE_EXPANDED64_COLLECTED_PENDING_OWN_REPLAY"
                    if a.balance_expanded
                    else "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                    if a.balance_teacher_pair is not None
                    else "NEURAL_RELAY_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                    if a.new_context_block is not None
                    else "NEURAL_RELAY_TEAM_ENTRY_COLLECTED_PENDING_OWN_REPLAY"
                    if a.successor_alignment_study
                    else "NEURAL_RELAY_REFERENCE_BRIDGE_COLLECTED_PENDING_OWN_REPLAY"
                ),
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                counts=counts,
                records=records,
                input_file_hashes=pins,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="NEURAL_RELAY_COLLECTION_FAILED_PRESERVE_NO_RETRY",
                source_head=a.expected_head,
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                completed_records=records,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
