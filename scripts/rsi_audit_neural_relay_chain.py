"""Own-history whole32 relay replay with an independent native PD clock.

Reuses observation/skill wrappers and geometric selector on OWN physics.
Does not execute saved actions, borrow RNN history or call producer advance.
Negative outcomes are mandatory too. Audit equality is NOT football success.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source-head", "source-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    rp, dp = a.source_root / "complete.json", a.source_root / "declaration.json"
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (rp, dp))
        or (a.source_root / "failed.json").exists()
        or digest(rp) != a.source_hash
    ):
        raise ValueError("unused own replay and exact completed source required")
    s, dec = [json.loads(x.read_text()) for x in (rp, dp)]
    projected_memory = (
        dec["protocol"] == "soccer.neural_relay_receiver_projected_action_memory.development.v27"
    )
    leg_limit_pair = projected_memory or (
        dec["protocol"] == "soccer.neural_relay_receiver_leg_target_projection.development.v26"
    )
    contact_lead = dec["protocol"] == "soccer.neural_relay_receiver_contact_lead.development.v29"
    early_intercept = contact_lead or (
        dec["protocol"] == "soccer.neural_relay_receiver_early_intercept.development.v28"
    )
    axis_local_fallback = early_intercept or (
        dec["protocol"] == "soccer.neural_relay_receiver_axis_local_fallback.development.v25"
    )
    predictive_foot_shin = axis_local_fallback or (
        dec["protocol"] == "soccer.neural_relay_receiver_predictive_foot_shin.development.v24"
    )
    foot_teacher_pair = predictive_foot_shin or (
        dec["protocol"] == "soccer.neural_relay_receiver_foot_teacher.development.v23"
    )
    precontact_smooth = dec["protocol"] == "soccer.neural_relay_precontact_smooth.development.v16"
    precontact_fullbody = precontact_smooth or (
        dec["protocol"] == "soccer.neural_relay_precontact_fullbody.development.v15"
    )
    if precontact_smooth and dec.get("precontact_smooth_controls") != 10:
        raise ValueError("exact preregistered smooth transition required")
    center_exploration = (
        dec["protocol"] == "soccer.neural_relay_receiver_center_exploration.development.v14"
    )
    exploration = (
        center_exploration
        or dec["protocol"] == "soccer.neural_relay_receiver_exploration.development.v9"
    )
    learned_exploration = dec["protocol"] in (
        "soccer.neural_relay_receiver_dense_exploration.development.v19",
        "soccer.neural_relay_receiver_motion_exploration.development.v20",
    )
    continuation_pair = (
        dec["protocol"] == "soccer.neural_relay_receiver_continuation_pair.development.v21"
    )
    motion_pair = dec["protocol"] in (
        "soccer.neural_relay_receiver_motion_pair.development.v18",
        "soccer.neural_relay_receiver_motion_exploration.development.v20",
    )
    dense_pair = (
        motion_pair
        or continuation_pair
        or learned_exploration
        or dec["protocol"] == "soccer.neural_relay_receiver_dense_pair.development.v17"
    )
    learned_pair = (
        dense_pair or dec["protocol"] == "soccer.neural_relay_receiver_learned_pair.development.v10"
    )
    contact_exit = dec["protocol"] == "soccer.neural_relay_passer_contact_exit.development.v12"
    passer_pair = (
        contact_exit
        or dec["protocol"] == "soccer.neural_relay_passer_residual_pair.development.v11"
    )
    center_identity = (
        dec["protocol"] == "soccer.neural_relay_receiver_center_identity.development.v13"
    )
    whole_identity = (
        center_identity
        or dec["protocol"] == "soccer.neural_relay_whole_receiver_identity.development.v8"
    )
    follow_study = dec["protocol"] == "soccer.neural_relay_post_touch_follow.development.v7"
    lateral_authority = (
        dec["protocol"] == "soccer.neural_relay_receiver_lateral_authority.development.v22"
    )
    balance_expanded = dec["protocol"] == "soccer.neural_relay_balance_expanded.development.v6"
    balance_study = (
        balance_expanded or dec["protocol"] == "soccer.neural_relay_balance_teacher.development.v5"
    )
    development_study = (
        precontact_fullbody
        or passer_pair
        or learned_pair
        or exploration
        or whole_identity
        or follow_study
        or lateral_authority
        or foot_teacher_pair
        or leg_limit_pair
        or balance_study
        or dec["protocol"] == "soccer.neural_relay_development_contexts.v4"
    )
    reference_study = dec["protocol"] == "soccer.neural_relay_reference_bridge.v2"
    team_entry_study = dec["protocol"] == "soccer.neural_relay_team_entry.v3"
    expected_worlds = (
        32
        if learned_exploration
        else 16
        if whole_identity
        or learned_pair
        or passer_pair
        or precontact_fullbody
        or foot_teacher_pair
        or leg_limit_pair
        else 64
        if balance_expanded
        else 32
    )
    if reference_study:
        if dec["reference_bridge_controls"] not in ([0], [0, 5, 15, 30]):
            raise ValueError("entire declared reference-bridge ablation required")
        expected_worlds *= len(dec["reference_bridge_controls"])
    if team_entry_study:
        if dec["reference_bridge_controls"] != [0]:
            raise ValueError("team-entry study must isolate reference0")
        expected_worlds = 128
    expected_status = (
        "NEURAL_RELAY_RECEIVER_PROJECTED_ACTION_MEMORY16_PENDING_OWN_REPLAY"
        if projected_memory
        else "NEURAL_RELAY_RECEIVER_LEG_TARGET_PROJECTION16_PENDING_OWN_REPLAY"
        if leg_limit_pair
        else "NEURAL_RELAY_RECEIVER_CONTACT_LEAD16_PENDING_OWN_REPLAY"
        if contact_lead
        else "NEURAL_RELAY_RECEIVER_EARLY_INTERCEPT16_PENDING_OWN_REPLAY"
        if early_intercept
        else "NEURAL_RELAY_RECEIVER_AXIS_LOCAL_FALLBACK16_PENDING_OWN_REPLAY"
        if axis_local_fallback
        else "NEURAL_RELAY_RECEIVER_PREDICTIVE_FOOT_SHIN16_PENDING_OWN_REPLAY"
        if predictive_foot_shin
        else "NEURAL_RELAY_RECEIVER_FOOT_TEACHER16_PENDING_OWN_REPLAY"
        if foot_teacher_pair
        else "NEURAL_RELAY_PRECONTACT_FULLBODY16_PENDING_OWN_REPLAY"
        if precontact_fullbody
        else "NEURAL_RELAY_RECEIVER_CENTER_EXPLORATION32_PENDING_OWN_REPLAY"
        if center_exploration
        else "NEURAL_RELAY_RECEIVER_CENTER_IDENTITY16_PENDING_OWN_REPLAY"
        if center_identity
        else "NEURAL_RELAY_PASSER_CONTACT_EXIT16_PENDING_OWN_REPLAY"
        if contact_exit
        else "NEURAL_RELAY_PASSER_RESIDUAL_PAIR16_PENDING_OWN_REPLAY"
        if passer_pair
        else "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY"
        if lateral_authority
        else "NEURAL_RELAY_RECEIVER_CONTINUATION_PAIR16_PENDING_OWN_REPLAY"
        if continuation_pair
        else "NEURAL_RELAY_RECEIVER_LEARNED_EXPLORATION32_PENDING_OWN_REPLAY"
        if learned_exploration
        else "NEURAL_RELAY_RECEIVER_MOTION_PAIR16_PENDING_OWN_REPLAY"
        if motion_pair
        else "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY"
        if dense_pair
        else "NEURAL_RELAY_RECEIVER_LEARNED_PAIR16_PENDING_OWN_REPLAY"
        if learned_pair
        else "NEURAL_RELAY_RECEIVER_EXPLORATION32_PENDING_OWN_REPLAY"
        if exploration
        else "NEURAL_RELAY_WHOLE_RECEIVER_IDENTITY16_PENDING_OWN_REPLAY"
        if whole_identity
        else "NEURAL_RELAY_POST_TOUCH_FOLLOW32_COLLECTED_PENDING_OWN_REPLAY"
        if follow_study
        else "NEURAL_RELAY_BALANCE_EXPANDED64_COLLECTED_PENDING_OWN_REPLAY"
        if balance_expanded
        else "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        if balance_study
        else "NEURAL_RELAY_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        if development_study
        else "NEURAL_RELAY_TEAM_ENTRY_COLLECTED_PENDING_OWN_REPLAY"
        if team_entry_study
        else "NEURAL_RELAY_REFERENCE_BRIDGE_COLLECTED_PENDING_OWN_REPLAY"
        if reference_study
        else "NEURAL_RELAY32_COLLECTED_PENDING_OWN_REPLAY"
    )
    if (
        s["status"] != expected_status
        or s["source_head"] != a.source_head
        or s["declaration_hash"] != digest(dp)
        or dec["protocol"]
        not in (
            "soccer.neural_relay_chain32.v1",
            "soccer.neural_relay_reference_bridge.v2",
            "soccer.neural_relay_team_entry.v3",
            "soccer.neural_relay_development_contexts.v4",
            "soccer.neural_relay_balance_teacher.development.v5",
            "soccer.neural_relay_balance_expanded.development.v6",
            "soccer.neural_relay_post_touch_follow.development.v7",
            "soccer.neural_relay_whole_receiver_identity.development.v8",
            "soccer.neural_relay_receiver_exploration.development.v9",
            "soccer.neural_relay_receiver_learned_pair.development.v10",
            "soccer.neural_relay_receiver_dense_pair.development.v17",
            "soccer.neural_relay_receiver_motion_pair.development.v18",
            "soccer.neural_relay_receiver_dense_exploration.development.v19",
            "soccer.neural_relay_receiver_motion_exploration.development.v20",
            "soccer.neural_relay_receiver_continuation_pair.development.v21",
            "soccer.neural_relay_receiver_lateral_authority.development.v22",
            "soccer.neural_relay_receiver_foot_teacher.development.v23",
            "soccer.neural_relay_receiver_predictive_foot_shin.development.v24",
            "soccer.neural_relay_receiver_axis_local_fallback.development.v25",
            "soccer.neural_relay_receiver_early_intercept.development.v28",
            "soccer.neural_relay_receiver_contact_lead.development.v29",
            "soccer.neural_relay_receiver_leg_target_projection.development.v26",
            "soccer.neural_relay_receiver_projected_action_memory.development.v27",
            "soccer.neural_relay_passer_residual_pair.development.v11",
            "soccer.neural_relay_passer_contact_exit.development.v12",
            "soccer.neural_relay_receiver_center_identity.development.v13",
            "soccer.neural_relay_receiver_center_exploration.development.v14",
            "soccer.neural_relay_precontact_fullbody.development.v15",
            "soccer.neural_relay_precontact_smooth.development.v16",
        )
        or len(s["records"]) != expected_worlds
        or len(dec["courses"]) != expected_worlds
        or dec["optimizer_updates"] != 0
        or any(
            s[k] is not False
            for k in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("all32 exact unpromoted neural relay trials required")
    blocks = ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18))
    if development_study:
        block = dec.get("development_context_block")
        if type(block) is not int or block not in (0, 1, 2):
            raise ValueError("exact preregistered development block required")
        indices = blocks[block]
        if dec.get("context_indices") != list(indices) or dec["reference_bridge_controls"] != [0]:
            raise ValueError("exact development context identity and reference required")
        arms = (
            [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_leg_projection=enabled,
                )
                for enabled in (False, True)
            ]
            if leg_limit_pair
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_foot_teacher_strength=strength,
                )
                for strength in (0.0, 0.5)
            ]
            if foot_teacher_pair
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    precontact_fullbody=enabled,
                )
                for enabled in (False, True)
            ]
            if precontact_fullbody
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_lateral_offset_m=offset,
                )
                for offset in (-0.35, -0.12, 0.12, 0.35)
            ]
            if lateral_authority
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_whole=True,
                    receiver_learned=True,
                    exploration_seed=seed,
                )
                for seed in (None, 92117, 92143, 92171)
            ]
            if learned_exploration
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    passer_residual_fraction=fraction,
                )
                for fraction in (1.0, 0.5)
            ]
            if passer_pair
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_whole=True,
                    receiver_learned=learned,
                )
                for learned in (False, True)
            ]
            if learned_pair
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_whole=True,
                    exploration_seed=seed,
                )
                for seed in (None, 92117, 92143, 92171)
            ]
            if exploration
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_whole=whole,
                )
                for whole in (False, True)
            ]
            if whole_identity
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_follow_gain=gain,
                )
                for gain in (0.0, 0.25, 0.5, 0.75)
            ]
            if follow_study
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    balance_teacher=i,
                )
                for i in (-1, 0, 1, 2)
            ]
            if balance_study
            else [
                dict(start=start, memory=memory, reference_bridge_controls=0)
                for start in (0, 80)
                for memory in ("new_skill_zero", "own_shadow_carry")
            ]
        )
        sides = (
            (-0.2, 0.2)
            if center_identity or center_exploration
            else (-0.35, 0.35)
            if dense_pair or lateral_authority or foot_teacher_pair or leg_limit_pair
            else (-0.75, -0.35, 0.35, 0.75)
            if balance_expanded
            else (-0.55, 0.55)
        )
        if (center_identity or center_exploration) and dec.get(
            "receiver_lateral_offsets_m"
        ) != list(sides):
            raise ValueError("exact center curriculum offsets required")
        if (exploration or learned_exploration) and (
            dec.get("receiver_exploration_is_stochastic_not_training") is not True
            or dec.get("receiver_sampling_profile") != "contact-v1"
            or dec.get("receiver_sampling_law") != "box-truncated-ar1"
            or dec.get("joint_targets_remain_actual_neural_outputs") is not False
            or dec.get("bounded_teacher_added_after_neural") is not False
        ):
            raise ValueError("exact bounded stochastic receiver provenance required")
        if precontact_fullbody and dec.get("precontact_fullbody_is_unlearned_teacher") is not True:
            raise ValueError("explicit unlearned precontact full-body research teacher required")
        if passer_pair and (
            dec.get("passer_residual_is_unlearned_teacher") is not True
            or dec.get("joint_targets_remain_actual_neural_outputs") is not False
            or dec.get("bounded_teacher_added_after_neural") is not False
        ):
            raise ValueError("explicit passer teacher provenance required")
        if whole_identity and (
            dec.get("whole_receiver_identity_not_trained") is not True
            or dec.get("actual_actor_calls_are_body_proposals_not_auxiliary_forward_passes")
            is not True
            or dec.get("joint_targets_remain_actual_neural_outputs") is not True
            or dec.get("bounded_teacher_added_after_neural") is not False
        ):
            raise ValueError("honest zero whole-receiver provenance required")
        if follow_study and (
            dec.get("post_touch_velocity_follow_is_unlearned_teacher") is not True
            or dec.get("joint_targets_remain_actual_neural_outputs") is not True
            or dec.get("bounded_teacher_added_after_neural") is not False
        ):
            raise ValueError("honest post-contact command teacher provenance required")
        if lateral_authority and (
            dec.get("lateral_intercept_is_unlearned_teacher") is not True
            or dec.get("joint_targets_remain_actual_neural_outputs") is not True
            or dec.get("bounded_teacher_added_after_neural") is not False
        ):
            raise ValueError("explicit unlearned lateral command authority required")
        if foot_teacher_pair and (
            dec.get("receiver_foot_teacher_is_unlearned") is not True
            or dec.get("cartesian_teachers_enabled") is not True
            or dec.get("joint_targets_remain_actual_neural_outputs") is not False
            or dec.get("bounded_teacher_added_after_neural") is not True
        ):
            raise ValueError("explicit unlearned foot teacher provenance required")
        if (
            predictive_foot_shin
            and dec.get("receiver_predictive_foot_shin_is_unlearned") is not True
        ):
            raise ValueError("explicit unlearned predictive foot/shin teacher required")
        if axis_local_fallback and dec.get("receiver_axis_local_fallback") is not True:
            raise ValueError("explicit axis-local fallback research contract required")
        if early_intercept and dec.get("receiver_early_foot_intercept_is_unlearned") is not True:
            raise ValueError("explicit unlearned early intercept contract required")
        if contact_lead and dec.get("receiver_early_contact_lead_is_unlearned") is not True:
            raise ValueError("explicit unlearned contact lead contract required")
        if leg_limit_pair and (
            dec.get("receiver_leg_projection_is_unlearned") is not True
            or dec.get("receiver_actor_previous_action_remains_raw") is not (not projected_memory)
            or dec.get("joint_targets_remain_actual_neural_outputs") is not False
            or dec.get("bounded_teacher_added_after_neural") is not True
        ):
            raise ValueError("explicit unlearned target projection and raw action memory required")
        if (
            projected_memory
            and dec.get("receiver_actor_previous_action_reflects_projection") is not True
        ):
            raise ValueError("explicit executed-action memory ablation required")
        if balance_expanded and (
            dec.get("receiver_lateral_offsets_m") != list(sides)
            or dec.get("joint_targets_remain_actual_neural_outputs") is not False
            or dec.get("bounded_teacher_added_after_neural") is not True
        ):
            raise ValueError("exact expanded domain and honest motor provenance required")
        expected_courses = [
            dict(
                context=i,
                lateral_m=side,
                arm=arm,
            )
            for i in indices
            for side in sides
            for arm in arms
        ]
        if dec["courses"] != expected_courses:
            raise ValueError("full fixed development relay factorial required")
    else:
        indices = (1, 4, 20, 21)
    pins = dict(s["input_file_hashes"])
    if learned_pair:
        for key in ("receiver_checkpoint", "receiver_candidate_audit", "receiver_dataset"):
            path = Path(dec[key])
            expected = dec[key + "_hash"]
            if path.is_symlink() or pins.get(str(path)) != expected or digest(path) != expected:
                raise ValueError("exact candidate evidence required for independent replay")
    if exploration:
        review = Path(dec["identity_review"])
        if (
            review.is_symlink()
            or pins.get(str(review)) != dec["identity_review_hash"]
            or digest(review) != dec["identity_review_hash"]
        ):
            raise ValueError("exact pinned zero identity review required")
        identity = json.loads(review.read_text())
        if (
            identity.get("status")
            != (
                "CENTER_RECEIVER_IDENTITY_REVIEWED"
                if center_exploration
                else "RELAY_WHOLE_RECEIVER_ZERO_IDENTITY_QUALIFIED"
            )
            or identity.get("source_head") != dec.get("identity_review_head")
            or identity.get("conditions") != 24
            or len(identity.get("records", [])) != 24
            or identity.get("identity_passed") is not True
            or (
                not center_exploration
                and identity.get("own_physics_and_neural_replay_complete") is not True
            )
            or (
                center_exploration
                and any(r.get("identity_exact") is not True for r in identity.get("records", []))
            )
            or any(
                identity.get(k) is not False
                for k in ("trained", "promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("complete strict own-replayed zero identity review required")
    balance_latents = []
    if balance_study:
        from rosclaw_soccer.rsi.relay_balance_evidence import admit_relay_balance_teachers

        balance_latents, balance_pins = admit_relay_balance_teachers(
            dec.get("balance_teacher_pairs")
        )
        if any(pins.get(path) != value for path, value in balance_pins.items()):
            raise ValueError("audited balance teachers must match frozen producer pins")
    if balance_expanded:
        pilot_path = Path(dec["balance_pilot_review"])
        pilot_hash = dec["balance_pilot_review_hash"]
        if (
            pilot_path.is_symlink()
            or pins.get(str(pilot_path)) != pilot_hash
            or digest(pilot_path) != pilot_hash
        ):
            raise ValueError("exact paired pilot admission must be pinned")
        pilot = json.loads(pilot_path.read_text())
        if (
            pilot.get("status") != "RELAY_BALANCE_PAIRED_DEVELOPMENT_NOT_PROMOTED"
            or pilot.get("source_head") != dec.get("balance_pilot_review_head")
            or pilot.get("paired_physical_and_private_handoff_prefix_exact") is not True
            or pilot.get("original_baselines_raw_byte_identical") is not True
            or len(pilot.get("records", [])) != 72
            or not any(
                row.get("body_failure_rescues", 0) > 0
                for row in pilot.get("summaries", {}).values()
            )
        ):
            raise ValueError("exact closed pilot with measured body rescue required")
    pins[str(rp)], pins[str(dp)] = a.source_hash, digest(dp)
    for i, claim in enumerate(s["records"]):
        if claim["index"] != i or claim["course"] != dec["courses"][i]:
            raise ValueError("full preregistered order required")
        artifacts = list(claim["artifacts"].values())
        if claim["handoff"] is not None:
            artifacts.append(claim["handoff"]["boundary"])
        for artifact in artifacts:
            if artifact is None:
                continue
            path = a.source_root / artifact["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (a.source_root / f"trial{i}").resolve()
                or digest(path) != artifact["hash"]
            ):
                raise ValueError("all exact confined relay raw artifacts required")
            pins[str(path)] = artifact["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_hash=a.source_hash,
            input_file_hashes=pins,
            independent_native_integrator=True,
            shared_neural_codecs_and_role_selector=True,
            source_actions_executed=False,
            borrowed_memory=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(native_steps=0, actual_actor_calls=0, shadow_actor_calls=0)
    if whole_identity or exploration or learned_pair:
        counts.update(receiver_onnx_calls=0, receiver_plastic_calls=0, receiver_anchor_calls=0)
    if exploration or learned_exploration:
        counts["receiver_stochastic_calls"] = 0
    records = []
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
        from rosclaw_soccer.rsi.native_foot_contact_teacher import (
            FootContactTeacherConfig,
            NativeFootContactTeacher,
        )
        from rosclaw_soccer.rsi.native_predictive_foot_shin_teacher import (
            NativePredictiveFootShinTeacher,
        )
        from rosclaw_soccer.rsi.neural_relay_incoming import incoming_delivery
        from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
        from rosclaw_soccer.rsi.passer_residual_blend import (
            PasserContactExitGate,
            blended_passer_action,
        )
        from rosclaw_soccer.rsi.precontact_fullbody_gate import precontact_fullbody_gate
        from rosclaw_soccer.rsi.receiver_exploration_stream import ReceiverExplorationStream
        from rosclaw_soccer.rsi.receiver_target_limit_projection import (
            projected_action_memory,
            receiver_target_limit_projection,
        )
        from rosclaw_soccer.rsi.receiving_handoff_opportunity import body_contact_kind
        from rosclaw_soccer.rsi.receiving_successor_alignment import (
            SuccessorAlignmentGate,
            assigned_receiver_pass_goal,
            successor_alignment,
        )
        from rosclaw_soccer.rsi.relay_audit_clock import RelayAuditClock
        from rosclaw_soccer.rsi.relay_balance_teacher import RelayBalanceTeacher
        from rosclaw_soccer.rsi.relay_post_touch_follow import post_touch_follow
        from rosclaw_soccer.rsi.relay_receiver_candidate import load_research_receiver
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
        from scripts.rsi_stroke_search_common import open_motor

        base = dec["baseline_training_hash"]
        audits = [
            json.loads(Path(path).read_text())
            for path in pins
            if path.endswith("/native-audit/complete.json")
        ]
        audits = [row for row in audits if row.get("training_hash") == base]
        if len(audits) != 1:
            raise ValueError("unique consumed native baseline context witness required")
        native_path, native = receipt(pins, audits[0]["source_hash"])
        context_path = native_path.parent / "declaration.json"
        if digest(context_path) != native["declaration_hash"]:
            raise ValueError("original context declaration mismatch")
        contexts = json.loads(context_path.read_text())["contexts"]
        bench, baseline = open_motor(pins, base, {i: contexts[i] for i in indices})
        candidate = dec["candidate"]
        _, training = receipt(pins, candidate["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            baseline.parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=Path(candidate["training_root"]),
            expected_receipt_hash=candidate["training_hash"],
            checkpoint_kind=candidate["kind"],
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor, context_indices=indices)
        whole_actor = (
            zero_receiver_actor(setup) if whole_identity or exploration or learned_pair else None
        )
        learned_actor = (
            load_research_receiver(
                setup,
                Path(dec["receiver_checkpoint"]),
                dec["receiver_checkpoint_hash"],
                Path(dec["receiver_candidate_audit"]),
                dec["receiver_candidate_audit_hash"],
                Path(dec["receiver_dataset"]),
                dec["receiver_dataset_hash"],
                dense_mode=dense_pair,
                motion_mode=motion_pair,
                continuation_mode=continuation_pair,
            )
            if learned_pair
            else None
        )
        if (a.source_root / "scene.xml").read_text() != setup.xml:
            raise ValueError("scene reconstruction mismatch")

        def load(claim, name):
            artifact = claim["artifacts"][name]
            if artifact is None:
                return {}
            with np.load(a.source_root / artifact["path"], allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def equal_trace(actual, saved, index):
            if set(actual) != set(saved):
                raise ValueError("neural trace keys differ")
            for k, value in actual.items():
                if not np.array_equal(value, saved[k][index]):
                    raise ValueError(f"own neural trace differs {k}:{index}")

        for claim in s["records"]:
            ci, course = claim["index"], claim["course"]
            ep, initialized, forward, home, anchor = setup.prepare(
                course["context"], course["lateral_m"]
            )
            w = RelayAuditClock(initialized)
            raw = load(claim, "physics")
            initial = load(claim, "initial")
            if not np.array_equal(initial["qpos"], w.data.qpos) or not np.array_equal(
                initial["qvel"], w.data.qvel
            ):
                raise ValueError("own original physical initialization differs")
            nn = {
                name: load(claim, name)
                for name in (
                    "passer_nn",
                    "receiver_velocity",
                    "receiver_shadow",
                    "receiver_return_nn",
                )
            }
            segment = motion_segment(ep.clip, course["arm"]["start"])
            kwargs = dict(
                torso_index=ep.torso_index,
                build_observation=ep.build_observation,
                offset=ep.offset,
                scale=ep.scale,
                order=ep.order,
            )
            pass_goal = (
                assigned_receiver_pass_goal(anchor, forward)
                if course["arm"].get("assigned_receiver_pass", False)
                else ep.goal
            )
            passer = SharedNativeNeuralPlayer(
                w, "a/", actor, clip=ep.clip, goal=pass_goal, **kwargs
            )
            shadow = SharedShadowMotor(
                w,
                "b/",
                actor,
                clip=segment,
                goal=receiving_target_for_player(w, "a/"),
                **kwargs,
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
                    learned_actor if course["arm"].get("receiver_learned", False) else whole_actor,
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
            readiness = OnlineReceivingBoundary(w, "b/")
            foot_teacher = (
                NativeFootContactTeacher(
                    FootContactTeacherConfig(
                        strength=course["arm"]["receiver_foot_teacher_strength"]
                    )
                )
                if foot_teacher_pair
                else None
            )
            if predictive_foot_shin and course["arm"]["receiver_foot_teacher_strength"] == 0.5:
                foot_teacher = NativePredictiveFootShinTeacher(
                    axis_local_fallback=axis_local_fallback,
                    early_intercept=early_intercept,
                    contact_lead=contact_lead,
                )
            foot_bodies = [
                w.model.body("b/" + side + "_ankle_roll_link").id for side in ("left", "right")
            ]
            ball_geoms = np.flatnonzero(w.model.geom_bodyid == w.layouts["b/"].ball)
            if foot_teacher is not None and len(ball_geoms) != 1:
                raise ValueError("one qualified soccer ball geometry required")
            entry = SuccessorAlignmentGate(enabled=course["arm"].get("align_successor", False))
            receiver = None
            balance_teacher = None
            ready = launch = False
            passer_foot_seen = False
            passer_exit_gate = PasserContactExitGate()
            handoff_control = None
            transition_start_action = None
            goal = origin = None
            excluded = set()
            last_target = None
            largest_jump = 0.0
            physical = []
            vi = bi = 0
            for step in range(len(raw["qpos"])):
                if receiver is None and ready:
                    boundary = w.switch_pd_profile(
                        "b/", NativePDParameters(ep.kp, ep.kd, ep.effort)
                    )
                    h = claim["handoff"]
                    if h is None or h["control"] != step:
                        raise ValueError("own past-only readiness boundary differs")
                    with np.load(a.source_root / h["boundary"]["path"], allow_pickle=False) as z:
                        for side in ("before", "after"):
                            for k, value in boundary[side].items():
                                if not np.array_equal(value, z[side + "_" + k]):
                                    raise ValueError("own unchanged physical handoff differs")
                        for key, value in dict(
                            shadow_h=shadow.state[0].numpy(),
                            shadow_c=shadow.state[1].numpy(),
                            previous_executed_action=shadow.previous_action,
                        ).items():
                            if not np.array_equal(value, z[key]):
                                raise ValueError("own shadow carry history differs")
                    goal = receiving_target_for_player(w, "a/")
                    layout = w.layouts["b/"]
                    origin = w.data.qpos[layout.bq : layout.bq + 3].copy()
                    if goal.tolist() != h["return_goal"] or origin.tolist() != h["kick_origin"]:
                        raise ValueError("own current return task differs")
                    receiver = shadow.handoff(
                        goal=goal,
                        carry_memory=course["arm"]["memory"] == "own_shadow_carry",
                        reference_bridge_controls=course["arm"].get("reference_bridge_controls", 0),
                    )
                    transition_start_action = shadow.previous_action.copy()
                    teacher_index = course["arm"].get("balance_teacher", -1)
                    if teacher_index >= 0:
                        balance_teacher = RelayBalanceTeacher(
                            balance_latents[teacher_index], scale=ep.scale, order=ep.order
                        )
                    handoff_control = step
                if receiver is None and step >= 300:
                    raise ValueError("source exceeded original velocity H300")
                mujoco.mj_forward(w.model, w.data)
                if passer.phase == "MOTION_COMPLETE":
                    passer.begin_standing()
                fraction = course["arm"].get("passer_residual_fraction", 1.0)
                teacher_arm = fraction != 1.0
                target_a, trace = passer.propose(with_reference_evidence=teacher_arm)
                executed_a = None
                if teacher_arm:
                    active = passer_exit_gate.active if contact_exit else not passer_foot_seen
                    executed_a = blended_passer_action(
                        trace["NN_action"],
                        trace["NN_reference_nominal"],
                        passer.scale,
                        passer.order,
                        fraction=fraction if active else 1.0,
                    )
                    target_a = passer.offset + executed_a[passer.order] * passer.scale
                    trace.update(BLEND_active=np.array(active), BLEND_action=executed_a.copy())
                trace["control"] = np.array(step)
                equal_trace(trace, nn["passer_nn"], step)
                if receiver is None:
                    trace_s = shadow.observe()
                    trace_s["control"] = np.array(step)
                    equal_trace(trace_s, nn["receiver_shadow"], vi)
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
                        lateral_offset_m=course["arm"].get("receiver_lateral_offset_m", 0.12),
                    )
                    launch = intent.launch_seen
                    command = np.array(intent.command_body, np.float32)
                    if follow_study:
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
                    if foot_teacher is not None:
                        if isinstance(foot_teacher, NativePredictiveFootShinTeacher):
                            foot_trace = foot_teacher.propose(
                                model=w.model,
                                snapshot=w.data,
                                layout=w.layouts["b/"],
                                nominal=np.asarray(target_b, np.float64),
                                enabled=passer_foot_seen,
                                frame=step,
                            )
                        else:
                            foot_trace = foot_teacher.propose(
                                model=w.model,
                                snapshot=w.data,
                                layout=w.layouts["b/"],
                                nominal=np.asarray(target_b, np.float64),
                                ball_geom=int(ball_geoms[0]),
                                foot_bodies=foot_bodies,
                                enabled=passer_foot_seen,
                                frame=step,
                            )
                        target_b = foot_trace["target"]
                        trace_v.update({"FOOT_" + k: np.asarray(v) for k, v in foot_trace.items()})
                    if leg_limit_pair:
                        layout = w.layouts["b/"]
                        projection = receiver_target_limit_projection(
                            np.asarray(target_b, np.float64),
                            w.model.jnt_range[layout.joints].copy(),
                            w.model.jnt_limited[layout.joints].astype(np.bool_),
                            enabled=course["arm"]["receiver_leg_projection"],
                        )
                        target_b = projection["target"]
                        trace_v.update({"LIMIT_" + k: np.asarray(v) for k, v in projection.items()})
                        if projected_memory and course["arm"]["receiver_leg_projection"]:
                            velocity.last_action = projected_action_memory(
                                velocity.last_action,
                                target_b,
                                np.asarray(velocity.params.action_offset, np.float64),
                                np.asarray(velocity.params.action_scale, np.float64),
                                projection["changed_axes"],
                            )
                        if projected_memory:
                            trace_v["LIMIT_memory_after"] = velocity.last_action.copy()
                    if whole_identity or exploration or learned_pair:
                        counts["receiver_onnx_calls"] += 1
                        if whole_receiver is not None:
                            counts["receiver_plastic_calls"] += 1
                            counts["receiver_anchor_calls"] += 1
                    equal_trace(trace_v, nn["receiver_velocity"], vi)
                    counts["shadow_actor_calls"] += 1
                    vi += 1
                else:
                    if receiver.phase == "MOTION_COMPLETE":
                        receiver.begin_standing()
                    smooth_index = step - handoff_control
                    smooth_mode = precontact_smooth and course["arm"].get(
                        "precontact_fullbody", False
                    )
                    smooth_active = smooth_mode and smooth_index < 10
                    target_b, trace_b = receiver.propose(
                        with_reference_evidence=balance_teacher is not None or smooth_mode
                    )
                    if smooth_active:
                        from rosclaw_soccer.rsi.receiver_transition_blend import (
                            receiver_transition_action,
                        )

                        action_b = receiver_transition_action(
                            transition_start_action,
                            trace_b["NN_action"],
                            trace_b["NN_reference_nominal"],
                            ep.scale,
                            ep.order,
                            control_index=smooth_index,
                            duration=10,
                        )
                        target_b = ep.offset + action_b[ep.order] * ep.scale
                        trace_b.update(
                            SMOOTH_action=action_b.copy(), SMOOTH_index=np.array(smooth_index)
                        )
                    elif smooth_mode:
                        trace_b.update(
                            SMOOTH_action=trace_b["NN_action"].copy(), SMOOTH_index=np.array(-1)
                        )
                    if balance_teacher is not None:
                        q, v = w.layouts["b/"].canonical(w.data)
                        action_b, balance_trace = balance_teacher.apply(
                            trace_b["NN_action"], trace_b["NN_reference_nominal"], q, v
                        )
                        target_b = ep.offset + action_b[ep.order] * ep.scale
                        trace_b.update(balance_trace)
                    trace_b["control"] = np.array(step)
                    equal_trace(trace_b, nn["receiver_return_nn"], bi)
                    bi += 1
                row = w.advance({"a/": target_a, "b/": target_b})
                counts["native_steps"] += 10
                counts["actual_actor_calls"] += 2
                for key in ("qpos", "qvel", "torque", "ball_forces"):
                    if not np.array_equal(row[key], raw[key][step]):
                        raise ValueError(f"own native trajectory differs {ci}:{step}:{key}")
                if (
                    not np.array_equal(target_b, raw["target_b"][step])
                    or row["time_before"] != raw["time_before"][step]
                    or not np.array_equal(
                        list(row["body_failed"].values()), raw["body_failed"][step]
                    )
                ):
                    raise ValueError("own control, time or health differs")
                passer.commit(executed_action=executed_a if teacher_arm and active else None)
                passer_foot_seen = passer_foot_seen or bool(
                    (row["ball_forces"][:, 0, :2] > 10).any()
                )
                if contact_exit:
                    passer_exit_gate.observe(row["ball_forces"][:, 0, :])
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
                    if course["arm"].get("precontact_fullbody", False):
                        ready = ready or precontact_fullbody_gate(
                            current_q,
                            current_v,
                            passer_foot_seen=passer_foot_seen,
                            receiver_contact_seen=readiness.contact_seen
                            or readiness.excluded_contact,
                            world_healthy=not w.failed,
                        )
                else:
                    receiver.commit(
                        executed_action=action_b
                        if balance_teacher is not None or smooth_active
                        else None
                    )
                    if precontact_fullbody:
                        readiness.update()
                if last_target is not None:
                    largest_jump = max(largest_jump, float(np.abs(target_b - last_target).max()))
                last_target = target_b.copy()
                for prefix, contacts in w.last_ball_contact_bodies.items():
                    excluded.update(
                        name
                        for name in contacts
                        if body_contact_kind(name, prefix=prefix) == "excluded_or_unknown"
                    )
                physical.append(row)
                if w.failed and step != len(raw["qpos"]) - 1:
                    raise ValueError("source continued after body failure")
            termination = (
                "BODY_FAILURE"
                if w.failed
                else "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300"
                if receiver is None
                else "WORLD_HORIZON_COMPLETE"
            )
            if (
                termination != claim["termination"]
                or w.control_steps != claim["controls"]
                or w.native_steps != claim["native_steps"]
                or (not w.failed and w.control_steps != (300 if receiver is None else 600))
            ):
                raise ValueError("own complete outcome horizon differs")
            if (
                (receiver is None and ready and not w.failed)
                or len(nn["passer_nn"]["control"]) != w.control_steps
                or len(nn["receiver_velocity"].get("control", [])) != vi
                or len(nn["receiver_shadow"].get("control", [])) != vi
                or len(nn["receiver_return_nn"].get("control", [])) != bi
            ):
                raise ValueError("complete neural horizons or final online transition differ")
            if receiver_sampler is not None:
                import math

                import torch

                from rosclaw_soccer.rsi.motor_sampling_law import history_motor_logp

                saved_velocity = nn["receiver_velocity"]
                density = (
                    history_motor_logp(
                        "box-truncated-ar1",
                        torch.from_numpy(saved_velocity["action"]),
                        torch.from_numpy(saved_velocity["SAM_executed_action"]),
                        torch.from_numpy(saved_velocity["reference_action"]),
                        torch.from_numpy(
                            np.asarray(setup.velocity_parameters.action_scale, dtype=np.float32)
                        ),
                        torch.full((29,), math.log(0.02), dtype=torch.float32),
                        rho=0.9,
                    )
                    .detach()
                    .numpy()
                )
                if not np.allclose(density, saved_velocity["SAM_logp"], rtol=0, atol=1e-4):
                    raise ValueError(
                        "independent full-history normalized receiver sampling density differs"
                    )
            result = dict(
                handoff_executed=(
                    readiness.first_ready_control is not None
                    if precontact_fullbody
                    else handoff_control is not None
                ),
                outgoing_foot_strike=False,
                minimum_return_goal_error_3d_m=None,
                return_0p1m_and_healthy=False,
                maximum_B_target_change_rad=largest_jump,
                excluded_contact_bodies=sorted(excluded),
            )
            if handoff_control is not None:
                post = physical[handoff_control:]
                if post:
                    layout = w.layouts["b/"]
                    q = np.stack([r["qpos"] for r in post]).reshape(-1, w.model.nq)
                    v = np.stack([r["qvel"] for r in post]).reshape(-1, w.model.nv)
                    f = np.stack([r["ball_forces"] for r in post]).reshape(-1, 2, 3)
                    direction = goal[:2] - origin[:2]
                    direction /= np.linalg.norm(direction)
                    strike = (f[:, 1, :2].max(axis=1) > 10) & (
                        v[:, layout.bv : layout.bv + 2] @ direction >= 1
                    )
                    points = q[:, layout.bq : layout.bq + 3]
                    error = float(np.linalg.norm(points - goal, axis=1).min())
                    result["outgoing_foot_strike"] = bool(strike.any())
                    if strike.any():
                        error = float(
                            np.linalg.norm(
                                points[int(np.flatnonzero(strike)[0]) :] - goal, axis=1
                            ).min()
                        )
                        result["return_0p1m_and_healthy"] = bool(
                            error <= 0.1 and not w.failed and not excluded
                        )
                    result["minimum_return_goal_error_3d_m"] = error
            if team_entry_study:
                layout = w.layouts["b/"]
                all_q = np.stack([r["qpos"] for r in physical]).reshape(-1, w.model.nq)
                all_forces = np.stack([r["ball_forces"] for r in physical]).reshape(-1, 2, 3)
                result["incoming_delivery"] = incoming_delivery(
                    all_q[:, layout.bq : layout.bq + 3],
                    all_forces,
                    assigned_receiver_pass_goal(anchor, forward),
                )
                result["A_requested_pass_goal"] = pass_goal.tolist()
            if result != claim["result"]:
                raise ValueError("own return outcome differs")
            if precontact_fullbody and (
                claim.get("motor_transition_executed") != (handoff_control is not None)
                or claim.get("original_receiving_ready_control") != readiness.first_ready_control
            ):
                raise ValueError("own early transition or actual original reception differs")
            records.append(dict(index=ci, course=course, termination=termination, result=result))
            write_once(a.output / f"trial{ci}.json", records[-1])
            print(json.dumps(dict(index=ci, counts=counts, result=result)), flush=True)
        if counts != s["counts"]:
            raise ValueError("own complete physical/neural counts differ")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH",
                worlds=expected_worlds,
                source_head=a.expected_head,
                producer_hash=a.source_hash,
                input_file_hashes=pins,
                counts=counts,
                records=records,
                independent_native_integrator=True,
                shared_neural_codecs_and_role_selector=True,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="NEURAL_RELAY_OWN_REPLAY_FAILED_NO_RETRY",
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
