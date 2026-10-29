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
    p.add_argument("--receiver-lateral-authority", action="store_true")
    p.add_argument("--receiver-adaptive-intercept", action="store_true")
    p.add_argument("--receiver-coordinated-intercept", action="store_true")
    p.add_argument("--passer-return-pair", action="store_true")
    p.add_argument("--passer-return-align-pair", action="store_true")
    p.add_argument("--passer-successor-pair", action="store_true")
    p.add_argument("--receiver-longitudinal-authority", action="store_true")
    p.add_argument("--receiver-foot-teacher-pair", action="store_true")
    p.add_argument("--receiver-predictive-foot-shin", action="store_true")
    p.add_argument("--receiver-axis-local-fallback", action="store_true")
    p.add_argument("--receiver-early-foot-intercept", action="store_true")
    p.add_argument("--receiver-early-contact-lead", action="store_true")
    p.add_argument("--receiver-leg-limit-pair", action="store_true")
    p.add_argument("--receiver-projection-executed-memory", action="store_true")
    p.add_argument("--receiver-whole-identity", action="store_true")
    p.add_argument("--receiver-center-curriculum", action="store_true")
    p.add_argument("--receiver-learned-pair", action="store_true")
    p.add_argument("--receiver-dense-pair", action="store_true")
    p.add_argument("--receiver-motion-pair", action="store_true")
    p.add_argument("--receiver-continuation-pair", action="store_true")
    p.add_argument("--passer-residual-pair", action="store_true")
    p.add_argument("--passer-strike-teacher-pair", action="store_true")
    p.add_argument("--passer-reference-tempo-pair", action="store_true")
    p.add_argument("--passer-velocity-recovery-pair", action="store_true")
    p.add_argument("--passer-velocity-blend", action="store_true")
    p.add_argument("--precontact-fullbody-pair", action="store_true")
    p.add_argument("--precontact-smooth", action="store_true")
    p.add_argument("--passer-contact-exit", action="store_true")
    for key in ("receiver-checkpoint", "receiver-candidate-audit", "receiver-dataset"):
        p.add_argument("--" + key, type=Path)
    for key in (
        "receiver-checkpoint-hash",
        "receiver-candidate-audit-hash",
        "receiver-dataset-hash",
    ):
        p.add_argument("--" + key)
    p.add_argument("--receiver-exploration", action="store_true")
    p.add_argument("--receiver-learned-exploration", action="store_true")
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
    if a.passer_reference_tempo_pair and (
        a.new_context_block is None
        or a.reference_bridge_controls != [0]
        or any(
            getattr(a, name)
            for name in (
                "successor_alignment_study",
                "balance_teacher_pair",
                "balance_expanded",
                "receiver_follow_study",
                "receiver_lateral_authority",
                "receiver_foot_teacher_pair",
                "receiver_leg_limit_pair",
                "receiver_whole_identity",
                "receiver_center_curriculum",
                "receiver_learned_pair",
                "receiver_dense_pair",
                "receiver_motion_pair",
                "receiver_continuation_pair",
                "passer_residual_pair",
                "passer_strike_teacher_pair",
                "passer_velocity_recovery_pair",
                "precontact_fullbody_pair",
                "passer_contact_exit",
                "receiver_exploration",
                "receiver_learned_exploration",
            )
        )
    ):
        p.error("passer reference tempo requires one isolated development block")
    if a.passer_strike_teacher_pair and (
        a.new_context_block is None
        or a.reference_bridge_controls != [0]
        or any(
            getattr(a, name)
            for name in (
                "successor_alignment_study",
                "balance_teacher_pair",
                "balance_expanded",
                "receiver_follow_study",
                "receiver_lateral_authority",
                "receiver_foot_teacher_pair",
                "receiver_leg_limit_pair",
                "receiver_whole_identity",
                "receiver_center_curriculum",
                "receiver_learned_pair",
                "receiver_dense_pair",
                "receiver_motion_pair",
                "receiver_continuation_pair",
                "passer_residual_pair",
                "passer_velocity_recovery_pair",
                "precontact_fullbody_pair",
                "passer_contact_exit",
                "receiver_exploration",
                "receiver_learned_exploration",
            )
        )
    ):
        p.error("passer strike teacher requires one isolated development block")
    if a.passer_successor_pair and not a.passer_return_align_pair:
        p.error("passer successor requires the return alignment protocol")
    if a.passer_return_align_pair and not a.passer_return_pair:
        p.error("return alignment requires the passer return pair")
    if a.passer_return_pair and not (
        a.receiver_coordinated_intercept
        and not a.passer_velocity_recovery_pair
        and not a.passer_velocity_blend
    ):
        p.error("return pair requires the isolated coordinated receiver protocol")
    if a.receiver_coordinated_intercept and not (
        a.receiver_foot_teacher_pair
        and a.receiver_predictive_foot_shin
        and a.receiver_axis_local_fallback
        and a.receiver_early_foot_intercept
        and a.receiver_early_contact_lead
        and not a.receiver_adaptive_intercept
    ):
        p.error("coordinated intercept requires the isolated early-contact foot teacher pair")
    if a.receiver_adaptive_intercept and (
        not a.receiver_lateral_authority or a.receiver_longitudinal_authority
    ):
        p.error("adaptive intercept requires isolated lateral authority")
    if a.receiver_longitudinal_authority and not a.receiver_lateral_authority:
        p.error("longitudinal authority requires the isolated approach authority study")
    if a.passer_velocity_blend and not a.passer_velocity_recovery_pair:
        p.error("passer velocity blend requires the paired recovery study")
    if a.passer_velocity_recovery_pair and (
        a.new_context_block is None
        or a.receiver_foot_teacher_pair
        or a.receiver_leg_limit_pair
        or a.receiver_lateral_authority
        or a.receiver_follow_study
        or a.receiver_whole_identity
        or a.receiver_exploration
        or a.receiver_learned_pair
        or a.passer_residual_pair
        or a.precontact_fullbody_pair
        or a.balance_teacher_pair is not None
        or a.balance_expanded
        or a.successor_alignment_study
        or a.receiver_center_curriculum
        or a.reference_bridge_controls != [0]
    ):
        p.error("passer velocity recovery requires one isolated development block")
    if a.receiver_early_contact_lead and not a.receiver_early_foot_intercept:
        p.error("contact lead requires the early foot intercept study")
    if a.receiver_early_foot_intercept and not (
        a.receiver_predictive_foot_shin and a.receiver_axis_local_fallback
    ):
        p.error("early intercept requires the predictive axis-local foot teacher")
    if a.receiver_projection_executed_memory and not a.receiver_leg_limit_pair:
        p.error("projected executed-action memory requires the isolated leg-limit pair")
    if a.receiver_leg_limit_pair and (
        a.new_context_block is None
        or a.receiver_foot_teacher_pair
        or a.receiver_lateral_authority
        or a.receiver_follow_study
        or a.receiver_whole_identity
        or a.receiver_exploration
        or a.receiver_learned_pair
        or a.passer_residual_pair
        or a.precontact_fullbody_pair
        or a.balance_teacher_pair is not None
        or a.balance_expanded
        or a.successor_alignment_study
        or a.receiver_center_curriculum
        or a.reference_bridge_controls != [0]
    ):
        p.error("leg-limit projection requires an isolated development block")
    if a.receiver_axis_local_fallback and not a.receiver_predictive_foot_shin:
        p.error("axis-local fallback requires the predictive foot/shin study")
    if a.receiver_predictive_foot_shin and not a.receiver_foot_teacher_pair:
        p.error("predictive foot/shin mode requires the isolated foot teacher pair")
    if a.receiver_foot_teacher_pair and (
        a.new_context_block is None
        or a.receiver_lateral_authority
        or a.receiver_follow_study
        or a.receiver_whole_identity
        or a.receiver_exploration
        or a.receiver_learned_pair
        or a.passer_residual_pair
        or a.precontact_fullbody_pair
        or a.balance_teacher_pair is not None
        or a.balance_expanded
        or a.successor_alignment_study
        or a.receiver_center_curriculum
        or a.reference_bridge_controls != [0]
    ):
        p.error("foot teacher pair requires one isolated development block")
    if a.receiver_lateral_authority and (
        a.new_context_block is None
        or a.receiver_follow_study
        or a.receiver_whole_identity
        or a.receiver_exploration
        or a.receiver_learned_pair
        or a.passer_residual_pair
        or a.precontact_fullbody_pair
        or a.balance_teacher_pair is not None
        or a.balance_expanded
        or a.successor_alignment_study
        or a.receiver_center_curriculum
        or a.reference_bridge_controls != [0]
    ):
        p.error("lateral authority requires one isolated development block")
    if a.receiver_continuation_pair and (
        not a.receiver_dense_pair or a.receiver_motion_pair or a.receiver_learned_exploration
    ):
        p.error("continuation evaluation requires an isolated dense learned pair")
    if a.receiver_learned_exploration and not a.receiver_dense_pair:
        p.error("learned exploration requires the pinned dense learned candidate")
    if a.receiver_motion_pair and not a.receiver_dense_pair:
        p.error("motion candidate requires explicit dense-pair evaluation")
    if a.receiver_dense_pair and not a.receiver_learned_pair:
        p.error("dense research candidate requires the isolated learned pair")
    if a.precontact_smooth and not a.precontact_fullbody_pair:
        p.error("smooth transition requires the isolated precontact pair")
    if a.precontact_fullbody_pair and (
        a.new_context_block is None
        or a.passer_residual_pair
        or a.receiver_whole_identity
        or a.receiver_exploration
        or a.receiver_learned_pair
        or a.receiver_follow_study
        or a.balance_teacher_pair is not None
        or a.balance_expanded
        or a.successor_alignment_study
        or a.reference_bridge_controls != [0]
    ):
        p.error("precontact full-body pair requires one isolated development block")
    if a.passer_contact_exit and not a.passer_residual_pair:
        p.error("contact-exit is only defined for the isolated passer pair")
    if a.receiver_center_curriculum and not (a.receiver_whole_identity or a.receiver_exploration):
        p.error("center curriculum requires the receiver identity pair or audited exploration")
    if a.passer_residual_pair and (
        a.new_context_block is None
        or a.balance_teacher_pair is not None
        or a.balance_expanded
        or a.receiver_follow_study
        or a.receiver_whole_identity
        or a.receiver_learned_pair
        or a.receiver_exploration
        or a.successor_alignment_study
        or a.reference_bridge_controls != [0]
    ):
        p.error("passer residual pair requires one isolated development block")
    candidate_args = (
        a.receiver_checkpoint,
        a.receiver_checkpoint_hash,
        a.receiver_candidate_audit,
        a.receiver_candidate_audit_hash,
        a.receiver_dataset,
        a.receiver_dataset_hash,
    )
    if a.receiver_learned_pair and (
        not all(candidate_args)
        or a.new_context_block is None
        or a.balance_teacher_pair is not None
        or a.balance_expanded
        or a.receiver_follow_study
        or a.receiver_whole_identity
        or a.receiver_exploration
        or a.successor_alignment_study
        or a.reference_bridge_controls != [0]
    ):
        p.error("learned pair requires its own block and all candidate evidence")
    if not a.receiver_learned_pair and any(candidate_args):
        p.error("candidate evidence requires learned-pair mode")
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
    from rosclaw_soccer.rsi.shared_native_receive_teacher import SharedNativeReceiveTeacher
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused relay study root required")
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    if a.receiver_learned_pair:
        for path, expected in (
            (a.receiver_checkpoint, a.receiver_checkpoint_hash),
            (a.receiver_candidate_audit, a.receiver_candidate_audit_hash),
            (a.receiver_dataset, a.receiver_dataset_hash),
        ):
            if path.is_symlink() or digest(path) != expected:
                raise ValueError("exact candidate evidence required before physics")
            pins[str(path)] = expected
    if a.receiver_exploration:
        if a.identity_review.is_symlink() or digest(a.identity_review) != a.identity_review_hash:
            raise ValueError("exact independent whole-receiver identity review required")
        identity = json.loads(a.identity_review.read_text())
        if (
            identity.get("status")
            != (
                "CENTER_RECEIVER_IDENTITY_REVIEWED"
                if a.receiver_center_curriculum
                else "RELAY_WHOLE_RECEIVER_ZERO_IDENTITY_QUALIFIED"
            )
            or identity.get("source_head") != a.identity_review_head
            or identity.get("conditions") != 24
            or len(identity.get("records", [])) != 24
            or identity.get("identity_passed") is not True
            or (
                not a.receiver_center_curriculum
                and identity.get("own_physics_and_neural_replay_complete") is not True
            )
            or (
                a.receiver_center_curriculum
                and any(r.get("identity_exact") is not True for r in identity.get("records", []))
            )
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
    if a.passer_residual_pair:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                passer_residual_fraction=fraction,
            )
            for fraction in (1.0, 0.5)
        ]
    if a.passer_strike_teacher_pair:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                passer_strike_teacher=enabled,
            )
            for enabled in (False, True)
        ]
    if a.passer_reference_tempo_pair:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                passer_reference_rate=rate,
            )
            for rate in (1.0, 0.8)
        ]
    if a.precontact_fullbody_pair:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                precontact_fullbody=enabled,
            )
            for enabled in (False, True)
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
    if a.receiver_lateral_authority:
        arms = (
            [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_lateral_offset_m=offset,
                    receiver_adaptive_intercept=adaptive,
                )
                for offset in (-0.12, 0.12)
                for adaptive in (False, True)
            ]
            if a.receiver_adaptive_intercept
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_lateral_offset_m=0.12,
                    receiver_longitudinal_offset_m=offset,
                )
                for offset in (0.1, 0.35, 0.6, 0.8)
            ]
            if a.receiver_longitudinal_authority
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_lateral_offset_m=offset,
                )
                for offset in (-0.35, -0.12, 0.12, 0.35)
            ]
        )
    if a.receiver_foot_teacher_pair:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                receiver_foot_teacher_strength=strength,
                **(
                    {"receiver_adaptive_intercept": bool(strength)}
                    if a.receiver_coordinated_intercept
                    else {}
                ),
            )
            for strength in (0.0, 0.5)
        ]
    if a.passer_return_pair:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                receiver_foot_teacher_strength=0.5,
                receiver_adaptive_intercept=True,
                passer_velocity_recovery=True if a.passer_return_align_pair else enabled,
                **(
                    {"passer_return_align": True if a.passer_successor_pair else enabled}
                    if a.passer_return_align_pair
                    else {}
                ),
                **({"passer_successor": enabled} if a.passer_successor_pair else {}),
            )
            for enabled in (False, True)
        ]
    if a.receiver_leg_limit_pair:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                receiver_leg_projection=enabled,
            )
            for enabled in (False, True)
        ]
    if a.passer_velocity_recovery_pair:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                passer_velocity_recovery=enabled,
            )
            for enabled in (False, True)
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
    if a.receiver_learned_pair:
        arms = [
            dict(
                start=80,
                memory="own_shadow_carry",
                reference_bridge_controls=0,
                receiver_whole=True,
                receiver_learned=learned,
            )
            for learned in (False, True)
        ]
    if a.receiver_learned_exploration:
        arms = [
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
    sides = (
        (-0.2, 0.2)
        if a.receiver_center_curriculum
        else (-0.35, 0.35)
        if a.receiver_dense_pair
        or a.receiver_lateral_authority
        or a.receiver_foot_teacher_pair
        or a.receiver_leg_limit_pair
        or a.passer_velocity_recovery_pair
        else (-0.75, -0.35, 0.35, 0.75)
        if a.balance_expanded
        else (-0.55, 0.55)
    )
    courses = [
        dict(context=i, lateral_m=side, arm=arm) for i in indices for side in sides for arm in arms
    ]
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol=(
                "soccer.neural_relay_passer_successor.development.v37"
                if a.passer_successor_pair
                else "soccer.neural_relay_passer_return_alignment.development.v36"
                if a.passer_return_align_pair
                else "soccer.neural_relay_passer_return_pair.development.v35"
                if a.passer_return_pair
                else "soccer.neural_relay_passer_reference_tempo.development.v1"
                if a.passer_reference_tempo_pair
                else "soccer.neural_relay_passer_strike_torque.development.v1"
                if a.passer_strike_teacher_pair
                else "soccer.neural_relay_passer_velocity_blend.development.v31"
                if a.passer_velocity_blend
                else "soccer.neural_relay_passer_velocity_recovery.development.v30"
                if a.passer_velocity_recovery_pair
                else "soccer.neural_relay_receiver_projected_action_memory.development.v27"
                if a.receiver_projection_executed_memory
                else "soccer.neural_relay_receiver_leg_target_projection.development.v26"
                if a.receiver_leg_limit_pair
                else "soccer.neural_relay_receiver_coordinated_intercept.development.v34"
                if a.receiver_coordinated_intercept
                else "soccer.neural_relay_receiver_contact_lead.development.v29"
                if a.receiver_early_contact_lead
                else "soccer.neural_relay_receiver_early_intercept.development.v28"
                if a.receiver_early_foot_intercept
                else "soccer.neural_relay_receiver_axis_local_fallback.development.v25"
                if a.receiver_axis_local_fallback
                else "soccer.neural_relay_receiver_predictive_foot_shin.development.v24"
                if a.receiver_predictive_foot_shin
                else "soccer.neural_relay_receiver_foot_teacher.development.v23"
                if a.receiver_foot_teacher_pair
                else "soccer.neural_relay_precontact_smooth.development.v16"
                if a.precontact_smooth
                else "soccer.neural_relay_precontact_fullbody.development.v15"
                if a.precontact_fullbody_pair
                else "soccer.neural_relay_receiver_center_exploration.development.v14"
                if a.receiver_center_curriculum and a.receiver_exploration
                else "soccer.neural_relay_receiver_center_identity.development.v13"
                if a.receiver_center_curriculum
                else "soccer.neural_relay_passer_contact_exit.development.v12"
                if a.passer_contact_exit
                else "soccer.neural_relay_passer_residual_pair.development.v11"
                if a.passer_residual_pair
                else "soccer.neural_relay_receiver_longitudinal_authority.development.v32"
                if a.receiver_longitudinal_authority
                else "soccer.neural_relay_receiver_adaptive_intercept.development.v33"
                if a.receiver_adaptive_intercept
                else "soccer.neural_relay_receiver_lateral_authority.development.v22"
                if a.receiver_lateral_authority
                else "soccer.neural_relay_receiver_continuation_pair.development.v21"
                if a.receiver_continuation_pair
                else "soccer.neural_relay_receiver_motion_exploration.development.v20"
                if a.receiver_learned_exploration and a.receiver_motion_pair
                else "soccer.neural_relay_receiver_dense_exploration.development.v19"
                if a.receiver_learned_exploration
                else "soccer.neural_relay_receiver_motion_pair.development.v18"
                if a.receiver_motion_pair
                else "soccer.neural_relay_receiver_dense_pair.development.v17"
                if a.receiver_dense_pair
                else "soccer.neural_relay_receiver_learned_pair.development.v10"
                if a.receiver_learned_pair
                else "soccer.neural_relay_receiver_exploration.development.v9"
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
            receiver_checkpoint=str(a.receiver_checkpoint) if a.receiver_learned_pair else None,
            receiver_checkpoint_hash=a.receiver_checkpoint_hash,
            receiver_candidate_audit=str(a.receiver_candidate_audit)
            if a.receiver_learned_pair
            else None,
            receiver_candidate_audit_hash=a.receiver_candidate_audit_hash,
            receiver_dataset=str(a.receiver_dataset) if a.receiver_learned_pair else None,
            receiver_dataset_hash=a.receiver_dataset_hash,
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
            and not a.receiver_exploration
            and not a.receiver_learned_exploration
            and not a.passer_residual_pair
            and not a.precontact_smooth
            and not a.receiver_foot_teacher_pair
            and not a.receiver_leg_limit_pair,
            # Projected targets deliberately differ from raw actor outputs;
            # the velocity actor's observation retains its original raw action.
            bounded_teacher_added_after_neural=a.balance_teacher_pair is not None
            or a.receiver_foot_teacher_pair
            or a.receiver_leg_limit_pair
            or a.passer_strike_teacher_pair,
            all_bodies_initialized_once=True,
            no_ball_reposition=True,
            no_future_readiness=True,
            cartesian_teachers_enabled=a.receiver_foot_teacher_pair or a.passer_strike_teacher_pair,
            passer_strike_teacher_is_unlearned=a.passer_strike_teacher_pair,
            passer_reference_tempo_is_unlearned=a.passer_reference_tempo_pair,
            receiver_foot_teacher_is_unlearned=a.receiver_foot_teacher_pair,
            receiver_predictive_foot_shin_is_unlearned=a.receiver_predictive_foot_shin,
            receiver_axis_local_fallback=a.receiver_axis_local_fallback,
            receiver_early_foot_intercept_is_unlearned=a.receiver_early_foot_intercept,
            receiver_early_contact_lead_is_unlearned=a.receiver_early_contact_lead,
            receiver_leg_projection_is_unlearned=a.receiver_leg_limit_pair,
            receiver_actor_previous_action_remains_raw=(
                a.receiver_leg_limit_pair and not a.receiver_projection_executed_memory
            ),
            receiver_actor_previous_action_reflects_projection=a.receiver_projection_executed_memory,
            geometry_role_selector_is_teacher=True,
            post_touch_velocity_follow_is_unlearned_teacher=a.receiver_follow_study,
            lateral_intercept_is_unlearned_teacher=a.receiver_lateral_authority,
            adaptive_intercept_is_unlearned_teacher=a.receiver_adaptive_intercept,
            coordinated_intercept_is_unlearned_teacher=a.receiver_coordinated_intercept,
            longitudinal_intercept_is_unlearned_teacher=a.receiver_longitudinal_authority,
            whole_receiver_identity_not_trained=a.receiver_whole_identity,
            receiver_exploration_is_stochastic_not_training=(
                a.receiver_exploration or a.receiver_learned_exploration
            ),
            passer_residual_is_unlearned_teacher=a.passer_residual_pair,
            passer_velocity_recovery_pair=a.passer_velocity_recovery_pair or a.passer_return_pair,
            passer_velocity_blend_controls=20
            if a.passer_velocity_blend or a.passer_return_pair
            else 0,
            passer_return_role_is_unlearned=a.passer_return_pair,
            passer_return_alignment_is_unlearned=a.passer_return_align_pair,
            passer_successor_selection_is_unlearned=a.passer_successor_pair,
            passer_successor_direct_profile_switch=a.passer_successor_pair,
            precontact_fullbody_is_unlearned_teacher=a.precontact_fullbody_pair,
            precontact_smooth_controls=10 if a.precontact_smooth else 0,
            receiver_sampling_profile="contact-v1"
            if a.receiver_exploration or a.receiver_learned_exploration
            else None,
            receiver_sampling_law="box-truncated-ar1"
            if a.receiver_exploration or a.receiver_learned_exploration
            else None,
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
    if a.passer_velocity_recovery_pair or a.passer_return_pair:
        counts["passer_velocity_inferences"] = 0
    if a.passer_velocity_blend or a.passer_return_pair:
        counts["passer_blend_actor_inferences"] = 0
    if a.passer_successor_pair:
        counts["passer_successor_shadow_inferences"] = 0
    if a.receiver_whole_identity or a.receiver_exploration or a.receiver_learned_pair:
        counts.update(receiver_onnx_calls=0, receiver_plastic_calls=0, receiver_anchor_calls=0)
    if a.receiver_exploration or a.receiver_learned_exploration:
        counts["receiver_stochastic_calls"] = 0
    records = []
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.live_neural_transition import blend_live_targets, quintic_weight
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
        from rosclaw_soccer.rsi.paid_velocity_shadow import policy_action_from_hg_target
        from rosclaw_soccer.rsi.passer_residual_blend import (
            PasserContactExitGate,
            blended_passer_action,
        )
        from rosclaw_soccer.rsi.passer_velocity_recovery import PasserVelocityRecovery
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
        from rosclaw_soccer.rsi.relay_balance_teacher import RelayBalanceTeacher
        from rosclaw_soccer.rsi.relay_post_touch_follow import post_touch_follow
        from rosclaw_soccer.rsi.relay_receiver_candidate import load_research_receiver
        from rosclaw_soccer.rsi.relay_whole_receiver_identity import zero_receiver_actor
        from rosclaw_soccer.rsi.return_role_intent import return_receive_command, return_strike_seen
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
            if a.receiver_whole_identity or a.receiver_exploration or a.receiver_learned_pair
            else None
        )
        learned_actor = (
            load_research_receiver(
                setup,
                a.receiver_checkpoint,
                a.receiver_checkpoint_hash,
                a.receiver_candidate_audit,
                a.receiver_candidate_audit_hash,
                a.receiver_dataset,
                a.receiver_dataset_hash,
                dense_mode=a.receiver_dense_pair,
                motion_mode=a.receiver_motion_pair,
                continuation_mode=a.receiver_continuation_pair,
            )
            if a.receiver_learned_pair
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
            if course["arm"].get("passer_residual_fraction") == 0.5:
                probe = SharedNativeNeuralPlayer(
                    w,
                    "a/",
                    actor,
                    clip=ep.clip,
                    torso_index=ep.torso_index,
                    build_observation=ep.build_observation,
                    goal=ep.goal,
                    offset=ep.offset,
                    scale=ep.scale,
                    order=ep.order,
                )
                _, probe_trace = probe.propose(with_reference_evidence=True)
                blended_passer_action(
                    probe_trace["NN_action"],
                    probe_trace["NN_reference_nominal"],
                    probe.scale,
                    probe.order,
                    fraction=0.5,
                )
                if w.native_steps != 0 or w.control_steps != 0:
                    raise ValueError("passer preflight must not integrate or mutate physics")
            if course["arm"].get("receiver_whole", False):
                if whole_actor is None:
                    raise ValueError("qualified zero whole receiver required before any physics")
                SharedNativeReceivingPlayer(
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
                reference_rate=course["arm"].get("passer_reference_rate", 1.0),
            )
            velocity = setup.velocity()
            passer_recovery = None
            if course["arm"].get("passer_velocity_recovery", False):
                stream = ContinuousVelocityStream(
                    setup.velocity_path,
                    setup.velocity_config,
                    expected_model_hash=pins[str(setup.velocity_path)],
                    expected_config_hash=pins[str(setup.velocity_config)],
                    horizon_controls=600,
                )
                passer_recovery = PasserVelocityRecovery(w, "a/", stream)
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
            successor_shadow = (
                SharedShadowMotor(
                    w,
                    "a/",
                    actor,
                    clip=segment,
                    torso_index=ep.torso_index,
                    build_observation=ep.build_observation,
                    goal=receiving_target_for_player(w, "b/"),
                    offset=ep.offset,
                    scale=ep.scale,
                    order=ep.order,
                )
                if a.passer_successor_pair
                else None
            )
            readiness = OnlineReceivingBoundary(w, "b/")
            foot_teacher = (
                NativeFootContactTeacher(
                    FootContactTeacherConfig(
                        strength=course["arm"]["receiver_foot_teacher_strength"]
                    )
                )
                if a.receiver_foot_teacher_pair
                else None
            )
            if (
                a.receiver_predictive_foot_shin
                and course["arm"]["receiver_foot_teacher_strength"] == 0.5
            ):
                foot_teacher = NativePredictiveFootShinTeacher(
                    axis_local_fallback=a.receiver_axis_local_fallback,
                    early_intercept=a.receiver_early_foot_intercept,
                    contact_lead=a.receiver_early_contact_lead,
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
            launch = False
            passer_foot_seen = False
            return_seen = False
            return_readiness = None
            return_alignment_active = False
            successor_motor = None
            successor_ready_count = 0
            successor_handoff = None
            successor_previous_target = None
            passer_exit_gate = PasserContactExitGate()
            passer_strike_teacher = None
            folder = a.output / f"trial{index}"
            folder.mkdir()
            physical, nn_a, nn_b, nn_v, nn_shadow, nn_passer_velocity = [], [], [], [], [], []
            nn_successor_shadow = []
            ready = False
            handoff = None
            return_goal = kick_origin = None
            initial_q, initial_v = w.data.qpos.copy(), w.data.qvel.copy()
            excluded_contacts = set()
            termination = "WORLD_HORIZON_COMPLETE"
            previous_target_b = None
            passer_recovery_boundary = None
            passer_old_pd = passer_new_pd = None
            transition_start_action = None
            largest_jump = 0.0
            for step in range(600):
                if (
                    course["arm"].get("passer_successor", False)
                    and successor_motor is None
                    and successor_ready_count >= 10
                ):
                    assert successor_shadow is not None
                    successor_previous_target = w.last_applied_target("a/").copy()
                    profile_s = w.switch_pd_profile(
                        "a/", NativePDParameters(ep.kp, ep.kd, ep.effort)
                    )
                    goal_s = receiving_target_for_player(w, "b/")
                    successor_boundary = folder / "passer_successor_boundary.npz"
                    np.savez_compressed(
                        successor_boundary,
                        **{"before_" + k: v for k, v in profile_s["before"].items()},
                        **{"after_" + k: v for k, v in profile_s["after"].items()},
                        shadow_h=successor_shadow.state[0].numpy(),
                        shadow_c=successor_shadow.state[1].numpy(),
                        previous_executed_action=successor_shadow.previous_action,
                    )
                    successor_motor = successor_shadow.handoff(goal=goal_s, carry_memory=True)
                    successor_handoff = dict(
                        control=step,
                        goal=goal_s.tolist(),
                        shadow_observations=successor_shadow.observations,
                        profile={
                            k: v for k, v in profile_s.items() if k not in ("before", "after")
                        },
                        boundary=dict(
                            path=str(successor_boundary.relative_to(a.output)),
                            hash=digest(successor_boundary),
                        ),
                    )
                successor_shadow_trace = None
                if successor_shadow is not None and successor_motor is None:
                    successor_shadow_trace = successor_shadow.observe()
                    successor_shadow_trace["control"] = np.array(step)
                    counts["passer_successor_shadow_inferences"] += 1
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
                    transition_start_action = shadow.previous_action.copy()
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
                if a.passer_return_pair and passer.phase == "MOTION_COMPLETE":
                    passer.begin_standing()
                if (
                    (passer.phase == "MOTION_COMPLETE" and not a.passer_return_pair)
                    or (
                        a.passer_return_pair
                        and passer_recovery is not None
                        and passer.phase == "STANDING"
                        and return_seen
                    )
                ) and not (passer_recovery is not None and passer_recovery.active):
                    if passer_recovery is not None and passer_foot_seen:
                        vp = passer_recovery.stream.params
                        passer_old_pd = w.parameters["a/"].owned_copy()
                        passer_new_pd = NativePDParameters(
                            np.asarray(vp.stiffness, np.float64),
                            np.asarray(vp.damping, np.float64),
                            passer_old_pd.effort.copy(),
                        )
                        profile_a = w.switch_pd_profile(
                            "a/",
                            passer_old_pd
                            if a.passer_velocity_blend or a.passer_return_pair
                            else passer_new_pd,
                        )
                        boundary_a = folder / "passer_recovery_boundary.npz"
                        np.savez_compressed(
                            boundary_a,
                            **{"before_" + k: v for k, v in profile_a["before"].items()},
                            **{"after_" + k: v for k, v in profile_a["after"].items()},
                        )
                        passer_recovery_boundary = dict(
                            control=step,
                            profile={
                                k: v for k, v in profile_a.items() if k not in ("before", "after")
                            },
                            boundary=dict(
                                path=str(boundary_a.relative_to(a.output)), hash=digest(boundary_a)
                            ),
                        )
                        passer_recovery.activate(motion_complete=True, passer_foot_seen=True)
                        if a.passer_velocity_blend:
                            passer.begin_standing()
                    else:
                        passer.begin_standing()
                recovery_trace = None
                if passer_recovery is not None and successor_motor is None:
                    q_a, v_a = w.layouts["a/"].canonical(w.data)
                    recovery_command = (
                        return_receive_command(
                            q_a, v_a, return_direction=-forward, launch_seen=return_seen
                        )
                        if a.passer_return_pair
                        else np.zeros(3, np.float32)
                    )
                    if course["arm"].get("passer_return_align", False) and return_alignment_active:
                        recovery_command = np.asarray(
                            successor_alignment(
                                q_a, v_a, receiving_target_for_player(w, "b/")
                            ).command_body,
                            np.float32,
                        )
                    recovery_target = passer_recovery.observe(recovery_command)
                    recovery_trace = dict(
                        control=np.array(step),
                        obs=passer_recovery.stream.last_observation.copy(),
                        proposed_action=passer_recovery.stream.last_action.copy(),
                        phase=np.array(passer_recovery.stream.phase),
                    )
                    if a.passer_return_align_pair:
                        recovery_trace.update(
                            command=recovery_command.copy(),
                            alignment_active=np.array(return_alignment_active),
                        )
                    counts["passer_velocity_inferences"] += 1
                fraction = course["arm"].get("passer_residual_fraction", 1.0)
                teacher_arm = fraction != 1.0
                blend_index = (
                    step - passer_recovery_boundary["control"]
                    if (a.passer_velocity_blend or a.passer_return_pair)
                    and passer_recovery_boundary is not None
                    else -1
                )
                blend_active = 0 <= blend_index < 20
                if successor_motor is not None:
                    if successor_motor.phase == "MOTION_COMPLETE":
                        successor_motor.begin_standing()
                    target_a, trace_a = successor_motor.propose()
                    if (
                        successor_handoff is not None
                        and "first_target_jump_rad" not in successor_handoff
                    ):
                        assert successor_previous_target is not None
                        successor_handoff["first_target_jump_rad"] = float(
                            np.max(np.abs(target_a - successor_previous_target))
                        )
                elif passer_recovery is not None and passer_recovery.active and not blend_active:
                    if (a.passer_velocity_blend or a.passer_return_pair) and blend_index == 20:
                        w.switch_pd_profile("a/", passer_new_pd)
                    target_a, trace_a = recovery_target, None
                else:
                    target_a, trace_a = passer.propose(with_reference_evidence=teacher_arm)
                executed_a = None
                if blend_active:
                    weight = quintic_weight(blend_index, start=0, duration=20)
                    target_a, profile = blend_live_targets(
                        target_a, recovery_target, passer_old_pd, passer_new_pd, weight=weight
                    )
                    w.switch_pd_profile("a/", profile)
                    executed_a = policy_action_from_hg_target(
                        target_a, passer.offset, passer.scale, passer.order
                    )
                    # Execute precisely the representable target retained in the
                    # recurrent policy's own last-action history.
                    target_a = passer.offset + executed_a[passer.order] * passer.scale
                    counts["passer_blend_actor_inferences"] += 1
                if recovery_trace is not None and (a.passer_velocity_blend or a.passer_return_pair):
                    recovery_trace["blend_index"] = np.array(blend_index)
                if teacher_arm:
                    active = (
                        passer_exit_gate.active if a.passer_contact_exit else not passer_foot_seen
                    )
                    executed_a = blended_passer_action(
                        trace_a["NN_action"],
                        trace_a["NN_reference_nominal"],
                        passer.scale,
                        passer.order,
                        fraction=fraction if active else 1.0,
                    )
                    target_a = passer.offset + executed_a[passer.order] * passer.scale
                    trace_a.update(BLEND_active=np.array(active), BLEND_action=executed_a.copy())
                if trace_a is not None:
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
                        mode="velocity-match-v4"
                        if course["arm"].get("receiver_adaptive_intercept", False)
                        else "pursuit-v1",
                        lateral_offset_m=course["arm"].get("receiver_lateral_offset_m", 0.12),
                        longitudinal_offset_m=course["arm"].get(
                            "receiver_longitudinal_offset_m", 0.35
                        ),
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
                    if a.receiver_leg_limit_pair:
                        layout = w.layouts["b/"]
                        projection = receiver_target_limit_projection(
                            np.asarray(target_b, np.float64),
                            w.model.jnt_range[layout.joints].copy(),
                            w.model.jnt_limited[layout.joints].astype(np.bool_),
                            enabled=course["arm"]["receiver_leg_projection"],
                        )
                        target_b = projection["target"]
                        trace_v.update({"LIMIT_" + k: np.asarray(v) for k, v in projection.items()})
                        if (
                            a.receiver_projection_executed_memory
                            and course["arm"]["receiver_leg_projection"]
                        ):
                            velocity.last_action = projected_action_memory(
                                velocity.last_action,
                                target_b,
                                np.asarray(velocity.params.action_offset, np.float64),
                                np.asarray(velocity.params.action_scale, np.float64),
                                projection["changed_axes"],
                            )
                        if a.receiver_projection_executed_memory:
                            trace_v["LIMIT_memory_after"] = velocity.last_action.copy()
                    if (
                        a.receiver_whole_identity
                        or a.receiver_exploration
                        or a.receiver_learned_pair
                    ):
                        counts["receiver_onnx_calls"] += 1
                        if whole_receiver is not None:
                            counts["receiver_plastic_calls"] += 1
                            counts["receiver_anchor_calls"] += 1
                else:
                    if receiver.phase == "MOTION_COMPLETE":
                        receiver.begin_standing()
                    smooth_index = step - handoff["control"]
                    smooth_mode = a.precontact_smooth and course["arm"].get(
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
                counts["actual_actor_calls"] += 1
                if course["arm"].get("passer_strike_teacher", False) and step == 80:
                    passer_strike_teacher = SharedNativeReceiveTeacher(
                        w, "a/", horizon_controls=520, strike_foot_speed_mps=0.8
                    )
                row = w.advance(
                    {"a/": target_a, "b/": target_b},
                    torque_layers={"a/": passer_strike_teacher}
                    if passer_strike_teacher is not None
                    else None,
                )
                counts["native_steps"] += 10
                if trace_a is not None:
                    if blend_active:
                        passer.commit_sim_transition(executed_action=executed_a, target=target_a)
                    else:
                        (successor_motor if successor_motor is not None else passer).commit(
                            executed_action=executed_a if teacher_arm and active else None
                        )
                if passer_recovery is not None and successor_motor is None:
                    recovery_row = passer_recovery.commit(
                        transition_target=np.asarray(target_a, np.float64) if blend_active else None
                    )
                    assert recovery_trace is not None
                    recovery_trace.update({k: np.asarray(v) for k, v in recovery_row.items()})
                    nn_passer_velocity.append(recovery_trace)
                if successor_shadow_trace is not None:
                    assert successor_shadow is not None
                    successor_shadow.commit_actual()
                    nn_successor_shadow.append(successor_shadow_trace)
                passer_foot_seen = passer_foot_seen or bool(
                    (row["ball_forces"][:, 0, :2] > 10).any()
                )
                if return_readiness is not None:
                    return_alignment_active = return_readiness.update()
                if a.passer_successor_pair and successor_motor is None:
                    qa_gate, va_gate = w.layouts["a/"].canonical(w.data)
                    aligned = (
                        return_alignment_active
                        and successor_alignment(
                            qa_gate, va_gate, receiving_target_for_player(w, "b/")
                        ).geometry_ready
                    )
                    successor_ready_count = successor_ready_count + 1 if aligned else 0
                if a.passer_return_pair and receiver is not None:
                    return_seen = return_seen or return_strike_seen(
                        row["ball_forces"],
                        row["qvel"],
                        ball_v=w.layouts["a/"].bv,
                        return_direction=-forward,
                    )
                if a.passer_return_align_pair and return_seen and return_readiness is None:
                    return_readiness = OnlineReceivingBoundary(w, "a/")
                if a.passer_contact_exit:
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
                    nn_shadow.append(trace_shadow)
                    nn_v.append(trace_v)
                else:
                    receiver.commit(
                        executed_action=action_b
                        if balance_teacher is not None or smooth_active
                        else None
                    )
                    if a.precontact_fullbody_pair:
                        readiness.update()
                    nn_b.append(trace_b)
                if trace_a is not None:
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
                    | (
                        dict(
                            parent_torque=row.get("parent_torque", row["torque"]),
                            applied_torque_change=row.get(
                                "applied_torque_change", np.zeros_like(row["torque"])
                            ),
                        )
                        if a.passer_strike_teacher_pair
                        else {}
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
                    ("passer_velocity", nn_passer_velocity),
                    ("passer_successor_shadow", nn_successor_shadow),
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
                handoff_executed=(
                    readiness.first_ready_control is not None
                    if a.precontact_fullbody_pair
                    else handoff is not None
                ),
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
                passer_velocity_recovery_boundary=passer_recovery_boundary,
                passer_successor_handoff=successor_handoff,
                passer_strike_teacher_commands=passer_strike_teacher.commands
                if passer_strike_teacher is not None
                else 0,
                passer_strike_teacher_active=passer_strike_teacher.active_commands
                if passer_strike_teacher is not None
                else 0,
            )
            if a.precontact_fullbody_pair:
                record["motor_transition_executed"] = handoff is not None
                record["original_receiving_ready_control"] = readiness.first_ready_control
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
                    "NEURAL_RELAY_PASSER_SUCCESSOR16_PENDING_OWN_REPLAY"
                    if a.passer_successor_pair
                    else "NEURAL_RELAY_PASSER_RETURN_ALIGNMENT16_PENDING_OWN_REPLAY"
                    if a.passer_return_align_pair
                    else "NEURAL_RELAY_PASSER_RETURN_PAIR16_PENDING_OWN_REPLAY"
                    if a.passer_return_pair
                    else "NEURAL_RELAY_PASSER_REFERENCE_TEMPO16_PENDING_OWN_REPLAY"
                    if a.passer_reference_tempo_pair
                    else "NEURAL_RELAY_PASSER_STRIKE_TEACHER16_PENDING_OWN_REPLAY"
                    if a.passer_strike_teacher_pair
                    else "NEURAL_RELAY_PASSER_VELOCITY_BLEND16_PENDING_OWN_REPLAY"
                    if a.passer_velocity_blend
                    else "NEURAL_RELAY_PASSER_VELOCITY_RECOVERY16_PENDING_OWN_REPLAY"
                    if a.passer_velocity_recovery_pair
                    else "NEURAL_RELAY_RECEIVER_PROJECTED_ACTION_MEMORY16_PENDING_OWN_REPLAY"
                    if a.receiver_projection_executed_memory
                    else "NEURAL_RELAY_RECEIVER_LEG_TARGET_PROJECTION16_PENDING_OWN_REPLAY"
                    if a.receiver_leg_limit_pair
                    else "NEURAL_RELAY_RECEIVER_COORDINATED_INTERCEPT16_PENDING_OWN_REPLAY"
                    if a.receiver_coordinated_intercept
                    else "NEURAL_RELAY_RECEIVER_CONTACT_LEAD16_PENDING_OWN_REPLAY"
                    if a.receiver_early_contact_lead
                    else "NEURAL_RELAY_RECEIVER_EARLY_INTERCEPT16_PENDING_OWN_REPLAY"
                    if a.receiver_early_foot_intercept
                    else "NEURAL_RELAY_RECEIVER_AXIS_LOCAL_FALLBACK16_PENDING_OWN_REPLAY"
                    if a.receiver_axis_local_fallback
                    else "NEURAL_RELAY_RECEIVER_PREDICTIVE_FOOT_SHIN16_PENDING_OWN_REPLAY"
                    if a.receiver_predictive_foot_shin
                    else "NEURAL_RELAY_RECEIVER_FOOT_TEACHER16_PENDING_OWN_REPLAY"
                    if a.receiver_foot_teacher_pair
                    else "NEURAL_RELAY_PRECONTACT_FULLBODY16_PENDING_OWN_REPLAY"
                    if a.precontact_fullbody_pair
                    else "NEURAL_RELAY_RECEIVER_CENTER_EXPLORATION32_PENDING_OWN_REPLAY"
                    if a.receiver_center_curriculum and a.receiver_exploration
                    else "NEURAL_RELAY_RECEIVER_CENTER_IDENTITY16_PENDING_OWN_REPLAY"
                    if a.receiver_center_curriculum
                    else "NEURAL_RELAY_PASSER_CONTACT_EXIT16_PENDING_OWN_REPLAY"
                    if a.passer_contact_exit
                    else "NEURAL_RELAY_PASSER_RESIDUAL_PAIR16_PENDING_OWN_REPLAY"
                    if a.passer_residual_pair
                    else "NEURAL_RELAY_RECEIVER_LONGITUDINAL_AUTHORITY32_PENDING_OWN_REPLAY"
                    if a.receiver_longitudinal_authority
                    else "NEURAL_RELAY_RECEIVER_ADAPTIVE_INTERCEPT32_PENDING_OWN_REPLAY"
                    if a.receiver_adaptive_intercept
                    else "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY"
                    if a.receiver_lateral_authority
                    else "NEURAL_RELAY_RECEIVER_CONTINUATION_PAIR16_PENDING_OWN_REPLAY"
                    if a.receiver_continuation_pair
                    else "NEURAL_RELAY_RECEIVER_LEARNED_EXPLORATION32_PENDING_OWN_REPLAY"
                    if a.receiver_learned_exploration
                    else "NEURAL_RELAY_RECEIVER_MOTION_PAIR16_PENDING_OWN_REPLAY"
                    if a.receiver_motion_pair
                    else "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY"
                    if a.receiver_dense_pair
                    else "NEURAL_RELAY_RECEIVER_LEARNED_PAIR16_PENDING_OWN_REPLAY"
                    if a.receiver_learned_pair
                    else "NEURAL_RELAY_RECEIVER_EXPLORATION32_PENDING_OWN_REPLAY"
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
