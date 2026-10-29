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
    successor_blend_pair = (
        dec["protocol"] == "soccer.neural_relay_passer_successor_blend.development.v38"
    )
    successor_pair = (
        successor_blend_pair
        or dec["protocol"] == "soccer.neural_relay_passer_successor.development.v37"
    )
    return_align_pair = successor_pair or (
        dec["protocol"] == "soccer.neural_relay_passer_return_alignment.development.v36"
    )
    return_player_seeded_sampling = (
        dec["protocol"] == "soccer.neural_relay_return_player_own_sampling_seeded.development.v47"
    )
    receiver_velocity_match_prepared = (
        dec["protocol"] == "soccer.neural_relay_receiver_velocity_match_prepared.development.v51"
    )
    prepared_candidate = (
        dec["protocol"] == "soccer.neural_relay_prepared_return_candidate.development.v50"
    )
    prepared_sampling = (
        dec["protocol"] == "soccer.neural_relay_prepared_return_sampling.development.v49"
    )
    return_player_sampling = (
        prepared_sampling
        or return_player_seeded_sampling
        or (dec["protocol"] == "soccer.neural_relay_return_player_own_sampling.development.v46")
    )
    return_player_prepare = (
        receiver_velocity_match_prepared
        or prepared_candidate
        or prepared_sampling
        or (dec["protocol"] == "soccer.neural_relay_return_player_prepare.development.v48")
    )
    sampling_seed = 94111
    if return_player_seeded_sampling or prepared_sampling:
        declared_seeds = {c["arm"].get("passer_sampling_seed") for c in dec["courses"]}
        if len(declared_seeds) != 2 or None not in declared_seeds:
            raise ValueError("exact single declared sampled seed and baseline required")
        declared_seeds.remove(None)
        sampling_seed = declared_seeds.pop()
        if type(sampling_seed) is not int or sampling_seed not in (
            (94111, 94137, 94163) if prepared_sampling else (94137, 94163)
        ):
            raise ValueError("preregistered sampled seed required")
    return_player_identity = (
        prepared_candidate
        or return_player_sampling
        or (
            dec["protocol"] == "soccer.neural_relay_return_player_adaptive_identity.development.v45"
        )
    )
    child_return_role = (
        return_player_prepare
        or return_player_identity
        or (dec["protocol"] == "soccer.neural_relay_receiver_child_return_role.development.v44")
    )
    return_pair = (
        child_return_role
        or return_align_pair
        or (dec["protocol"] == "soccer.neural_relay_passer_return_pair.development.v35")
    )
    passer_velocity_blend = return_pair or (
        dec["protocol"] == "soccer.neural_relay_passer_velocity_blend.development.v31"
    )
    passer_velocity_recovery = passer_velocity_blend or (
        dec["protocol"] == "soccer.neural_relay_passer_velocity_recovery.development.v30"
    )
    projected_memory = (
        dec["protocol"] == "soccer.neural_relay_receiver_projected_action_memory.development.v27"
    )
    leg_limit_pair = projected_memory or (
        dec["protocol"] == "soccer.neural_relay_receiver_leg_target_projection.development.v26"
    )
    coordinated_intercept = (return_pair and not child_return_role) or (
        dec["protocol"] == "soccer.neural_relay_receiver_coordinated_intercept.development.v34"
    )
    contact_lead = coordinated_intercept or (
        dec["protocol"] == "soccer.neural_relay_receiver_contact_lead.development.v29"
    )
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
    precontact_exploration = (
        dec["protocol"] == "soccer.neural_relay_receiver_precontact_exploration.development.v42"
    )
    learned_exploration = (
        dec["protocol"]
        in (
            "soccer.neural_relay_receiver_dense_exploration.development.v19",
            "soccer.neural_relay_receiver_motion_exploration.development.v20",
        )
        or precontact_exploration
    )
    precontact_continuation = child_return_role or (
        dec["protocol"] == "soccer.neural_relay_receiver_precontact_continuation.development.v43"
    )
    continuation_pair = precontact_continuation or (
        dec["protocol"] == "soccer.neural_relay_receiver_continuation_pair.development.v21"
    )
    motion_pair = dec["protocol"] in (
        "soccer.neural_relay_receiver_motion_pair.development.v18",
        "soccer.neural_relay_receiver_motion_exploration.development.v20",
    )
    precontact_pair = precontact_exploration or (
        dec["protocol"] == "soccer.neural_relay_receiver_precontact_pair.development.v41"
    )
    functional_timed_pair = precontact_pair or (
        dec["protocol"] == "soccer.neural_relay_receiver_functional_timed_pair.development.v40"
    )
    functional_pair = (
        functional_timed_pair
        or dec["protocol"] == "soccer.neural_relay_receiver_functional_pair.development.v39"
    )
    dense_pair = (
        functional_pair
        or motion_pair
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
    longitudinal_authority = (
        dec["protocol"] == "soccer.neural_relay_receiver_longitudinal_authority.development.v32"
    )
    adaptive_intercept = (
        dec["protocol"] == "soccer.neural_relay_receiver_adaptive_intercept.development.v33"
    )
    lateral_authority = (
        longitudinal_authority
        or adaptive_intercept
        or (dec["protocol"] == "soccer.neural_relay_receiver_lateral_authority.development.v22")
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
        or passer_velocity_recovery
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
        or passer_velocity_recovery
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
        "NEURAL_RELAY_PASSER_SUCCESSOR_BLEND16_PENDING_OWN_REPLAY"
        if successor_blend_pair
        else "NEURAL_RELAY_PASSER_SUCCESSOR16_PENDING_OWN_REPLAY"
        if successor_pair
        else "NEURAL_RELAY_PASSER_RETURN_ALIGNMENT16_PENDING_OWN_REPLAY"
        if return_align_pair
        else "NEURAL_RELAY_RECEIVER_VELOCITY_MATCH_PREPARED16_PENDING_OWN_REPLAY"
        if receiver_velocity_match_prepared
        else "NEURAL_RELAY_PREPARED_RETURN_CANDIDATE16_PENDING_OWN_REPLAY"
        if prepared_candidate
        else "NEURAL_RELAY_PREPARED_RETURN_SAMPLING16_PENDING_OWN_REPLAY"
        if prepared_sampling
        else "NEURAL_RELAY_RETURN_PLAYER_PREPARE16_PENDING_OWN_REPLAY"
        if return_player_prepare
        else "NEURAL_RELAY_RETURN_PLAYER_SEEDED_SAMPLING16_PENDING_OWN_REPLAY"
        if return_player_seeded_sampling
        else "NEURAL_RELAY_RETURN_PLAYER_OWN_SAMPLING16_PENDING_OWN_REPLAY"
        if return_player_sampling
        else "NEURAL_RELAY_RETURN_PLAYER_ADAPTIVE_IDENTITY16_PENDING_OWN_REPLAY"
        if return_player_identity
        else "NEURAL_RELAY_CHILD_RETURN_ROLE16_PENDING_OWN_REPLAY"
        if child_return_role
        else "NEURAL_RELAY_PASSER_RETURN_PAIR16_PENDING_OWN_REPLAY"
        if return_pair
        else "NEURAL_RELAY_PASSER_VELOCITY_BLEND16_PENDING_OWN_REPLAY"
        if passer_velocity_blend
        else "NEURAL_RELAY_PASSER_VELOCITY_RECOVERY16_PENDING_OWN_REPLAY"
        if passer_velocity_recovery
        else "NEURAL_RELAY_RECEIVER_PROJECTED_ACTION_MEMORY16_PENDING_OWN_REPLAY"
        if projected_memory
        else "NEURAL_RELAY_RECEIVER_LEG_TARGET_PROJECTION16_PENDING_OWN_REPLAY"
        if leg_limit_pair
        else "NEURAL_RELAY_RECEIVER_COORDINATED_INTERCEPT16_PENDING_OWN_REPLAY"
        if coordinated_intercept
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
        else "NEURAL_RELAY_RECEIVER_LONGITUDINAL_AUTHORITY32_PENDING_OWN_REPLAY"
        if longitudinal_authority
        else "NEURAL_RELAY_RECEIVER_ADAPTIVE_INTERCEPT32_PENDING_OWN_REPLAY"
        if adaptive_intercept
        else "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY"
        if lateral_authority
        else "NEURAL_RELAY_RECEIVER_PRECONTACT_CONTINUATION16_PENDING_OWN_REPLAY"
        if precontact_continuation
        else "NEURAL_RELAY_RECEIVER_CONTINUATION_PAIR16_PENDING_OWN_REPLAY"
        if continuation_pair
        else "NEURAL_RELAY_RECEIVER_PRECONTACT_EXPLORATION32_PENDING_OWN_REPLAY"
        if precontact_exploration
        else "NEURAL_RELAY_RECEIVER_LEARNED_EXPLORATION32_PENDING_OWN_REPLAY"
        if learned_exploration
        else "NEURAL_RELAY_RECEIVER_MOTION_PAIR16_PENDING_OWN_REPLAY"
        if motion_pair
        else "NEURAL_RELAY_RECEIVER_PRECONTACT_PAIR16_PENDING_OWN_REPLAY"
        if precontact_pair
        else "NEURAL_RELAY_RECEIVER_FUNCTIONAL_TIMED_PAIR16_PENDING_OWN_REPLAY"
        if functional_timed_pair
        else "NEURAL_RELAY_RECEIVER_FUNCTIONAL_PAIR16_PENDING_OWN_REPLAY"
        if functional_pair
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
            "soccer.neural_relay_receiver_functional_pair.development.v39",
            "soccer.neural_relay_receiver_functional_timed_pair.development.v40",
            "soccer.neural_relay_receiver_precontact_pair.development.v41",
            "soccer.neural_relay_receiver_precontact_exploration.development.v42",
            "soccer.neural_relay_receiver_precontact_continuation.development.v43",
            "soccer.neural_relay_receiver_child_return_role.development.v44",
            "soccer.neural_relay_return_player_adaptive_identity.development.v45",
            "soccer.neural_relay_return_player_own_sampling.development.v46",
            "soccer.neural_relay_return_player_own_sampling_seeded.development.v47",
            "soccer.neural_relay_return_player_prepare.development.v48",
            "soccer.neural_relay_prepared_return_sampling.development.v49",
            "soccer.neural_relay_prepared_return_candidate.development.v50",
            "soccer.neural_relay_receiver_velocity_match_prepared.development.v51",
            "soccer.neural_relay_receiver_motion_pair.development.v18",
            "soccer.neural_relay_receiver_dense_exploration.development.v19",
            "soccer.neural_relay_receiver_motion_exploration.development.v20",
            "soccer.neural_relay_receiver_continuation_pair.development.v21",
            "soccer.neural_relay_receiver_lateral_authority.development.v22",
            "soccer.neural_relay_receiver_adaptive_intercept.development.v33",
            "soccer.neural_relay_receiver_longitudinal_authority.development.v32",
            "soccer.neural_relay_receiver_foot_teacher.development.v23",
            "soccer.neural_relay_receiver_predictive_foot_shin.development.v24",
            "soccer.neural_relay_receiver_axis_local_fallback.development.v25",
            "soccer.neural_relay_receiver_early_intercept.development.v28",
            "soccer.neural_relay_receiver_contact_lead.development.v29",
            "soccer.neural_relay_receiver_coordinated_intercept.development.v34",
            "soccer.neural_relay_passer_return_pair.development.v35",
            "soccer.neural_relay_passer_return_alignment.development.v36",
            "soccer.neural_relay_passer_successor.development.v37",
            "soccer.neural_relay_passer_successor_blend.development.v38",
            "soccer.neural_relay_passer_velocity_recovery.development.v30",
            "soccer.neural_relay_passer_velocity_blend.development.v31",
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
                    receiver_foot_teacher_strength=0.5,
                    receiver_adaptive_intercept=True,
                    passer_velocity_recovery=True if return_align_pair else enabled,
                    **(
                        {"passer_return_align": True if successor_pair else enabled}
                        if return_align_pair
                        else {}
                    ),
                    **(
                        {"passer_successor": True if successor_blend_pair else enabled}
                        if successor_pair
                        else {}
                    ),
                    **({"passer_successor_blend": enabled} if successor_blend_pair else {}),
                )
                for enabled in (False, True)
            ]
            if return_pair and not child_return_role
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_whole=True,
                    receiver_learned=True,
                    passer_velocity_recovery=True,
                    passer_prepare=True,
                    receiver_adaptive_intercept=enabled,
                )
                for enabled in (False, True)
            ]
            if receiver_velocity_match_prepared
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_whole=True,
                    receiver_learned=True,
                    passer_velocity_recovery=True,
                    passer_prepare=True if (prepared_sampling or prepared_candidate) else enabled,
                    **(
                        {"passer_adaptive": enabled}
                        if prepared_candidate
                        else {
                            "passer_adaptive": enabled,
                            "passer_sampling_seed": sampling_seed if enabled else None,
                        }
                        if prepared_sampling or prepared_candidate
                        else {}
                    ),
                )
                for enabled in (False, True)
            ]
            if return_player_prepare
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_whole=True,
                    receiver_learned=True,
                    passer_velocity_recovery=True,
                    passer_adaptive=enabled,
                    **(
                        {"passer_sampling_seed": sampling_seed if enabled else None}
                        if return_player_sampling
                        else {}
                    ),
                )
                for enabled in (False, True)
            ]
            if return_player_identity
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    receiver_whole=True,
                    receiver_learned=True,
                    passer_velocity_recovery=enabled,
                )
                for enabled in (False, True)
            ]
            if child_return_role and not return_player_identity and not return_player_prepare
            else [
                dict(
                    start=80,
                    memory="own_shadow_carry",
                    reference_bridge_controls=0,
                    passer_velocity_recovery=enabled,
                )
                for enabled in (False, True)
            ]
            if passer_velocity_recovery
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
            if longitudinal_authority
            else [
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
                    **(
                        {"receiver_adaptive_intercept": bool(strength)}
                        if coordinated_intercept
                        else {}
                    ),
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
                    receiver_adaptive_intercept=adaptive,
                )
                for offset in (-0.12, 0.12)
                for adaptive in (False, True)
            ]
            if adaptive_intercept
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
            if dense_pair
            or lateral_authority
            or foot_teacher_pair
            or leg_limit_pair
            or passer_velocity_recovery
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
        if (adaptive_intercept or receiver_velocity_match_prepared) and dec.get(
            "adaptive_intercept_is_unlearned_teacher"
        ) is not True:
            raise ValueError("explicit unlearned adaptive command authority required")
        if (
            coordinated_intercept
            and dec.get("coordinated_intercept_is_unlearned_teacher") is not True
        ):
            raise ValueError("explicit unlearned coordinated intercept required")
        if return_pair and dec.get("passer_return_role_is_unlearned") is not True:
            raise ValueError("explicit unlearned return role required")
        if return_align_pair and dec.get("passer_return_alignment_is_unlearned") is not True:
            raise ValueError("explicit unlearned return alignment required")
        if successor_pair and (
            dec.get("passer_successor_selection_is_unlearned") is not True
            or dec.get("passer_successor_direct_profile_switch") is not (not successor_blend_pair)
        ):
            raise ValueError("explicit unlearned direct-switch successor required")
        if successor_blend_pair and dec.get("passer_successor_blend_controls") != 20:
            raise ValueError("explicit 20-control successor blend required")
        if (
            longitudinal_authority
            and dec.get("longitudinal_intercept_is_unlearned_teacher") is not True
        ):
            raise ValueError("explicit unlearned longitudinal command authority required")
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
        if passer_velocity_recovery and dec.get("passer_velocity_recovery_pair") is not True:
            raise ValueError("explicit passer velocity recovery pair required")
        if passer_velocity_blend and dec.get("passer_velocity_blend_controls") != 20:
            raise ValueError("explicit twenty-control passer blend required")
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
    if prepared_candidate:
        for key in ("return_player_candidate", "return_player_training_receipt"):
            path = Path(dec[key])
            expected = dec[key + "_hash"]
            if path.is_symlink() or pins.get(str(path)) != expected or digest(path) != expected:
                raise ValueError("exact pinned return candidate required for own replay")
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
        if claim.get("passer_velocity_recovery_boundary") is not None:
            artifacts.append(claim["passer_velocity_recovery_boundary"]["boundary"])
        if claim.get("passer_successor_handoff") is not None:
            artifacts.append(claim["passer_successor_handoff"]["boundary"])
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
    if passer_velocity_recovery:
        counts["passer_velocity_inferences"] = 0
    if passer_velocity_blend:
        counts["passer_blend_actor_inferences"] = 0
    if return_player_identity:
        counts["passer_adaptive_calls"] = 0
    if return_player_sampling:
        counts["passer_sampled_controls"] = 0
    if successor_pair:
        counts["passer_successor_shadow_inferences"] = 0
    if whole_identity or exploration or learned_pair:
        counts.update(receiver_onnx_calls=0, receiver_plastic_calls=0, receiver_anchor_calls=0)
    if exploration or learned_exploration:
        counts["receiver_stochastic_calls"] = 0
    records = []
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
        from rosclaw_soccer.rsi.adaptive_passer_velocity import AdaptivePasserVelocityRecovery
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
        from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor
        from rosclaw_soccer.rsi.prepared_return_candidate import load_prepared_return_candidate
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
        from rosclaw_soccer.rsi.return_player_sampling_trace import return_player_sampling_trace
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
                functional_mode=functional_pair,
                functional_timed_mode=functional_timed_pair,
                precontact_mode=precontact_pair,
                precontact_continuation_mode=precontact_continuation,
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
            nn["passer_velocity"] = (
                load(claim, "passer_velocity") if passer_velocity_recovery else {}
            )
            nn["passer_successor_shadow"] = (
                load(claim, "passer_successor_shadow") if successor_pair else {}
            )
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
            successor_shadow = (
                SharedShadowMotor(
                    w,
                    "a/",
                    actor,
                    clip=segment,
                    goal=receiving_target_for_player(w, "b/"),
                    **kwargs,
                )
                if successor_pair
                else None
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
                if course["arm"].get("passer_adaptive", False):
                    zero_actor = zero_receiver_actor(setup)
                    adapted_actor = (
                        PrecontactReceivingMotor(zero_actor.plastic, zero_actor.action_scale)
                        .eval()
                        .requires_grad_(False)
                    )
                    if prepared_candidate:
                        adapted_actor = load_prepared_return_candidate(
                            setup,
                            Path(dec["return_player_candidate"]),
                            dec["return_player_candidate_hash"],
                            Path(dec["return_player_training_receipt"]),
                            dec["return_player_training_receipt_hash"],
                        )
                    passer_recovery = AdaptivePasserVelocityRecovery(
                        w,
                        "a/",
                        "b/",
                        stream,
                        adapted_actor,
                        sampling_seed=course["arm"].get("passer_sampling_seed"),
                    )
                else:
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
            return_seen = False
            return_readiness = None
            return_alignment_active = False
            successor_motor = None
            successor_ready_count = 0
            successor_previous_target = None
            successor_old_pd = successor_new_pd = None
            passer_exit_gate = PasserContactExitGate()
            handoff_control = None
            transition_start_action = None
            goal = origin = None
            excluded = set()
            last_target = None
            passer_old_pd = passer_new_pd = None
            passer_recovery_start = None
            largest_jump = 0.0
            physical = []
            vi = bi = ai = pvi = ssi = 0
            for step in range(len(raw["qpos"])):
                if (
                    course["arm"].get("passer_successor", False)
                    and successor_motor is None
                    and successor_ready_count >= 10
                ):
                    assert successor_shadow is not None
                    successor_previous_target = w.last_applied_target("a/").copy()
                    successor_old_pd = w.parameters["a/"].owned_copy()
                    successor_new_pd = NativePDParameters(ep.kp, ep.kd, ep.effort)
                    boundary_s = w.switch_pd_profile(
                        "a/",
                        successor_old_pd
                        if course["arm"].get("passer_successor_blend", False)
                        else successor_new_pd,
                    )
                    h_s = claim.get("passer_successor_handoff")
                    goal_s = receiving_target_for_player(w, "b/")
                    if (
                        h_s is None
                        or h_s["control"] != step
                        or h_s["goal"] != goal_s.tolist()
                        or h_s["shadow_observations"] != successor_shadow.observations
                    ):
                        raise ValueError("own successor handoff differs")
                    with np.load(a.source_root / h_s["boundary"]["path"], allow_pickle=False) as z:
                        for side in ("before", "after"):
                            for k, value in boundary_s[side].items():
                                if not np.array_equal(value, z[side + "_" + k]):
                                    raise ValueError("successor profile differs")
                        for k, value in dict(
                            shadow_h=successor_shadow.state[0].numpy(),
                            shadow_c=successor_shadow.state[1].numpy(),
                            previous_executed_action=successor_shadow.previous_action,
                        ).items():
                            if not np.array_equal(value, z[k]):
                                raise ValueError("successor private history differs")
                    successor_motor = successor_shadow.handoff(goal=goal_s, carry_memory=True)
                successor_shadow_trace = None
                if successor_shadow is not None and successor_motor is None:
                    successor_shadow_trace = successor_shadow.observe()
                    successor_shadow_trace["control"] = np.array(step)
                    equal_trace(successor_shadow_trace, nn["passer_successor_shadow"], ssi)
                    ssi += 1
                    counts["passer_successor_shadow_inferences"] += 1
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
                if return_pair and passer.phase == "MOTION_COMPLETE":
                    passer.begin_standing()
                if (
                    (passer.phase == "MOTION_COMPLETE" and not return_pair)
                    or (
                        return_pair
                        and passer_recovery is not None
                        and passer.phase == "STANDING"
                        and (
                            return_seen
                            or (
                                course["arm"].get("passer_prepare", False)
                                and handoff_control is not None
                            )
                        )
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
                        boundary_a = w.switch_pd_profile(
                            "a/",
                            passer_old_pd if passer_velocity_blend else passer_new_pd,
                        )
                        recorded_boundary = claim.get("passer_velocity_recovery_boundary")
                        if recorded_boundary is None or recorded_boundary["control"] != step:
                            raise ValueError("own passer recovery boundary differs")
                        with np.load(
                            a.source_root / recorded_boundary["boundary"]["path"],
                            allow_pickle=False,
                        ) as z:
                            for side in ("before", "after"):
                                for k, value in boundary_a[side].items():
                                    if not np.array_equal(value, z[side + "_" + k]):
                                        raise ValueError("own unchanged passer boundary differs")
                        passer_recovery.activate(motion_complete=True, passer_foot_seen=True)
                        passer_recovery_start = step
                        if passer_velocity_blend and not return_pair:
                            passer.begin_standing()
                    else:
                        passer.begin_standing()
                recovery_trace = None
                successor_blend_index = (
                    step - claim["passer_successor_handoff"]["control"]
                    if successor_motor is not None
                    else -1
                )
                successor_blend_active = (
                    course["arm"].get("passer_successor_blend", False)
                    and 0 <= successor_blend_index < 20
                )
                if passer_recovery is not None and (
                    successor_motor is None or successor_blend_active
                ):
                    q_a, v_a = w.layouts["a/"].canonical(w.data)
                    recovery_command = (
                        return_receive_command(
                            q_a, v_a, return_direction=-forward, launch_seen=return_seen
                        )
                        if return_pair
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
                    if return_align_pair:
                        recovery_trace.update(
                            command=recovery_command.copy(),
                            alignment_active=np.array(return_alignment_active),
                        )
                    counts["passer_velocity_inferences"] += 1
                fraction = course["arm"].get("passer_residual_fraction", 1.0)
                teacher_arm = fraction != 1.0
                blend_index = (
                    step - passer_recovery_start
                    if passer_velocity_blend and passer_recovery_start is not None
                    else -1
                )
                blend_active = 0 <= blend_index < 20
                if successor_motor is not None:
                    if (
                        course["arm"].get("passer_successor_blend", False)
                        and successor_blend_index == 20
                    ):
                        w.switch_pd_profile("a/", successor_new_pd)
                    if successor_motor.phase == "MOTION_COMPLETE":
                        successor_motor.begin_standing()
                    target_a, trace = successor_motor.propose()
                elif passer_recovery is not None and passer_recovery.active and not blend_active:
                    if passer_velocity_blend and blend_index == 20:
                        w.switch_pd_profile("a/", passer_new_pd)
                    target_a, trace = recovery_target, None
                else:
                    target_a, trace = passer.propose(with_reference_evidence=teacher_arm)
                executed_a = None
                if successor_blend_active:
                    weight = quintic_weight(successor_blend_index, start=0, duration=20)
                    target_a, profile = blend_live_targets(
                        recovery_target, target_a, successor_old_pd, successor_new_pd, weight=weight
                    )
                    w.switch_pd_profile("a/", profile)
                    executed_a = policy_action_from_hg_target(
                        target_a,
                        successor_motor.offset,
                        successor_motor.scale,
                        successor_motor.order,
                    )
                    target_a = (
                        successor_motor.offset
                        + executed_a[successor_motor.order] * successor_motor.scale
                    )
                if (
                    successor_motor is not None
                    and step == claim["passer_successor_handoff"]["control"]
                ):
                    assert successor_previous_target is not None
                    if (
                        float(np.max(np.abs(target_a - successor_previous_target)))
                        != claim["passer_successor_handoff"]["first_target_jump_rad"]
                    ):
                        raise ValueError("own successor first target jump differs")
                if blend_active:
                    weight = quintic_weight(blend_index, start=0, duration=20)
                    target_a, profile = blend_live_targets(
                        target_a, recovery_target, passer_old_pd, passer_new_pd, weight=weight
                    )
                    w.switch_pd_profile("a/", profile)
                    executed_a = policy_action_from_hg_target(
                        target_a, passer.offset, passer.scale, passer.order
                    )
                    target_a = passer.offset + executed_a[passer.order] * passer.scale
                    counts["passer_blend_actor_inferences"] += 1
                if recovery_trace is not None and passer_velocity_blend:
                    recovery_trace["blend_index"] = np.array(blend_index)
                    if successor_blend_pair:
                        recovery_trace["successor_blend_index"] = np.array(successor_blend_index)
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
                if trace is not None:
                    trace["control"] = np.array(step)
                    equal_trace(trace, nn["passer_nn"], ai)
                    ai += 1
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
                if trace is not None:
                    if blend_active:
                        passer.commit_sim_transition(executed_action=executed_a, target=target_a)
                    elif successor_blend_active:
                        if successor_motor.phase == "MOTION":
                            successor_motor.commit_motion_sim_transition(
                                executed_action=executed_a, target=target_a
                            )
                        else:
                            successor_motor.commit_sim_transition(
                                executed_action=executed_a, target=target_a
                            )
                    else:
                        (successor_motor if successor_motor is not None else passer).commit(
                            executed_action=executed_a if teacher_arm and active else None
                        )
                if passer_recovery is not None and (
                    successor_motor is None or successor_blend_active
                ):
                    recovery_row = passer_recovery.commit(
                        transition_target=np.asarray(target_a, np.float64)
                        if blend_active or successor_blend_active
                        else None
                    )
                    assert recovery_trace is not None
                    recovery_trace.update({k: np.asarray(v) for k, v in recovery_row.items()})
                    if return_player_identity:
                        adapted = getattr(passer_recovery, "last_adaptive_trace", None)
                        recovery_trace.update(
                            adaptive_active=np.array(adapted is not None),
                            adaptive_gate=np.array(
                                adapted["precontact_gate"] if adapted is not None else 0.0
                            ),
                            adaptive_delta=np.asarray(
                                adapted["realized_delta_radians"]
                                if adapted is not None
                                else np.zeros(29),
                                dtype=np.float64,
                            ),
                        )
                        counts["passer_adaptive_calls"] += int(adapted is not None)
                        if return_player_sampling:
                            sampled_trace = return_player_sampling_trace(adapted)
                            recovery_trace.update(sampled_trace)
                            counts["passer_sampled_controls"] += int(
                                sampled_trace["sampling_active"]
                            )
                    equal_trace(recovery_trace, nn["passer_velocity"], pvi)
                    pvi += 1
                if successor_shadow_trace is not None:
                    assert successor_shadow is not None
                    successor_shadow.commit_actual()
                passer_foot_seen = passer_foot_seen or bool(
                    (row["ball_forces"][:, 0, :2] > 10).any()
                )
                if return_readiness is not None:
                    return_alignment_active = return_readiness.update()
                if successor_pair and successor_motor is None:
                    qa_gate, va_gate = w.layouts["a/"].canonical(w.data)
                    aligned = (
                        return_alignment_active
                        and successor_alignment(
                            qa_gate, va_gate, receiving_target_for_player(w, "b/")
                        ).geometry_ready
                    )
                    successor_ready_count = successor_ready_count + 1 if aligned else 0
                if return_pair and receiver is not None:
                    return_seen = return_seen or return_strike_seen(
                        row["ball_forces"],
                        row["qvel"],
                        ball_v=w.layouts["a/"].bv,
                        return_direction=-forward,
                    )
                if return_align_pair and return_seen and return_readiness is None:
                    return_readiness = OnlineReceivingBoundary(w, "a/")
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
                or len(nn["passer_nn"]["control"]) != ai
                or (
                    passer_velocity_recovery
                    and len(nn["passer_velocity"].get("control", [])) != pvi
                )
                or len(nn["receiver_velocity"].get("control", [])) != vi
                or len(nn["receiver_shadow"].get("control", [])) != vi
                or len(nn["receiver_return_nn"].get("control", [])) != bi
                or (successor_pair and len(nn["passer_successor_shadow"].get("control", [])) != ssi)
                or (
                    successor_pair
                    and bool(claim.get("passer_successor_handoff"))
                    != bool(successor_motor is not None)
                )
            ):
                raise ValueError("complete neural horizons or final online transition differ")
            if passer_velocity_recovery and (
                bool(claim.get("passer_velocity_recovery_boundary"))
                != bool(passer_recovery is not None and passer_recovery.active)
                or (
                    passer_recovery is not None
                    and pvi
                    != (
                        w.control_steps
                        if successor_motor is None
                        else min(
                            w.control_steps,
                            claim["passer_successor_handoff"]["control"]
                            + (20 if course["arm"].get("passer_successor_blend", False) else 0),
                        )
                    )
                )
                or (passer_recovery is None and pvi != 0)
            ):
                raise ValueError("complete own passer shadow and actual handoff required")
            if return_player_sampling:
                from rosclaw_soccer.rsi.return_player_sampling_audit import (
                    audit_return_player_sampling,
                )

                audit_return_player_sampling(
                    nn["passer_velocity"],
                    np.asarray(setup.velocity_parameters.action_scale, dtype=np.float32),
                )
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
