"""Compare all unique paired TRAIN cases; no training or promotion authority."""

from typing import Any

from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label
from rosclaw_soccer.rsi.refinement_onpolicy_learning import joint_delivery_reward
from rosclaw_soccer.rsi.refinement_paired_exam import verify_exam


def paired_review(source: dict[str, Any], own: dict[str, Any]) -> dict[str, Any]:
    courses = [source["records"][i]["course"] for i in (0, 4, 8)]
    verify_exam(source, courses)
    verify_exam(own, courses)
    if (
        source["status"] != "PPO_PAIRED_MEAN12_COLLECTED_PENDING_OWN_NATIVE"
        or own["status"] != "PPO_PAIRED_MEAN12_ALL_OWN_NATIVE_MATCH"
        or own["independent_whole_world_replay"] is not True
        or own["records"]
        != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
        or own["counts"] != source["counts"]
        or own["training_hash"] != source["training_hash"]
    ):
        raise ValueError("all paired mean worlds require independent own-native match")
    cases = []
    for i in (0, 4, 8):
        arms = {}
        for offset, name in ((0, "old"), (1, "learned")):
            o = source["records"][i + offset]["outcome"]
            arms[name] = dict(
                reward=joint_delivery_reward(o),
                strict=bool(
                    o["combined_precision_and_functional_receive"] and controlled_pass_label(o)
                ),
                controlled=controlled_pass_label(o),
                healthy=o["whole_body_healthy"],
                full_recovery=o["passer_full100NNrecovery"],
                clean_foot=o["passer_clean_foot"],
                error_m=o["incoming"]["incoming_target_error_3d_before_B_contact_m"],
                speed_mps=o["incoming_arrival_speed_mps"],
            )
        cases.append(dict(course=courses[i // 4], **arms))
    deltas = [r["learned"]["reward"] - r["old"]["reward"] for r in cases]
    gains = sum(not r["old"]["strict"] and r["learned"]["strict"] for r in cases)
    losses = sum(r["old"]["strict"] and not r["learned"]["strict"] for r in cases)
    safety_losses = sum(
        any(r["old"][k] and not r["learned"][k] for k in ("healthy", "full_recovery", "clean_foot"))
        for r in cases
    )
    local_screen = bool(
        not losses and not safety_losses and min(deltas) >= -1e-6 and max(deltas) > 1e-6
    )
    return dict(
        status="PPO_PAIRED_MEAN_LOCAL_TRAIN_REVIEWED",
        unique_consumed_TRAIN_initials=3,
        repeated_worlds_per_integrator=12,
        cases=cases,
        reward_deltas=deltas,
        strict_gains=gains,
        strict_losses=losses,
        safety_losses=safety_losses,
        local_research_screen_passed=local_screen,
        next_agent_action=(
            "IMPLEMENT_FULL80_AND_RETENTION44_EXAM_NOT_PROMOTION"
            if local_screen
            else "ANALYZE_ACTUAL_REGRESSIONS_BEFORE_NEXT_ONPOLICY_BATCH"
        ),
        physical_steps_executed=0,
        optimizer_updates=0,
        Fresh_verified=False,
        continuous_learning_proven=False,
        team_game_verified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
