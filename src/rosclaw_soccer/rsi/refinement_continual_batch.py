"""Select ONLY a learned behavior's new on-policy arm for continual training.

The paired experiment includes the old policy as a scientific control. Those
old-policy actions are NOT on-policy data for the learned successor.
"""

from typing import Any

from rosclaw_soccer.rsi.refinement_onpolicy_learning import learning_config
from rosclaw_soccer.rsi.refinement_sampling_exam import verify_sampling_exam


def continual_config(state: int) -> dict[str, Any]:
    cfg = learning_config(state)
    cfg.update(
        protocol="refinement.own_AR_PPO_generation1_carried_actor_critic.v3",
        generation_index=1,
        advantage_rule="MONTE_CARLO_MINUS_FROZEN_PRIOR_LEARNED_CRITIC_GLOBAL_STD",
        optimizer_initialization="CARRY_PRIOR_ADAM_MOMENTS_COUNTERS_REBASE_LR_PER_NEW_BATCH",
        parent_generation_index=0,
    )
    return cfg


def select_learned_arm(
    source: dict[str, Any], own: dict[str, Any], *, source_hash: str, training_hash: str
) -> list[dict[str, Any]]:
    if len(source["records"]) != 48 or len(own["records"]) != 48:
        raise ValueError("both complete paired48 histories required")
    courses = [source["records"][i]["course"] for i in (0, 16, 32)]
    verify_sampling_exam(source, courses)
    verify_sampling_exam(own, courses)
    if (
        source["status"] != "PPO_NEW_SEED_PAIRED48_COLLECTED_PENDING_OWN_NATIVE_LAW"
        or own["status"] != "PPO_NEW_SEED_PAIRED48_ALL_OWN_NATIVE_AND_LAW_MATCH"
        or own["source_hash"] != source_hash
        or source["training_hash"] != training_hash
        or own["training_hash"] != training_hash
        or own["sampling_law_audit_passed"] is not True
        or own["independent_whole_world_replay"] is not True
        or own["records"]
        != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
        or own["counts"] != source["counts"]
        or any(
            r[k] is not False
            for r in (source, own)
            for k in (
                "actor_training_admitted",
                "promotion_authorized",
                "hardware_authorized",
                "goal_complete",
                "Fresh_verified",
                "team_game_verified",
            )
        )
        or any(r["optimizer_updates"] != 0 for r in (source, own))
    ):
        raise ValueError("all exact new-behavior native/law replay evidence required")
    selected: list[dict[str, Any]] = source["records"][1::2]
    if len(selected) != 24 or any(r["arm"] != "learned" for r in selected):
        raise ValueError("ALL24 learned-arm trajectories only; no control-arm PPO credit")
    return selected
