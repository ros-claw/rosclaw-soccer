"""Versioned continuing-learning contracts; legacy gen1 semantics stay stable."""

from dataclasses import dataclass
from typing import Any

from rosclaw_soccer.rsi.refinement_continual_batch import continual_config

KIND = "refinement-actor-critic-continuing-v4"
SCHEMA = "soccer.refinement.continuing_checkpoint.v4"
STATUS = "REFINEMENT_CONTINUING_UPDATED_PENDING_PHYSICAL_VALIDATION"
PREPARED = "CONTINUING_NEW_POLICY_BATCH_AND_TRAINED_CRITIC_PREPARED_NOT_UPDATED"


def continuing_config(state: int, generation: int, prior_steps: int) -> dict[str, Any]:
    if (
        type(generation) is not int
        or not 2 <= generation <= 16
        or type(prior_steps) is not int
        or not generation + 3 <= prior_steps <= 4 * generation
    ):
        raise ValueError("bounded actual generation2..16 and non-reset prior counters required")
    cfg = continual_config(state)
    cfg.update(
        protocol="refinement.own_AR_PPO_continuing96.v4",
        generation_index=generation,
        parent_generation_index=generation - 1,
        prior_optimizer_steps=prior_steps,
        source_batch_worlds=96,
    )
    return cfg


@dataclass(frozen=True)
class ContinuationContract:
    kind: str
    schema: str
    status: str
    prepared_status: str
    prior_steps: int
    worlds: int
    selected_indices: tuple[int, ...]


def contract_for(config: dict[str, Any]) -> ContinuationContract:
    state = config.get("state_id")
    if type(state) is not int:
        raise ValueError("exact integer model state required")
    if config == continual_config(state):
        return ContinuationContract(
            "refinement-actor-critic-continual-gen1-v3",
            "soccer.refinement.continual_gen1_checkpoint.v3",
            "REFINEMENT_CONTINUAL_GEN1_UPDATED_PENDING_PHYSICAL_VALIDATION",
            "CONTINUAL_NEW_LEARNED_ARM24_AND_TRAINED_CRITIC_PREPARED_NOT_UPDATED",
            4,
            24,
            tuple(range(1, 48, 2)),
        )
    generation, prior = config.get("generation_index"), config.get("prior_optimizer_steps")
    if (
        type(generation) is not int
        or type(prior) is not int
        or config != continuing_config(state, generation, prior)
    ):
        raise ValueError("exact versioned continuing96 configuration required")
    return ContinuationContract(KIND, SCHEMA, STATUS, PREPARED, prior, 96, tuple(range(96)))


def select_parallel_experience(
    source: dict[str, Any], own: dict[str, Any], *, source_hash: str, training_hash: str, state: int
) -> list[dict[str, Any]]:
    from rosclaw_soccer.rsi.parallel_refinement_batch import parallel_config, verify_parallel_batch

    cfg = parallel_config(state, 32)
    if (
        source["config"] != cfg
        or own["config"] != cfg
        or len(source["records"]) != 96
        or len(own["records"]) != 96
    ):
        raise ValueError("exact complete new96 same-policy batch required")
    courses = [source["records"][i]["course"] for i in (0, 32, 64)]
    verify_parallel_batch(source, courses)
    verify_parallel_batch(own, courses)
    if (
        source["status"] != "GEN1_PARALLEL_OWN_POLICY_BATCH_COLLECTED_PENDING_NATIVE_LAW"
        or own["status"] != "GEN1_PARALLEL_OWN_POLICY_BATCH_ALL_NATIVE_LAW_MATCH"
        or own["source_hash"] != source_hash
        or source["training_hash"] != training_hash
        or own["training_hash"] != training_hash
        or own["sampling_law_audit_passed"] is not True
        or own["independent_whole_world_replay"] is not True
        or own["records"]
        != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
        or source["counts"] != own["counts"]
        or any("arm" in r for r in source["records"])
        or any(
            r[k] is not False
            for r in (source, own)
            for k in (
                "actor_training_admitted",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "Fresh_verified",
                "team_game_verified",
            )
        )
        or any(r["optimizer_updates"] != 0 for r in (source, own))
    ):
        raise ValueError("all96 actual new-policy independent native/RNG histories required")
    result: list[dict[str, Any]] = source["records"]
    return result
