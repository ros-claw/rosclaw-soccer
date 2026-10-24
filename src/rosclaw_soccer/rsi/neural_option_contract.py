"""Fixed SIM research contract for call-and-return neural motor composition."""

from typing import Any

KIND = "committed-neural-options-v1"
STATUS = "COMMITTED_NEURAL_OPTIONS_REFERENCE_BUNDLE_ASSEMBLED_PENDING_BOUND_CPU_AND_PHYSICS"
OWN97_STATUS = "COMMITTED_NEURAL_OPTIONS_OWN97_ACTIONS_MATCH_SOURCES_NOT_PHYSICS"
IDENTITIES = (
    "parent_training_hash",
    "parent_lineage_hash",
    "parent_checkpoint_hash",
    "data_hash",
    "feasibility_hash",
    "initiation_hash",
    "own97_hash",
    "expert_training_hashes",
    "expert_checkpoint_hashes",
    "generation_hash",
)


def option_config() -> dict[str, Any]:
    return dict(
        seed=0,
        deterministic_composition=True,
        source_head_seeds=[115023, 115024, 115025],
        new_optimizer_updates=0,
        reused_head_updates_per_seed=2048,
        initiation_histories=97,
        maximum_private_controls=512,
        learned_termination=False,
        online_RL_claim=False,
    )


def option_cpu_scope(own: dict[str, Any]) -> int:
    rows = own["records"]
    if (
        own["status"] != OWN97_STATUS
        or len(rows) != 97
        or own["physical_steps_executed"] != 0
        or own["optimizer_updates"] != 0
        or own["own_actor_decisions"] != 36276
        or any(
            own[k] is not False
            for k in (
                "combined_physical_performance_proven",
                "native_checkpoint_admitted",
                "Fresh_verified",
                "team_game_verified",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
        or own["selected_neural_skills_have_exact_actions_on_recorded_streams"] is not True
        or [r["index"] for r in rows] != list(range(97))
        or [r["scope"] for r in rows] != ["public_pass"] * 80 + ["protected_motion"] * 17
        or any(
            type(r["rows"]) is not int
            or not 1 <= r["rows"] <= 512
            or type(r["option"]) is not int
            or not 0 <= r["option"] <= 3
            or any(
                r[k] is not True
                for k in (
                    "all_source_actions_bitwise_equal",
                    "all_own_hc_bitwise_equal",
                    "option_committed_through_recovery",
                )
            )
            for r in rows
        )
        or any(r["option"] != 0 for r in rows[80:])
        or sum(r["rows"] for r in rows) != own["own_actor_decisions"]
    ):
        raise ValueError("complete exact own97 neural commitment proof required, not physics")
    return int(own["own_actor_decisions"])
