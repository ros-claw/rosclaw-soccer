"""Predeclare actual interventions independently of proxy/task outcomes."""

from typing import Any

from rosclaw_soccer.rsi.postcontact_robustness_plan import declare_robustness40
from rosclaw_soccer.sim.contracts import hash_json

OPPOSED_ACTIONS = (
    ("forward_plus", "forward_minus"),
    ("lateral_plus", "lateral_minus"),
    ("lift_plus", "lift_minus"),
    ("yaw_plus", "yaw_minus"),
    ("lateral-1_yaw-1", "lateral+1_yaw+1"),
    ("lateral-1_yaw+1", "lateral+1_yaw-1"),
)


def declare_response_probes(*, mean_model_hash: str) -> dict[str, Any]:
    parent = declare_robustness40(mean_model_hash=mean_model_hash)
    records = [
        {
            "course_index": r["index"],
            "course": r["course"],
            "sampling_seed": r["sampling_seed"],
            "source_cluster": r["source_cluster"],
            "stratum": r["stratum"],
            "actions": list(OPPOSED_ACTIONS[r["index"] % len(OPPOSED_ACTIONS)]),
            "heading_enabled": False,
        }
        for r in parent["records"]
    ]
    result = dict(
        schema="soccer.research.public_factual_response_probe80_plan.v1",
        parent_public_plan_hash=parent["report_hash"],
        mean_model_hash=mean_model_hash,
        records=records,
        unique_initial_states=40,
        source_seed_clusters=10,
        new_native_rollouts=80,
        native_step_budget=240000,
        forecast_step_budget=0,
        original_baseline_bodies_reused=40,
        reused_baselines_are_new_physical_executions=False,
        outcomes_used_to_choose_probe=False,
        all_failure_unsafe_censored_records_required=True,
        consumed_public_contexts_not_Fresh=True,
        new_optimizer_updates=0,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result
