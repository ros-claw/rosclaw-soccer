"""Two consumed TRAIN failures, explicit diagnostic-only receiver intervention.

The old factual worlds are REUSED, not quietly charged again. Only four new
worlds (two cases x two repeats) are collected and independently replayed.
This cannot admit data to PPO or turn a selected counterexample into Fresh.
"""

from typing import Any

SOURCE = "RECEIVER_COUNTEREXAMPLE4_COLLECTED_PENDING_OWN_NATIVE"
OWN = "RECEIVER_COUNTEREXAMPLE4_ALL_NATIVE_LAW_MATCH"


def config() -> dict[str, Any]:
    return dict(
        protocol="soccer.receiver_selected_counterexample4.v1",
        state_id=26,
        source_case_indices=[6, 24],
        new_worlds=4,
        repeats_per_case=2,
        maximum_native_steps=16000,
        maximum_actor_calls=3200,
        receiver_intent_mode="anchor-plane-v2",
        sampling_profile="contact-v1",
        A_behavior_generation_index=2,
        A_behavior_cumulative_actor_steps=12,
        original_total_authority_rad=0.2,
        formal_controls=300,
        recovery_controls=100,
        selected_consumed_TRAIN_not_Fresh=True,
        optimizer_updates=0,
        Fresh_cases_read=0,
    )


def verify(report: dict[str, Any], factual: dict[str, Any]) -> None:
    from rosclaw_soccer.rsi.parallel_refinement_batch import parallel_config

    cfg = config()
    rows = report["records"]
    if (
        report["config"] != cfg
        or len(rows) != 4
        or factual["config"] != parallel_config(26, 32, full_recovery=True)
    ):
        raise ValueError("all four declared diagnostic worlds required")
    totals = dict(native_steps=0, actor_calls=0)
    for i, r in enumerate(rows):
        j = cfg["source_case_indices"][i // 2]
        old = factual["records"][j]
        if (
            r["index"] != i
            or r["source_case_index"] != j
            or r["course"] != old["course"]
            or r["course"]["assisted"] is not False
            or r["sampling_seed"] != old["sampling_seed"]
            or r["sampling_profile"] != "contact-v1"
            or r["sampling_law"] != "box-truncated-ar1"
            or r["receiver_intent_mode"] != "anchor-plane-v2"
            or r["receiver_role_ablation"] is not True
            or r["controller_kind"] != "receiver-role-stochastic-diagnostic-native-v3"
            or r["deterministic_recovery_excluded_from_actor_rows"] is not False
            or r["stochastic_controls"] != r["controls"]
            or r["total_observed_controls"] != r["controls"]
            or not 1 <= r["controls"] <= 400
            or not 0 <= r["stand_controls"] <= 100
            or r["observed_recovery_controls"] != r["stand_controls"]
            or not 1 <= r["controls"] - r["stand_controls"] <= r["motion_reference_controls"] <= 300
            or (
                r["stand_controls"]
                and r["controls"] - r["stand_controls"] != r["motion_reference_controls"]
            )
            or r["termination"] not in ("BODY_FAILURE", "FULL_MOTION_PLUS100NN_RECOVERY")
            or (r["termination"] == "FULL_MOTION_PLUS100NN_RECOVERY" and r["stand_controls"] != 100)
            or r["native_steps"] != 10 * r["controls"]
            or r["actor_calls"] != 2 * r["controls"]
            or any(
                r[k] is not False for k in ("assisted", "external_teacher", "authority_rejected")
            )
            or any(
                r[k] is not True
                for k in ("all_actor_actions_actually_executed", "natural_terminal_observed")
            )
            or r["PPO_actor_rows"] != 0
            or r["neural_actor_optimizer_updates"] != 0
            or r["requires_whole_native_and_law_audit"] is not True
            or any(k in r for k in ("feedback_variant", "student_prefix_controls"))
        ):
            raise ValueError("exact diagnostic-only actual sampling/role/clock provenance required")
        if i % 2 and {k: v for k, v in r.items() if k not in ("index", "artifacts")} != {
            k: v for k, v in rows[i - 1].items() if k not in ("index", "artifacts")
        }:
            raise ValueError("actual same-seed repeats must match")
        for k in totals:
            totals[k] += r[k]
    if totals != report["counts"] or any(totals[k] > cfg["maximum_" + k] for k in totals):
        raise ValueError("all bounded charged clocks required")
