"""Finite own-stochastic experience calibration, not a teacher search or fit."""

from typing import Any

from rosclaw_soccer.rsi.joint_delivery_teacher import PILOTS


def pilot_config(state: int) -> dict[str, Any]:
    if type(state) is not int or state not in PILOTS:
        raise ValueError("two source-qualified near-delivery TRAIN pilots required")
    seed, task = PILOTS[state]
    return dict(
        protocol="soccer.refinement_own_AR_TRAIN24.v1",
        state_id=state,
        source_expert_seed=seed,
        task=task,
        sampling_law="box-truncated-ar1",
        sampling_profile="contact-v1",
        rho=0.9,
        sigma_policy_units=0.02,
        seeds=[196023 + 101 * state + i for i in range(8)],
        unsearched_TRAIN_offsets_forward_m=[-0.005, 0.005],
        offset_role="CONSUMED_PUBLIC_TRAIN_NOT_FRESH",
        physical_initial_conditions=3,
        worlds=24,
        maximum_worlds=24,
        maximum_native_steps=96000,
        maximum_actor_calls=19200,
        original_total_authority_rad=0.2,
        new_optimizer_updates=0,
        Fresh_cases_read=0,
        rollout_critic="INITIAL_DECLARED_ZERO_NOT_TRAINED",
    )


def verify_study(report: dict[str, Any], courses: list[dict[str, Any]]) -> None:
    cfg = pilot_config(report["config"]["state_id"])
    rows = report["records"]
    if report["config"] != cfg or len(rows) != 24 or len(courses) != 3:
        raise ValueError("complete3-initial x8-seed own-stochastic calibration required")
    totals = dict(native_steps=0, actor_calls=0)
    for i, r in enumerate(rows):
        if (
            r["index"] != i
            or r["course"] != courses[i // 8]
            or r["sampling_seed"] != cfg["seeds"][i % 8]
            or r["sampling_profile"] != cfg["sampling_profile"]
            or r["sampling_law"] != cfg["sampling_law"]
            or r["controller_kind"] != "pure-refinement-stochastic-native-v1"
            or any(
                r[k] is not False for k in ("assisted", "external_teacher", "authority_rejected")
            )
            or any(
                r[k] is not True
                for k in (
                    "all_actor_actions_actually_executed",
                    "natural_terminal_observed",
                    "deterministic_recovery_excluded_from_actor_rows",
                    "requires_whole_native_and_law_audit",
                )
            )
            or r["PPO_actor_rows"] != 0
            or r["neural_actor_optimizer_updates"] != 0
            or type(r["stochastic_controls"]) is not int
            or type(r["motion_reference_controls"]) is not int
            or type(r["stand_controls"]) is not int
            or not 0 <= r["stand_controls"] <= 100
            or not 1 <= r["stochastic_controls"] <= r["motion_reference_controls"] <= 300
            or r["controls"] != r["stochastic_controls"] + r["stand_controls"]
            or r["observed_recovery_controls"] != r["stand_controls"]
            or r["total_observed_controls"] != r["controls"]
            or r["native_steps"] != 10 * r["controls"]
            or r["actor_calls"] != 2 * r["controls"]
            or r["termination"] not in ("BODY_FAILURE", "FULL_MOTION_PLUS100NN_RECOVERY")
            or (
                r["termination"] == "FULL_MOTION_PLUS100NN_RECOVERY"
                and (
                    r["stand_controls"] != 100
                    or r["stochastic_controls"] != r["motion_reference_controls"]
                )
            )
        ):
            raise ValueError("exact actual per-world pure stochastic provenance/counters required")
        for k in totals:
            totals[k] += r[k]
    if (
        totals != report["counts"]
        or totals["native_steps"] > 96000
        or totals["actor_calls"] > 19200
    ):
        raise ValueError("bounded exact stochastic physics and neural counts required")
