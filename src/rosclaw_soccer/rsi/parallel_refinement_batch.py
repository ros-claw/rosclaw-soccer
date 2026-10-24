"""Finite native CPU sampling contracts for a fixed learned generation1 policy."""

from typing import Any


def parallel_config(state: int, seeds_per_initial: int) -> dict[str, Any]:
    if (
        type(state) is not int
        or state not in (25, 26)
        or type(seeds_per_initial) is not int
        or seeds_per_initial not in (4, 32)
    ):
        raise ValueError("two fixed TRAIN states and calibration4 or training32 seeds required")
    base = 346023 if seeds_per_initial == 4 else 446023
    n = 3 * seeds_per_initial
    return dict(
        protocol="soccer.refinement_gen1_parallel_native_sampling.v1",
        state_id=state,
        seeds=[base + 101 * state + i for i in range(seeds_per_initial)],
        seeds_per_initial=seeds_per_initial,
        worlds=n,
        physical_initial_conditions=3,
        scope="CONSUMED_TRAIN_INITIALS_NEW_PRIVATE_NOISE_NOT_FRESH",
        maximum_worlds=n,
        maximum_native_steps=4000 * n,
        maximum_actor_calls=800 * n,
        maximum_processes=4,
        worlds_per_task=2,
        sampling_profile="contact-v1",
        sampling_law="box-truncated-ar1",
        sigma_policy_units=0.02,
        rho=0.9,
        original_total_authority_rad=0.2,
        optimizer_updates=0,
        Fresh_cases_read=0,
    )


def pair_plan(
    config: dict[str, Any], courses: list[dict[str, Any]], pair: int
) -> list[dict[str, Any]]:
    if (
        config != parallel_config(config["state_id"], config["seeds_per_initial"])
        or len(courses) != 3
        or type(pair) is not int
        or not 0 <= pair < config["worlds"] // 2
    ):
        raise ValueError("exact complete finite pair plan required")
    n = config["seeds_per_initial"]
    return [
        dict(index=i, course=courses[i // n], sampling_seed=config["seeds"][i % n])
        for i in (2 * pair, 2 * pair + 1)
    ]


def verify_parallel_batch(report: dict[str, Any], courses: list[dict[str, Any]]) -> None:
    cfg = parallel_config(report["config"]["state_id"], report["config"]["seeds_per_initial"])
    records = report["records"]
    if report["config"] != cfg or len(records) != cfg["worlds"]:
        raise ValueError("all declared own-policy worlds required")
    counts = dict(native_steps=0, actor_calls=0)
    for i, r in enumerate(records):
        plan = pair_plan(cfg, courses, i // 2)[i % 2]
        if (
            any(r[k] != plan[k] for k in plan)
            or r["course"]["assisted"] is not False
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
            or not 1 <= r["stochastic_controls"] <= r["motion_reference_controls"] <= 300
            or type(r["stand_controls"]) is not int
            or not 0 <= r["stand_controls"] <= 100
            or r["observed_recovery_controls"] != r["stand_controls"]
            or r["total_observed_controls"] != r["controls"]
            or r["controls"] != r["stochastic_controls"] + r["stand_controls"]
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
            raise ValueError("exact pure own-policy native/law trajectory provenance required")
        for k in counts:
            counts[k] += r[k]
    if counts != report["counts"] or any(counts[k] > cfg["maximum_" + k] for k in counts):
        raise ValueError("complete bounded native clocks required")
