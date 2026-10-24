"""Finite native CPU sampling contracts for a fixed learned generation1 policy."""

from typing import Any

from rosclaw_soccer.rsi.refinement_actor_clock import DIAGNOSTIC, FULL, LEGACY, validate_actor_clock

FULL_SOURCE = "FULL_RECOVERY_PARALLEL_BATCH_COLLECTED_PENDING_NATIVE_LAW"
FULL_OWN = "FULL_RECOVERY_PARALLEL_BATCH_ALL_NATIVE_LAW_MATCH"
ROLE_SOURCE = "RECEIVER_ROLE_PARALLEL_BATCH_COLLECTED_PENDING_NATIVE_LAW"
ROLE_OWN = "RECEIVER_ROLE_PARALLEL_BATCH_ALL_NATIVE_LAW_MATCH"


def parallel_config(
    state: int,
    seeds_per_initial: int,
    *,
    full_recovery: bool = False,
    receiver_role_ablation: bool = False,
) -> dict[str, Any]:
    if (
        type(state) is not int
        or type(full_recovery) is not bool
        or type(receiver_role_ablation) is not bool
        or (receiver_role_ablation and (not full_recovery or state != 26))
        or state not in (25, 26)
        or type(seeds_per_initial) is not int
        or seeds_per_initial not in (4, 32)
    ):
        raise ValueError("two fixed TRAIN states and calibration4 or training32 seeds required")
    base = 346023 if seeds_per_initial == 4 else 446023
    if full_recovery:
        base += 200000  # Disjoint from legacy motion-only calibration and batch.
    n = 3 * seeds_per_initial
    cfg = dict(
        protocol=(
            "soccer.refinement_gen2_full_recovery_native_sampling.v2"
            if full_recovery
            else "soccer.refinement_gen1_parallel_native_sampling.v1"
        ),
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
    if full_recovery:
        cfg.update(
            sampling_include_recovery=True,
            behavior_generation_index=2,
            behavior_cumulative_actor_steps=12,
            formal_motion_deadline_controls=300,
            required_recovery_controls=100,
        )
    if receiver_role_ablation:
        cfg.update(
            protocol="soccer.gen2_receiver_anchor_plane_parallel_sampling.v3",
            receiver_role_ablation=True,
            receiver_intent_mode="anchor-plane-v2",
            scope="CONSUMED_TRAIN_SAME_PRIVATE_SEEDS_FIXED_A_NEW_B_ROLE_NOT_FRESH",
            noise_is_new=False,
            factual_worlds_reexecuted=0,
        )
    return cfg


def config_from_claim(config: dict[str, Any]) -> dict[str, Any]:
    cfg = parallel_config(
        config["state_id"],
        config["seeds_per_initial"],
        full_recovery=config.get("sampling_include_recovery", False),
        receiver_role_ablation=config.get("receiver_role_ablation", False),
    )
    if config != cfg:
        raise ValueError("exact versioned parallel sampling configuration required")
    return cfg


def pair_plan(
    config: dict[str, Any], courses: list[dict[str, Any]], pair: int
) -> list[dict[str, Any]]:
    if (
        config != config_from_claim(config)
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
    cfg = config_from_claim(report["config"])
    full = cfg.get("sampling_include_recovery", False)
    role = cfg.get("receiver_role_ablation", False)
    records = report["records"]
    if report["config"] != cfg or len(records) != cfg["worlds"]:
        raise ValueError("all declared own-policy worlds required")
    counts = dict(native_steps=0, actor_calls=0)
    for i, r in enumerate(records):
        validate_actor_clock(r, r["stochastic_controls"], allow_diagnostic=role)
        plan = pair_plan(cfg, courses, i // 2)[i % 2]
        if (
            any(r[k] != plan[k] for k in plan)
            or r["course"]["assisted"] is not False
            or r["sampling_profile"] != cfg["sampling_profile"]
            or r["sampling_law"] != cfg["sampling_law"]
            or r["controller_kind"] != (DIAGNOSTIC if role else (FULL if full else LEGACY))
            or r.get("receiver_role_ablation", False) is not role
            or (role and r.get("receiver_intent_mode") != "anchor-plane-v2")
            or (not role and "receiver_intent_mode" in r)
            or r["deterministic_recovery_excluded_from_actor_rows"] is not (not full)
            or any(
                r[k] is not False for k in ("assisted", "external_teacher", "authority_rejected")
            )
            or any(
                r[k] is not True
                for k in (
                    "all_actor_actions_actually_executed",
                    "natural_terminal_observed",
                    "requires_whole_native_and_law_audit",
                )
            )
            or r["PPO_actor_rows"] != 0
            or r["neural_actor_optimizer_updates"] != 0
            or type(r["stochastic_controls"]) is not int
            or type(r["stand_controls"]) is not int
            or not 0 <= r["stand_controls"] <= 100
            or r["observed_recovery_controls"] != r["stand_controls"]
            or r["total_observed_controls"] != r["controls"]
            or r["controls"] != r["stochastic_controls"] + (0 if full else r["stand_controls"])
            or r["native_steps"] != 10 * r["controls"]
            or r["actor_calls"] != 2 * r["controls"]
            or r["termination"] not in ("BODY_FAILURE", "FULL_MOTION_PLUS100NN_RECOVERY")
        ):
            raise ValueError("exact pure own-policy native/law trajectory provenance required")
        for k in counts:
            counts[k] += r[k]
    if counts != report["counts"] or any(counts[k] > cfg["maximum_" + k] for k in counts):
        raise ValueError("complete bounded native clocks required")
