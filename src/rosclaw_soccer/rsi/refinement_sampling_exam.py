"""New private noise seeds, paired OLD/PPO policies on consumed TRAIN initials."""

from typing import Any

from rosclaw_soccer.rsi.refinement_onpolicy_learning import joint_delivery_reward


def sampling_exam_config(state: int) -> dict[str, Any]:
    if type(state) is not int or state not in (25, 26):
        raise ValueError("two predeclared consumed TRAIN states required")
    return dict(
        protocol="soccer.refinement_PPO_new_seed_paired48.v1",
        state_id=state,
        seeds=[246023 + 101 * state + i for i in range(8)],
        arms=["old", "learned"],
        physical_initial_conditions=3,
        scope="NEW_NOISE_ON_CONSUMED_TRAIN_INITIALS_NOT_FRESH_OR_GENERALIZATION",
        sampling_profile="contact-v1",
        sampling_law="box-truncated-ar1",
        rho=0.9,
        sigma_policy_units=0.02,
        original_total_authority_rad=0.2,
        maximum_worlds=48,
        maximum_native_steps=192000,
        maximum_actor_calls=38400,
        optimizer_updates=0,
        Fresh_cases_read=0,
    )


def verify_sampling_exam(report: dict[str, Any], courses: list[dict[str, Any]]) -> None:
    cfg = sampling_exam_config(report["config"]["state_id"])
    rows = report["records"]
    if report["config"] != cfg or len(rows) != 48 or len(courses) != 3:
        raise ValueError("all3 initials x8 new seeds x2 policies required")
    totals = dict(native_steps=0, actor_calls=0)
    for i, r in enumerate(rows):
        if (
            r["index"] != i
            or r["arm"] != cfg["arms"][i % 2]
            or r["course"] != courses[i // 16]
            or r["course"]["assisted"] is not False
            or r["sampling_seed"] != cfg["seeds"][(i // 2) % 8]
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
            or r["neural_actor_optimizer_updates"] != 0
            or r["PPO_actor_rows"] != 0
            or type(r["stochastic_controls"]) is not int
            or not 1 <= r["stochastic_controls"] <= r["motion_reference_controls"] <= 300
            or type(r["stand_controls"]) is not int
            or not 0 <= r["stand_controls"] <= 100
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
            raise ValueError("exact unassisted paired own-state sampling provenance required")
        for k in totals:
            totals[k] += r[k]
    if totals != report["counts"] or any(totals[k] > cfg["maximum_" + k] for k in totals):
        raise ValueError("bounded all-world physical clocks required")


def paired_rewards(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive paired24 noise samples, not independent unseen course trials."""
    if len(records) != 48:
        raise ValueError("all paired48 outcomes required")
    pairs = []
    for i in range(0, 48, 2):
        old, new = records[i : i + 2]
        if (
            old["arm"] != "old"
            or new["arm"] != "learned"
            or old["sampling_seed"] != new["sampling_seed"]
            or old["course"] != new["course"]
        ):
            raise ValueError("matched declared policy pairs required")
        before, after = [joint_delivery_reward(r["outcome"]) for r in (old, new)]
        pairs.append(
            dict(
                initial_group=i // 16,
                seed=old["sampling_seed"],
                old_reward=before,
                learned_reward=after,
                difference=after - before,
            )
        )
    return dict(
        pairs=pairs,
        paired_noise_samples=24,
        unique_consumed_TRAIN_initials=3,
        mean_paired_reward_difference=sum(r["difference"] for r in pairs) / 24,
        positive_pairs=sum(r["difference"] > 1e-6 for r in pairs),
        negative_pairs=sum(r["difference"] < -1e-6 for r in pairs),
        generalization_proven=False,
        deterministic_mean_improvement_proven=False,
        promotion_authorized=False,
        goal_complete=False,
    )
