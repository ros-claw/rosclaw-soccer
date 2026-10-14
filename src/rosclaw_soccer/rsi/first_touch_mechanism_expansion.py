"""Outcome-independent public expansion; neither Fresh nor teacher admission."""

import math
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.sim.contracts import hash_json


def validate_public_course(course: tuple[float, float, float]) -> int:
    if (
        type(course) is not tuple
        or len(course) != 3
        or any(type(v) is not float or not math.isfinite(v) for v in course)
    ):
        raise ValueError("ordinary finite float public course required")
    x, y, vx = course
    if not ((2.2 <= x < 2.4 or 2.6 <= x < 2.8) and -0.16 <= y <= 0.16 and 0.25 <= abs(vx) <= 0.7):
        raise ValueError("entire public course must remain inside declared domain")
    return 2 * int(x >= 2.6) + int(vx > 0)


def declare_expansion(plan: PublicFirstTouchTrainingPlan, *, seed: int) -> dict[str, Any]:
    if type(plan) is not PublicFirstTouchTrainingPlan or len(plan.seeds) != 10:
        raise ValueError("exact complete40 public context plan required")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("bounded declared perturbation seed required")
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    strata = [0, 0, 0, 0]
    for draw in plan.draws():
        if draw.sample_index:
            continue
        bucket = validate_public_course(draw.course)
        x, y, vx = draw.course
        xmin, xmax = (2.2, 2.4) if x < 2.4 else (2.6, 2.8)
        vmin, vmax = (0.25, 0.7) if vx > 0 else (-0.7, -0.25)
        # Draw from a predeclared conditional domain. No generated sample is
        # clipped/replaced/dropped, and no physical outcome is consulted.
        ranges = (
            (max(xmin, x - 0.005), min(xmax, x + 0.005)),
            (max(-0.16, y - 0.01), min(0.16, y + 0.01)),
            (max(vmin, vx - 0.025), min(vmax, vx + 0.025)),
        )
        values = [float(rng.uniform(lo, hi)) for lo, hi in ranges]
        perturbed = (values[0], values[1], values[2])
        if validate_public_course(perturbed) != bucket:
            raise ValueError("reject entire expansion before any model allocation")
        strata[bucket] += 1
        records.append(
            dict(
                group=draw.group,
                context_id=draw.context_id,
                course=draw.course,
                perturbed_course=perturbed,
                original_sampling_seed=draw.sampling_seed,
                pilot_adopted=draw.seed in plan.seeds[:4],
            )
        )
    all_courses = [tuple(r[k]) for r in records for k in ("course", "perturbed_course")]
    if len(records) != 40 or strata != [10] * 4 or len(set(all_courses)) != 80:
        raise ValueError("whole40 balanced states and40 distinct perturbations required")
    result = dict(
        schema="soccer.rsi.mechanism_expansion.v1",
        seed=seed,
        public_training_plan_hash=plan.contract()["plan_hash"],
        records=records,
        strata_counts=strata,
        planned_new_base_candidate_episodes=24,
        planned_perturbed_paired_episodes=80,
        planned_exact_control_episodes=1,
        planned_new_native_executions=105,
        all_courses_preflighted=True,
        perturbations_used_for_optimization=False,
        private_fresh_accessed=False,
        clustered_original_contexts=40,
        independent_scenarios_claimed=40,
        privileged_teacher_proven=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))


def select_pilot_mechanism(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a research expansion, never a teacher or promotion, from primitives.

    Safety regression is a veto even when task success increases. Caller must
    independently authenticate/review these physical records first.
    """
    summaries = {}
    identities = None
    baselines: dict[int, str] = {}
    for arm in ("lateral", "direction", "combined"):
        rows = [r for r in records if r.get("arm") == arm]
        groups = [r.get("group") for r in rows]
        if (
            len(rows) != 16
            or any(type(g) is not int for g in groups)
            or len(set(groups)) != 16
            or (identities is not None and set(groups) != identities)
        ):
            raise ValueError("complete paired16 public states required for every arm")
        identities = set(groups)
        for row in rows:
            for name in ("baseline_outcome", "outcome"):
                if type(row.get(name)) is not dict or any(
                    type(row[name].get(k)) is not bool for k in ("task_pass", "safe")
                ):
                    raise ValueError("explicit primitive boolean outcomes required")
            group = row["group"]
            baseline_hash = hash_json(row["baseline_outcome"])
            if group in baselines and baselines[group] != baseline_hash:
                raise ValueError("all arms must share the exact same per-state baseline")
            baselines[group] = baseline_hash
        rescue = sum(
            r["outcome"]["task_pass"] and not r["baseline_outcome"]["task_pass"] for r in rows
        )
        regression = sum(
            not r["outcome"]["task_pass"] and r["baseline_outcome"]["task_pass"] for r in rows
        )
        safety = sum(r["baseline_outcome"]["safe"] and not r["outcome"]["safe"] for r in rows)
        summaries[arm] = dict(
            rescue=rescue,
            regression=regression,
            safety_regression=safety,
            net_rescue=rescue - regression,
            eligible_for_expansion=rescue > regression and safety == 0,
        )
    eligible = [a for a in summaries if summaries[a]["eligible_for_expansion"]]
    chosen = max(eligible, key=lambda a: summaries[a]["net_rescue"]) if eligible else None
    result = dict(
        schema="soccer.rsi.public_pilot_mechanism_decision.v1",
        summaries=summaries,
        chosen_mechanism=chosen,
        decision="EXPAND_PUBLIC_MECHANISM" if chosen else "PIVOT_ACTION_INTERFACE",
        safety_veto_overrides_task_gain=True,
        privileged_teacher_proven=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
