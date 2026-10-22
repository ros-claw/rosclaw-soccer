"""Predeclared new initial conditions for the native reference-motion skill.

Known motion references, unseen continuous ball/entry conditions. Not the
legacy H300/1N52 exam, not unseen motions, and not a continuous football match.
"""

import copy
import json
from typing import Any

import numpy as np

FRESH_SEED = 2026102304
FRESH_PROTOCOL = "dual256.frozen_four_condition40_and_run16.v1"


def physical_context_key(ctx: dict[str, Any]) -> str:
    c = ctx["case"]
    return json.dumps(
        dict(
            source=c["source"],
            ball=c["ball_world_initial"],
            velocity=c["velocity_world_initial"],
            goal=c["goal_world"],
            backoff_m=ctx["backoff_m"],
        ),
        sort_keys=True,
        allow_nan=False,
    )


def native_fresh_courses(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(contexts) != 22 or not 1 <= len({c["case"]["source"] for c in contexts[:20]}) <= 10:
        raise ValueError("original fixed20 motion conditions and two consumed run entries required")
    if any(c["case"]["bank"] != "train" or c["case"]["not_fresh"] is not True for c in contexts):
        raise ValueError(
            "Fresh must be derived from declared consumed contexts, not relabel old cases"
        )
    rng = np.random.default_rng(FRESH_SEED)
    result: list[dict[str, Any]] = []

    def directions(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        goal = np.asarray(case["goal_world"][:2], np.float64)
        norm = np.linalg.norm(goal)
        if not np.isfinite(goal).all() or not 5 <= norm <= 7:
            raise ValueError("same declared native six-meter goal frame required")
        f = goal / norm
        return f, np.array([-f[1], f[0]])

    def append(ctx: dict[str, Any], cohort: str, condition: str, source_index: int) -> None:
        i = len(result)
        ctx["case"].update(case=20000 + i, bank="fresh_native_20261023", not_fresh=False)
        result.append(
            dict(index=i, cohort=cohort, condition=condition, source_context=source_index, ctx=ctx)
        )

    for pair_index in range(10):
        if (
            contexts[2 * pair_index]["case"]["source"]
            != contexts[2 * pair_index + 1]["case"]["source"]
        ):
            raise ValueError("two original conditions per fixed motion required")
        for far in (False, True):
            source_index = 2 * pair_index + int(far)
            for away in (False, True):
                ctx = copy.deepcopy(contexts[source_index])
                f, lateral = directions(ctx["case"])
                old = np.asarray(ctx["case"]["ball_world_initial"][:2])
                distance = (1.05 if far else 0.65) + rng.uniform(-0.04, 0.04)
                side = float(old @ lateral) + rng.uniform(-0.035, 0.035)
                ball = distance * f + side * lateral
                velocity = (1 if away else -1) * rng.uniform(0.17, 0.27) * f + rng.uniform(
                    -0.035, 0.035
                ) * lateral
                ctx["case"]["ball_world_initial"] = [float(ball[0]), float(ball[1]), 0.11]
                ctx["case"]["velocity_world_initial"] = velocity.tolist()
                ctx["backoff_m"] = 0.0
                append(
                    ctx,
                    "base40",
                    ("far" if far else "near") + "_" + ("away" if away else "approaching"),
                    source_index,
                )
    for source_index in (20, 21):
        for j in range(8):
            ctx = copy.deepcopy(contexts[source_index])
            f, lateral = directions(ctx["case"])
            offset = (1 if j % 2 else -1) * rng.uniform(0.025, 0.045) * f + rng.uniform(
                -0.035, 0.035
            ) * lateral
            ball = np.asarray(ctx["case"]["ball_world_initial"][:2]) + offset
            away = (j // 2) % 2 == 1
            velocity = (1 if away else -1) * rng.uniform(0.17, 0.27) * f + rng.uniform(
                -0.035, 0.035
            ) * lateral
            ctx["case"]["ball_world_initial"] = [float(ball[0]), float(ball[1]), 0.11]
            ctx["case"]["velocity_world_initial"] = velocity.tolist()
            ctx["backoff_m"] = float(
                rng.uniform(1.45, 1.75) if source_index == 20 else rng.uniform(1.05, 1.35)
            )
            append(
                ctx,
                "run16",
                ("long" if source_index == 20 else "short")
                + "_"
                + ("away" if away else "approaching"),
                source_index,
            )
    keys = [physical_context_key(c["ctx"]) for c in result]
    if (
        len(result) != 56
        or len(set(keys)) != 56
        or set(keys) & {physical_context_key(c) for c in contexts}
    ):
        raise ValueError("56 distinct physical conditions, not metadata relabellings, required")
    return result


def native_fresh_summary(
    records: list[dict[str, Any]], courses: list[dict[str, Any]], seeds: list[int]
) -> dict[str, Any]:
    arms = ["baseline"] + [str(s) for s in seeds]
    if (
        not seeds
        or seeds != sorted(set(seeds))
        or any(type(s) is not int or s not in (65023, 65024, 65025) for s in seeds)
        or len(courses) != 56
        or len(records) != 56 * len(arms)
        or {(r["course_index"], r["arm"]) for r in records}
        != {(i, a) for i in range(56) for a in arms}
        or any(
            type(r[k]) is not bool
            for r in records
            for k in ("healthy_quality", "body_failed", "forbidden")
        )
        or any(r["healthy_quality"] and (r["body_failed"] or r["forbidden"]) for r in records)
    ):
        raise ValueError("all declared matched outcomes, including every failure, required")
    lookup = {(r["course_index"], r["arm"]): r for r in records}
    reports = []
    for seed in seeds:
        by_group = {}
        for cohort in ("base40", "run16"):
            indices = [c["index"] for c in courses if c["cohort"] == cohort]
            parent = sum(lookup[i, "baseline"]["healthy_quality"] for i in indices)
            child = sum(lookup[i, str(seed)]["healthy_quality"] for i in indices)
            by_group[cohort] = dict(
                courses=len(indices),
                baseline_successes=parent,
                candidate_successes=child,
                net=child - parent,
            )
        conditions = {}
        for label in sorted({c["condition"] for c in courses}):
            indices = [c["index"] for c in courses if c["condition"] == label]
            conditions[label] = sum(
                int(lookup[i, str(seed)]["healthy_quality"])
                - int(lookup[i, "baseline"]["healthy_quality"])
                for i in indices
            )
        safety = sum(
            (lookup[i, str(seed)]["body_failed"] and not lookup[i, "baseline"]["body_failed"])
            or (lookup[i, str(seed)]["forbidden"] and not lookup[i, "baseline"]["forbidden"])
            for i in range(56)
        )
        reports.append(
            dict(
                seed=seed,
                cohorts=by_group,
                condition_net=conditions,
                critical_task_regressions=safety,
                rescues=sum(
                    lookup[i, str(seed)]["healthy_quality"]
                    and not lookup[i, "baseline"]["healthy_quality"]
                    for i in range(56)
                ),
                regressions=sum(
                    not lookup[i, str(seed)]["healthy_quality"]
                    and lookup[i, "baseline"]["healthy_quality"]
                    for i in range(56)
                ),
                preregistered_local_Fresh_screen_pass=by_group["base40"]["net"] >= 0
                and by_group["run16"]["net"] >= 4
                and all(v >= 0 for v in conditions.values())
                and safety == 0,
            )
        )
    return dict(
        reports=reports,
        courses=56,
        worlds=len(records),
        known_motion_families_base=len(
            {c["ctx"]["case"]["source"] for c in courses if c["cohort"] == "base40"}
        ),
        known_motion_families_run=len(
            {c["ctx"]["case"]["source"] for c in courses if c["cohort"] == "run16"}
        ),
        calibrated_probability=False,
        unseen_motion_generalization=False,
        legacy52_proven=False,
        team_game_proven=False,
        independent_native_audit_required=True,
        promotion_authorized=False,
        goal_complete=False,
    )
