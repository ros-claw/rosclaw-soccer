"""Predeclared new contexts, not new motion sources or a historical52 exam."""

from collections.abc import Iterable
from typing import Any

import numpy as np

PROTOCOL = "paid.fresh_lifecycle.8sources.4contexts.v1"
ROUTER_FRESH_PROTOCOL = "paid.fresh_lifecycle.8sources.4contexts.ancestral_router.v2"
CASE_SEEDS = {PROTOCOL: 2026102207, ROUTER_FRESH_PROTOCOL: 2026102301}


def fresh_lifecycle_cases(
    source_cases: list[dict[str, Any]], *, protocol: str = PROTOCOL
) -> list[dict[str, Any]]:
    if protocol not in CASE_SEEDS:
        raise ValueError("one explicitly versioned fixed Fresh generator required")
    if (
        len(source_cases) != 16
        or [x.get("case") for x in source_cases] != list(range(16))
        or len({x.get("source") for x in source_cases}) != 8
        or any(source_cases[2 * i]["source"] != source_cases[2 * i + 1]["source"] for i in range(8))
    ):
        raise ValueError(
            "eight frozen motion sources with two declared TRAIN entries each required"
        )
    rng = np.random.Generator(np.random.PCG64(CASE_SEEDS[protocol]))
    cases: list[dict[str, Any]] = []
    for cluster in range(8):
        for distance in range(2):
            row = source_cases[2 * cluster + distance]
            ball, goal = (
                np.asarray(row["ball_world_initial"], float),
                np.asarray(row["goal_world"], float),
            )
            if (
                row.get("profile") != "cached-standard"
                or row.get("ball_mass_kg") != 0.43
                or ball.shape != (3,)
                or goal.shape != (3,)
                or not np.isfinite(ball).all()
                or not np.isfinite(goal).all()
                or ball[2] != 0.11
                or goal[2] != 0.11
                or np.linalg.norm(goal[:2]) < 4
            ):
                raise ValueError("unchanged standard body and explicit baseline geometry required")
            forward = goal[:2] / np.linalg.norm(goal[:2])
            lateral = np.array([-forward[1], forward[0]])
            for sign in (-1, 1):
                # No labels copied, no condition resampling/clipping. PCG seed,
                # draw order, ranges and axes are fixed before physics results.
                position = ball.copy()
                position[:2] += (
                    rng.uniform(-0.02, 0.02) * forward + rng.uniform(-0.02, 0.02) * lateral
                )
                velocity = (
                    sign * rng.uniform(0.17, 0.23) * forward + rng.uniform(-0.04, 0.04) * lateral
                )
                angle = rng.uniform(-0.025, 0.025)
                rotation = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                target = goal.copy()
                target[:2] = rotation @ target[:2]
                cases.append(
                    dict(
                        case=len(cases),
                        source=row["source"],
                        profile="cached-standard",
                        ball_mass_kg=0.43,
                        ball_world_initial=position.tolist(),
                        velocity_world_initial=velocity.tolist(),
                        goal_world=target.tolist(),
                        source_cluster=cluster,
                        fresh_class=("near" if distance == 0 else "far")
                        + ("_toward" if sign < 0 else "_away"),
                    )
                )
    return cases


def fresh_summary(worlds: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    if len(worlds) != 32 or [w.get("case") for w in worlds] != list(range(32)) or len(cases) != 32:
        raise ValueError("complete32 contexts required, including all legal failures")
    for w in worlds:
        q = w.get("quality_by_kick")
        if (
            type(w.get("stable")) is not bool
            or not isinstance(q, list)
            or len(q) > 3
            or any(type(v) is not bool for v in q)
        ):
            raise ValueError("explicit physical failure/success outcomes required")

    def counts(ids: Iterable[int]) -> dict[str, Any]:
        ids = list(ids)
        return dict(
            stable=sum(worlds[i]["stable"] for i in ids),
            quality_by_kick=[
                sum(
                    len(worlds[i]["quality_by_kick"]) > k and worlds[i]["quality_by_kick"][k]
                    for i in ids
                )
                for k in range(3)
            ],
        )

    return dict(
        **counts(range(32)),
        by_class={
            name: counts([i for i, c in enumerate(cases) if c["fresh_class"] == name])
            for name in ("near_toward", "near_away", "far_toward", "far_away")
        },
        declared_contexts=32,
        motion_source_clusters=8,
        repeats_are_not_independent=True,
    )


def paired_fresh_review(
    arms: dict[str, list[dict[str, Any]]], cases: list[dict[str, Any]], criteria: dict[str, Any]
) -> dict[str, Any]:
    """Per-case comparisons and source-cluster uncertainty, no policy promotion."""
    if set(arms) != {"r2", "parent", "progressive"}:
        raise ValueError("all three frozen arms required")
    summaries = {name: fresh_summary(rows, cases) for name, rows in arms.items()}
    candidate = arms["progressive"]
    comparisons = {}
    for reference in ("r2", "parent"):
        old = arms[reference]
        gained, lost = [], []
        for k in range(3):
            a = {
                i
                for i, w in enumerate(old)
                if len(w["quality_by_kick"]) > k and w["quality_by_kick"][k]
            }
            b = {
                i
                for i, w in enumerate(candidate)
                if len(w["quality_by_kick"]) > k and w["quality_by_kick"][k]
            }
            gained.append(sorted(b - a))
            lost.append(sorted(a - b))
        effects = np.zeros(8)
        for i, case in enumerate(cases):
            effects[case["source_cluster"]] += (float(i in gained[2]) - float(i in lost[2])) / 4
        rng = np.random.Generator(np.random.PCG64(2026102208))
        bootstrap = effects[rng.integers(0, 8, size=(10000, 8))].mean(axis=1)
        comparisons[reference] = dict(
            gained_by_kick=gained,
            lost_by_kick=lost,
            body_regressions=[
                i for i in range(32) if old[i]["stable"] and not candidate[i]["stable"]
            ],
            third_net_gain=len(gained[2]) - len(lost[2]),
            third_cluster_effects=effects.tolist(),
            third_source_cluster_bootstrap95_effect_per_case=np.quantile(
                bootstrap, [0.025, 0.975]
            ).tolist(),
            bootstrap_resamples=10000,
            bootstrap_seed=2026102208,
            statistical_clusters=8,
        )
    checks = dict(
        candidate_stable=summaries["progressive"]["stable"]
        == criteria["candidate_stable_required"],
        no_case_or_body_regressions=all(
            not any(c["lost_by_kick"]) and not c["body_regressions"] for c in comparisons.values()
        ),
        third_gain_vs_R2=comparisons["r2"]["third_net_gain"]
        >= criteria["third_gain_vs_R2_minimum"],
        third_gain_vs_parent=comparisons["parent"]["third_net_gain"]
        >= criteria["third_gain_vs_parent_minimum"],
    )
    return dict(
        summaries=summaries,
        comparisons=comparisons,
        checks=checks,
        preregistered_outcome_conditions_met=all(checks.values()),
        statistical_interval_is_not_a_promotion_gate=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )
