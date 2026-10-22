"""Paired TRAIN-only neural relay metrics; no policy promotion or physics.

The unit of distinct initial condition is (context, lateral), not arm/frame.
Counterfactual arms of one condition stay together in every comparison.
"""

from collections import Counter
from typing import Any

import numpy as np


def relay_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
    c = row["course"]
    return c["context"], c["lateral_m"], c["arm"]["start"], c["arm"]["memory"]


def compare_relay_arms(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 128:
        raise ValueError("complete128 preregistered trials required")
    by_bridge: dict[int, dict[tuple[int, float, int, str], dict[str, Any]]] = {
        n: {} for n in (0, 5, 15, 30)
    }
    expected = {
        (i, side, phase, memory)
        for i in (1, 4, 20, 21)
        for side in (-0.55, 0.55)
        for phase in (0, 80)
        for memory in ("new_skill_zero", "own_shadow_carry")
    }
    for row in records:
        n = row["course"]["arm"]["reference_bridge_controls"]
        key = relay_key(row)
        if n not in by_bridge or key not in expected or key in by_bridge[n]:
            raise ValueError("exact distinct declared matched arms required")
        by_bridge[n][key] = row
    if any(set(rows) != expected for rows in by_bridge.values()):
        raise ValueError("all32 matched executions per reference horizon required")
    summaries = []
    baseline = by_bridge[0]
    for n, arm in by_bridge.items():
        rescues, regressions = [], []
        conditions: dict[tuple[int, float], dict[str, int]] = {}
        for key in sorted(expected):
            old = baseline[key]["result"]["return_0p1m_and_healthy"]
            new = arm[key]["result"]["return_0p1m_and_healthy"]
            if type(old) is not bool or type(new) is not bool:
                raise ValueError("actual bool precise/healthy outcomes required")
            if new and not old:
                rescues.append(arm[key]["index"])
            if old and not new:
                regressions.append(arm[key]["index"])
            cluster = conditions.setdefault(key[:2], dict(baseline=0, candidate=0))
            cluster["baseline"] += int(old)
            cluster["candidate"] += int(new)
        winners = [r["index"] for r in arm.values() if r["result"]["return_0p1m_and_healthy"]]
        errors = [
            float(r["result"]["minimum_return_goal_error_3d_m"])
            for r in arm.values()
            if r["result"]["minimum_return_goal_error_3d_m"] is not None
        ]
        if not all(np.isfinite(x) and x >= 0 for x in errors):
            raise ValueError("finite observed error or explicit unavailable value required")
        summaries.append(
            dict(
                reference_bridge_controls=n,
                executions=32,
                distinct_initial_conditions=8,
                terminations=dict(Counter(r["termination"] for r in arm.values())),
                handoffs=sum(r["result"]["handoff_executed"] for r in arm.values()),
                outgoing_strikes=sum(r["result"]["outgoing_foot_strike"] for r in arm.values()),
                precise_healthy=len(winners),
                precise_indices=sorted(winners),
                observed_minimum_error_3d_m=min(errors) if errors else None,
                precise_rescues=sorted(rescues),
                precise_regressions=sorted(regressions),
                precise_net_change=len(rescues) - len(regressions),
                initial_condition_results=[
                    dict(context=k[0], lateral_m=k[1], **v) for k, v in sorted(conditions.items())
                ],
                initial_conditions_with_any_precise_result=sum(
                    v["candidate"] > 0 for v in conditions.values()
                ),
                clustered_conditions_improved=sum(
                    v["candidate"] > v["baseline"] for v in conditions.values()
                ),
                clustered_conditions_regressed=sum(
                    v["candidate"] < v["baseline"] for v in conditions.values()
                ),
            )
        )
    return dict(
        summaries=summaries,
        total_executions=128,
        distinct_initial_conditions=8,
        train_only=True,
        Fresh_verified=False,
        policy_updates=0,
        whole_game_verified=False,
        promotion_authorized=False,
        goal_complete=False,
    )
