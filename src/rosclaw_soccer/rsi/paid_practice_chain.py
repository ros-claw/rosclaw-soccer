"""Reusable SIM-only physical skill chains, not a disguised autonomous match.

Each world has one initial body setup. Subsequent skills preserve the actual
integration state and previous executed action. The declared practice fixture
may return only the BALL. Artifact persistence and task grading stay outside.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_skill_handoff import practice_ball_drop, snapshot


def require_identical_snapshot(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> None:
    if set(before) != set(after) or any(not np.array_equal(before[k], after[k]) for k in before):
        raise ValueError("skill handoff changed physical integration state")


def advance_practice_boundary(
    ep: Any,
    clip: Any,
    *,
    foot: str,
    store: Callable[[str, dict[str, np.ndarray]], dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    """Reuse exactly the previous bounded drop/settle/body-preserving boundary."""
    if ep.termination_reason != "REFERENCE_COMPLETE":
        raise ValueError("only a healthy complete prior skill may begin another")
    artifacts: list[dict[str, Any]] = []
    try:
        drop = practice_ball_drop(ep, foot=foot)
    except ValueError as error:
        if str(error) != "declared practice drop overlaps body; retain failure, no resample":
            raise
        return dict(status="FIXTURE_REJECTED", reason=str(error), artifacts=artifacts)
    artifacts.append(store(prefix + "-drop", drop))
    settled = True
    if bool(drop["performed"]):
        holds = []
        for _ in range(50):
            holds.append(ep.advance_hold())
            q, v = ep.layout.canonical(ep.env.data)
            if holds[-1]["body_failed"] or (q[38] <= 0.12 and abs(v[37]) <= 0.1):
                break
        artifacts.append(
            store(prefix + "-hold", {k: np.array([r[k] for r in holds]) for k in holds[0]})
        )
        settled = bool(q[38] <= 0.12 and abs(v[37]) <= 0.1)
    if ep.termination_reason != "REFERENCE_COMPLETE":
        return dict(status="HOLD_BODY_FAILURE", artifacts=artifacts)
    if not settled:
        return dict(status="BALL_NOT_SETTLED", artifacts=artifacts)
    before = snapshot(ep.env.data)
    previous = ep.last_action.copy()
    goal = ep.goal.copy()
    ep.begin_current_state(clip, goal=goal)
    after = snapshot(ep.env.data)
    require_identical_snapshot(before, after)
    if not np.array_equal(ep.last_action, previous) or not np.array_equal(ep.goal, goal):
        raise ValueError("physical handoff erased executed action or moved world goal")
    artifacts.append(
        store(
            prefix + "-boundary",
            {
                **{"before_" + k: v for k, v in before.items()},
                **{"after_" + k: v for k, v in after.items()},
                "previous_actual_action": previous,
            },
        )
    )
    return dict(status="READY", artifacts=artifacts)


def run_practice_chain(
    bench: Any,
    case: dict[str, Any],
    actors: list[Any],
    *,
    store: Callable[[str, dict[str, np.ndarray]], dict[str, Any]],
) -> dict[str, Any]:
    """Execute 2..4 full learned skills. No actor/role/case winner selection.

    All actor weights are caller-pinned; each recurrent invocation starts its
    own h/c at zero. Motion reference resets, simulator state NEVER does. All
    ordinary failures are returned, not resampled or converted to task success.
    """
    if not 2 <= len(actors) <= 4:
        raise ValueError("bounded predeclared two-to-four-skill chain required")
    source = case["source"]
    if ("left" in source) == ("right" in source):
        raise ValueError("one explicit admitted reference contact foot required")
    clip = bench.clips[source]
    ep = bench.make(case)
    skills: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    status = "STARTED"
    for index, actor in enumerate(actors):
        if index:
            boundary = advance_practice_boundary(
                ep,
                clip,
                foot="left" if "left" in source else "right",
                store=store,
                prefix=f"between{index - 1}-{index}",
            )
            boundaries.append(boundary)
            if boundary["status"] != "READY":
                status = boundary["status"]
                break
        start_native = ep.native_steps
        raw, row = bench.collect_current(ep, actor)
        if any(k.startswith("PLAN_") for k in raw) or not np.array_equal(
            raw["NN_action"], raw["NN_nominal_action"]
        ):
            raise ValueError("chain exams must execute learned neural actors, not teacher mixtures")
        artifact = store(f"skill{index}", raw)
        terminal = store(f"terminal{index}", snapshot(ep.env.data))
        skills.append(dict(index=index, result=dict(row, **artifact), terminal=terminal))
        if ep.native_steps - start_native != row["native_steps"]:
            raise ValueError("per-skill native work accounting differs")
        if row["body_failed"]:
            status = "SKILL_BODY_FAILURE"
            break
        status = "ALL_SKILLS_EXECUTED" if index == len(actors) - 1 else "SKILL_COMPLETE"
    if not np.isclose(float(ep.env.data.time), ep.native_steps * 0.002, rtol=0, atol=1e-10):
        raise ValueError("chain physical clock or total native work changed")
    return dict(
        status=status,
        requested_skills=len(actors),
        skills=skills,
        boundaries=boundaries,
        executed_skills=len(skills),
        native_steps=ep.native_steps,
        neural_rows=sum(s["result"]["neural_rows"] for s in skills),
        final_sim_time=float(ep.env.data.time),
        robot_initializations=1,
        body_resets_between_skills=0,
        previous_actual_action_carried=True,
        reference_clock_reset=True,
        recurrent_memory_policy="private_h0_each_skill",
        world_stable=status == "ALL_SKILLS_EXECUTED",
        fixture_ball_returns_are_not_teammate_passes=True,
        goal_complete=False,
        team_game=False,
        hardware_authorized=False,
    )
