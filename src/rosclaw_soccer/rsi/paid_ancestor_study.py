"""New TRAIN contexts for same-entry ancestor competence, not consumed Fresh fit."""

from typing import Any

import numpy as np

SCHEDULES = {
    "grown": ("base", "student", "student"),
    "ancestor_third": ("base", "student", "base"),
    "r2_all": ("base", "base", "base"),
}


def ancestor_train_cases(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if (
        len(sources) != 16
        or [c["case"] for c in sources] != list(range(16))
        or len({c["source"] for c in sources}) != 8
        or any(sources[2 * i]["source"] != sources[2 * i + 1]["source"] for i in range(8))
    ):
        raise ValueError("all original TRAIN source entries required")
    rng = np.random.Generator(np.random.PCG64(2026102211))
    cases: list[dict[str, Any]] = []
    entries = [(row, (-1 if i % 2 == 0 else 1), "balanced") for i, row in enumerate(sources)]
    target = [c for c in sources if c["source"] == "soccer-standard-009_right.npz"]
    if len(target) != 2:
        raise ValueError("declared source009 pair required, no source replacement")
    entries += [(target[1], 1, "source009_coverage") for _ in range(4)]
    for row, sign, group in entries:
        ball, goal = (
            np.asarray(row["ball_world_initial"], float),
            np.asarray(row["goal_world"], float),
        )
        if (
            row["profile"] != "cached-standard"
            or row["ball_mass_kg"] != 0.43
            or ball.shape != (3,)
            or goal.shape != (3,)
            or not np.isfinite(ball).all()
            or not np.isfinite(goal).all()
            or ball[2] != 0.11
            or goal[2] != 0.11
            or np.linalg.norm(goal[:2]) < 4
        ):
            raise ValueError("unchanged original TRAIN physical recipe required")
        forward = goal[:2] / np.linalg.norm(goal[:2])
        side = np.array([-forward[1], forward[0]])
        position = ball.copy()
        position[:2] += rng.uniform(-0.04, 0.04) * forward + rng.uniform(-0.04, 0.04) * side
        speed = rng.uniform(0.13, 0.29) if group == "balanced" else rng.uniform(0.08, 0.32)
        velocity = sign * speed * forward + rng.uniform(-0.05, 0.05) * side
        angle = rng.uniform(-0.04, 0.04)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        target_goal = goal.copy()
        target_goal[:2] = rot @ target_goal[:2]
        cases.append(
            dict(
                case=len(cases),
                source=row["source"],
                profile="cached-standard",
                ball_mass_kg=0.43,
                ball_world_initial=position.tolist(),
                velocity_world_initial=velocity.tolist(),
                goal_world=target_goal.tolist(),
                cohort=group,
                bank="train",
                not_fresh=True,
            )
        )
    return cases


def case_condition(case: dict[str, Any]) -> dict[str, Any]:
    return {
        k: case[k] for k in ("source", "ball_world_initial", "velocity_world_initial", "goal_world")
    }


def ancestor_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summaries retain partial legal worlds, including no third kick."""
    return dict(
        stable=sum(r["stable"] for r in rows),
        quality_by_kick=[
            sum(len(r["quality_by_kick"]) > k and r["quality_by_kick"][k] for r in rows)
            for k in range(3)
        ],
    )
