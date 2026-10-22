"""Fixed development perturbations for contact generalization, never Fresh.

This declaration is independent of candidate outcomes. Every sign combination
is retained, including failures. It changes neither old exams nor body assets.
"""

import copy
import itertools
import math
from typing import Any


def neighborhood_config() -> dict[str, Any]:
    return dict(
        protocol="paid.contact_neighborhood.development.v1",
        source_cases=[4, 5],
        position_forward_offset_m=0.02,
        position_lateral_offset_m=0.02,
        velocity_forward_offset_mps=0.04,
        velocity_lateral_offset_mps=0.04,
        full_factorial_signs=[-1, 1],
        distinct_declared_contexts=32,
        statistical_independence_not_established=True,
        arms=["frozen_parent", "learned_memory", "fixed_successful_teacher"],
        repeats=2,
        maximum_episodes=192,
        maximum_native_steps=746880,
        kick_horizon_controls=289,
        stand_horizon_controls=100,
        native_dt_s=0.002,
        native_steps_per_control=10,
        task_quality_contract_unchanged=True,
        source_teacher_reoptimized=False,
        optimizer_updates=0,
        Fresh_rows_read=0,
        bank="development_not_fresh",
        runtime_case_id_input=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def contact_neighborhood_cases(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """32 exact signed perturbations in the original goal-relative frame.

    Positions and velocities are actual initial conditions, NOT policy input
    padding. Physical preflight must verify every compiled entry before rollout.
    """
    if (
        not isinstance(sources, list)
        or len(sources) != 2
        or [s.get("case") for s in sources] != [4, 5]
        or len({s.get("source") for s in sources}) != 1
    ):
        raise ValueError("exact source pair4/5 required")
    cases: list[dict[str, Any]] = []
    for original in sources:
        if original.get("profile") != "cached-standard" or original.get("ball_mass_kg") != 0.43:
            raise ValueError("unchanged legacy body/ball profile required")
        for key, width in (
            ("ball_world_initial", 3),
            ("velocity_world_initial", 2),
            ("goal_world", 3),
        ):
            value = original.get(key)
            if (
                not isinstance(value, list)
                or len(value) != width
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
            ):
                raise ValueError("explicit finite source geometry/velocity required")
        ball, goal, velocity = [
            original[k] for k in ("ball_world_initial", "goal_world", "velocity_world_initial")
        ]
        distance = math.hypot(*goal[:2])
        if ball[2] != 0.11 or goal[2] != 0.11 or distance < 4 or math.hypot(*velocity) > 0.6:
            raise ValueError("original standard-ball geometry and bounded velocity required")
        forward = [v / distance for v in goal[:2]]
        side = [-forward[1], forward[0]]
        for signs in itertools.product((-1, 1), repeat=4):
            row = copy.deepcopy(original)
            row.update(
                case=len(cases),
                source_case=original["case"],
                signs=list(signs),
                bank="development_not_fresh",
                not_fresh=True,
            )
            row["ball_world_initial"] = [
                ball[j] + 0.02 * (signs[0] * forward[j] + signs[1] * side[j]) for j in range(2)
            ] + [ball[2]]
            row["velocity_world_initial"] = [
                velocity[j] + 0.04 * (signs[2] * forward[j] + signs[3] * side[j]) for j in range(2)
            ]
            if math.hypot(*row["velocity_world_initial"]) > 0.6:
                raise ValueError("entire declared bank rejected, never clip a boundary course")
            cases.append(row)
    identities = {
        (
            c["source"],
            tuple(c["ball_world_initial"]),
            tuple(c["velocity_world_initial"]),
            tuple(c["goal_world"]),
        )
        for c in cases
    }
    if len(identities) != 32:
        raise ValueError("duplicate physical contexts cannot inflate the declaration")
    return cases
