"""New post-freeze near generalization population, not canonical RSI Fresh.

Same six strata as the consumed portfolio192, with a different fixed stream.
No rejection sampling or outcome-dependent replacement. Whole96 preflight.
"""

from dataclasses import asdict, dataclass

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

GENERATOR_SEED = 2026102021


@dataclass(frozen=True)
class CategoricalMotionHoldoutCourse:
    agent_id: str
    seed: int
    course_index: int
    stratum: str
    course: tuple[float, float, float]


def categorical_motion_holdout_courses() -> tuple[CategoricalMotionHoldoutCourse, ...]:
    rng = np.random.default_rng(GENERATOR_SEED)
    rows: list[CategoricalMotionHoldoutCourse] = []
    for side, (low, high) in enumerate(((0.30, 0.44), (0.44, 0.58))):
        for kind in ("approaching", "stationary", "receding"):
            for _ in range(16):
                x, y = float(rng.uniform(low, high)), float(rng.uniform(-0.24, 0.04))
                vx = 0.0 if kind == "stationary" else float(rng.uniform(0.14, 0.16))
                if kind == "approaching":
                    vx = -vx
                i = len(rows)
                rows.append(
                    CategoricalMotionHoldoutCourse(
                        "tracker",
                        GENERATOR_SEED + i,
                        i,
                        ("low" if side == 0 else "high") + "_" + kind,
                        (x, y, vx),
                    )
                )
    return tuple(rows)


def preflight_categorical_motion_holdout(
    courses: tuple[CategoricalMotionHoldoutCourse, ...],
) -> str:
    expected = categorical_motion_holdout_courses()
    if type(courses) is not tuple or len(courses) != 96:
        raise ValueError("entire declared96 post-freeze near population required")
    for item, gold in zip(courses, expected, strict=True):
        if (
            type(item) is not CategoricalMotionHoldoutCourse
            or type(item.agent_id) is not str
            or type(item.stratum) is not str
            or any(type(v) is not int for v in (item.seed, item.course_index))
            or type(item.course) is not tuple
            or len(item.course) != 3
            or any(type(v) is not float or not np.isfinite(v) for v in item.course)
            or item != gold
        ):
            raise ValueError("exact typed immutable96 required; no clipping/dropping/replacement")
    from rosclaw_soccer.training.motion_portfolio_holdout import motion_portfolio_holdout_courses

    states = {c.course for c in courses}
    if len(states) != 96 or states & {c.course for c in motion_portfolio_holdout_courses()}:
        raise ValueError("unique new states, not consumed192, required")
    return hash_json(
        dict(
            schema="soccer.categorical_motion_near96_preflight.v1",
            courses=[asdict(c) for c in courses],
            generator_seed=GENERATOR_SEED,
            original_rsi_exam=False,
            private_fresh_accessed=False,
            physics_qualified=False,
            hardware_authorized=False,
        )
    )
