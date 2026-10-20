"""Separate near-ball holdout generated after the motion router was frozen.

Not the canonical far-ball RSI/private Fresh exam. Six declared strata include
all failures; no outcome-driven clipping, replacement or sampling is permitted.
"""

from dataclasses import asdict, dataclass

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

GENERATOR_SEED = 2026102007


@dataclass(frozen=True)
class MotionPortfolioHoldoutCourse:
    agent_id: str
    seed: int
    course_index: int
    stratum: str
    course: tuple[float, float, float]


def motion_portfolio_holdout_courses() -> tuple[MotionPortfolioHoldoutCourse, ...]:
    rng = np.random.default_rng(GENERATOR_SEED)
    rows: list[MotionPortfolioHoldoutCourse] = []
    for side, (low, high) in enumerate(((0.30, 0.44), (0.44, 0.58))):
        for speed_name in ("approaching", "stationary", "receding"):
            for _ in range(32):
                x, y = float(rng.uniform(low, high)), float(rng.uniform(-0.24, 0.04))
                speed = 0.0 if speed_name == "stationary" else float(rng.uniform(0.14, 0.16))
                if speed_name == "approaching":
                    speed = -speed
                index = len(rows)
                rows.append(
                    MotionPortfolioHoldoutCourse(
                        "tracker",
                        GENERATOR_SEED + index,
                        index,
                        ("low" if side == 0 else "high") + "_" + speed_name,
                        (x, y, speed),
                    )
                )
    return tuple(rows)


def preflight_motion_portfolio_holdout(courses: tuple[MotionPortfolioHoldoutCourse, ...]) -> str:
    expected = motion_portfolio_holdout_courses()
    if type(courses) is not tuple or len(courses) != 192:
        raise ValueError("entire declared192 near holdout required")
    for item, gold in zip(courses, expected, strict=True):
        if (
            type(item) is not MotionPortfolioHoldoutCourse
            or type(item.agent_id) is not str
            or any(type(x) is not int for x in (item.seed, item.course_index))
            or type(item.stratum) is not str
            or type(item.course) is not tuple
            or len(item.course) != 3
            or any(type(x) is not float or not np.isfinite(x) for x in item.course)
            or item != gold
        ):
            raise ValueError(
                "exact unique stratified holdout declaration required; never clip or replace"
            )
    if len({c.course for c in courses}) != 192:
        raise ValueError("distinct declared holdout states required")
    return hash_json(
        dict(
            schema="soccer.near_motion_portfolio_holdout_preflight.v1",
            courses=[asdict(c) for c in courses],
            generator_seed=GENERATOR_SEED,
            original_rsi_exam=False,
            private_fresh_accessed=False,
            physics_qualified=False,
            hardware_authorized=False,
        )
    )
