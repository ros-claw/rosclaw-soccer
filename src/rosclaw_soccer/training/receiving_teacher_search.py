"""Bounded physical-feedback search for receiving teachers, not motor promotion.

The seven parameters coordinate existing foot control and navigation. Search
does not change bodies, ball state, capture criteria or actuator limits. A
caller must collect and authenticate complete physical courses before ranking.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from rosclaw_soccer.growth.locomotion_contact_teacher import G1LocomotionContactTeacherConfig
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig

PARAMETERS = (
    "aim_yaw_bias_rad",
    "committed_receive_aim_yaw_bias_rad",
    "receive_cushion_depth_m",
    "committed_receive_ankle_lateral_offset_m",
    "committed_receive_velocity_damping_n_per_mps",
    "contact_leg_stiffness_scale",
    "receive_pacing_ratio",
)
LOWER = (-1.0, -1.0, -0.12, 0.12, 5.0, 0.4, 0.2)
UPPER = (1.0, 1.0, 0.12, 0.24, 15.0, 1.0, 0.8)


def _vector(value: np.ndarray) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (7,)
        or value.dtype.kind != "f"
        or not np.isfinite(value).all()
        or (abs(value) > 1).any()
    ):
        raise ValueError("finite seven-dimensional normalized teacher vector required")
    return value.astype(np.float64, copy=True)


def normalized_teacher(values: Sequence[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (7,) or not np.isfinite(raw).all():
        raise ValueError("seven finite physical teacher parameters required")
    lower, upper = np.asarray(LOWER), np.asarray(UPPER)
    if ((raw < lower) | (raw > upper)).any():
        raise ValueError("receiving teacher parameters exceed existing bounds")
    return _vector(2 * (raw - lower) / (upper - lower) - 1)


def teacher_parameters(vector: np.ndarray) -> dict[str, float]:
    raw = np.asarray(LOWER) + (_vector(vector) + 1) * (np.asarray(UPPER) - LOWER) / 2
    return dict(zip(PARAMETERS, map(float, raw), strict=True))


def apply_receiving_teacher(
    vector: np.ndarray,
    world: IndependentTeamWorldConfig,
    teacher: G1LocomotionContactTeacherConfig,
) -> tuple[IndependentTeamWorldConfig, G1LocomotionContactTeacherConfig]:
    if (
        not isinstance(world, IndependentTeamWorldConfig)
        or not isinstance(teacher, G1LocomotionContactTeacherConfig)
        or not world.loose_ball_capture_follow_navigation
        or teacher.one_touch_finish_enabled
    ):
        raise ValueError("explicit native follow receiving classroom required")
    values = teacher_parameters(vector)
    pace = values.pop("receive_pacing_ratio")
    return replace(world, receive_pacing_ratio=pace), replace(
        teacher,
        aim_yaw_bias_rad=values["aim_yaw_bias_rad"],
        committed_receive_aim_yaw_bias_rad=values["committed_receive_aim_yaw_bias_rad"],
        receive_cushion_depth_m=values["receive_cushion_depth_m"],
        committed_receive_ankle_lateral_offset_m=values["committed_receive_ankle_lateral_offset_m"],
        committed_receive_velocity_damping_n_per_mps=values[
            "committed_receive_velocity_damping_n_per_mps"
        ],
        contact_leg_stiffness_scale=values["contact_leg_stiffness_scale"],
    )


def sample_teacher_population(
    *, mean: np.ndarray, sigma: np.ndarray, incumbent: np.ndarray, seed: int, count: int
) -> np.ndarray:
    center, best = _vector(mean), _vector(incumbent)
    if (
        not isinstance(sigma, np.ndarray)
        or sigma.shape != (7,)
        or sigma.dtype.kind != "f"
        or not np.isfinite(sigma).all()
        or ((sigma < 0.05) | (sigma > 1)).any()
        or type(count) is not int
        or not 4 <= count <= 32
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise ValueError("bounded reproducible receiving population required")
    candidates = np.clip(np.random.default_rng(seed).normal(center, sigma, (count, 7)), -1, 1)
    candidates[0] = best
    return candidates


def receiving_feedback_rank(
    rows: Sequence[Mapping[str, Any]], *, expected_courses: tuple[str, ...]
) -> tuple[int, int, int, float, float]:
    """Lexicographic physical safety, worst-role successes, total, then shaping.

    Course completeness is mandatory. Dense reward cannot outweigh a fall or
    substitute for controlled reception. This is development search, not a
    confidence estimate, evidence authenticator or promotion permission.
    """
    if not expected_courses or len(set(expected_courses)) != len(expected_courses):
        raise ValueError("unique explicit expected receiving courses required")
    seen: set[str] = set()
    by_agent: dict[str, int] = {}
    safe = True
    rewards = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("course_id"), str)
            or row["course_id"] not in expected_courses
            or row["course_id"] in seen
            or not isinstance(row.get("agent_id"), str)
            or not row["agent_id"]
            or type(row.get("safe")) is not bool
            or type(row.get("controlled_reception")) is not bool
            or type(row.get("shaped_return")) not in (int, float)
            or not np.isfinite(row["shaped_return"])
        ):
            raise ValueError("complete finite physical receiving feedback required")
        seen.add(row["course_id"])
        safe = safe and row["safe"]
        by_agent[row["agent_id"]] = by_agent.get(row["agent_id"], 0) + int(
            row["safe"] and row["controlled_reception"]
        )
        rewards.append(float(row["shaped_return"]))
    if seen != set(expected_courses):
        raise ValueError("missing receiving failures cannot be omitted")
    return int(safe), min(by_agent.values()), sum(by_agent.values()), min(rewards), sum(rewards)
