"""Two-round, bounded three-dimensional response optimization; no physics here."""

import math
from typing import Any

import numpy as np


class SamplingExhausted(ValueError):
    def __init__(self, draws: list[dict[str, Any]]) -> None:
        super().__init__("CEM bounded rejection sampling exhausted; no clipping or partial batch")
        self.draws = draws


def bounded_offset(offset: Any) -> np.ndarray:
    a = np.asarray(offset, dtype=np.float64)
    if (
        a.shape != (3,)
        or not np.isfinite(a).all()
        or np.max(abs(a)) > 0.04
        or np.linalg.norm(a) > 0.06
    ):
        raise ValueError("declared contact proposal domain required; no clipping")
    return a.copy()


def response_merit(score: dict[str, Any]) -> float:
    for key in ("qualified", "physical_prefix_and_suffix_safe", "complete_post60_observable"):
        if type(score[key]) is not bool:
            raise ValueError("typed original research score required")
    keys = (
        "maximum_nonfoot_force_n",
        "minimum_pelvis_z_m",
        "minimum_upright",
        "forward_to_task_end_m",
        "lateral_to_task_end_m",
        "maximum_ball_vx_mps",
    )
    if any(type(score[k]) not in (int, float) or not math.isfinite(score[k]) for k in keys):
        raise ValueError("finite actual physical response required")
    drift = score["post_touch_maximum_root_drift_m"]
    if drift is not None and (
        type(drift) not in (int, float) or not math.isfinite(drift) or drift < 0
    ):
        raise ValueError("finite measured drift or explicit unobserved required")
    if score["qualified"]:
        cost = score["cost"]
        if (
            not score["physical_prefix_and_suffix_safe"]
            or not score["complete_post60_observable"]
            or type(cost) not in (int, float)
            or not math.isfinite(cost)
        ):
            raise ValueError("qualified response needs original safety/observation/cost")
        return float(1e6 - cost)
    if not score["physical_prefix_and_suffix_safe"]:
        return float(
            -1e6
            - min(score["maximum_nonfoot_force_n"], 1e4)
            - 100 * max(0, 0.65 - score["minimum_pelvis_z_m"])
            - 100 * max(0, 0.9 - score["minimum_upright"])
        )
    return float(
        100 * score["complete_post60_observable"]
        + 10 * np.clip(score["forward_to_task_end_m"], -3, 3)
        - 10 * abs(score["lateral_to_task_end_m"])
        + np.clip(score["maximum_ball_vx_mps"], 0, 3)
        - 0.1 * (drift or 0)
    )


def sample_four(
    rng: np.random.Generator, mean: np.ndarray, std: np.ndarray
) -> tuple[list[tuple[float, ...]], list[dict[str, Any]]]:
    bounded_offset(mean)
    if (
        type(std) is not np.ndarray
        or std.shape != (3,)
        or not np.isfinite(std).all()
        or np.any(std < 0.003)
        or np.any(std > 0.04)
    ):
        raise ValueError("declared finite CEM standard deviation required")
    samples, draws = [], []
    for _ in range(256):
        point = rng.normal(mean, std)
        accepted = bool(
            np.isfinite(point).all()
            and np.max(abs(point)) <= 0.04
            and np.linalg.norm(point) <= 0.06
        )
        draws.append(dict(offset_root_m=point.tolist(), accepted=accepted))
        if accepted:
            samples.append(tuple(float(x) for x in point))
            if len(samples) == 4:
                return samples, draws
    raise SamplingExhausted(draws)


def elite_update(
    points: list[tuple[float, ...]], merits: list[float]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if (
        len(points) != 5
        or len(merits) != 5
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in merits)
    ):
        raise ValueError("four new responses and one retained best required")
    data = np.asarray([bounded_offset(p) for p in points])
    elite = sorted(range(5), key=lambda i: (-merits[i], i))[:2]
    return data[elite].mean(axis=0), np.maximum(data[elite].std(axis=0), 0.003), elite


def choose_response_candidate(rows: list[dict[str, Any]]) -> int:
    if (
        len(rows) != 30
        or [r["law_index"] for r in rows] != [0, 0, 3, 3, 4, 4] + [0] * 8 + [3] * 8 + [4] * 8
    ):
        raise ValueError("six retained anchors plus24 declared responses required")
    if [r["offset_root_m"] for r in rows[:6]] != [[0.0, 0.0, 0.0], [0.0, -0.025, -0.025]] * 3:
        raise ValueError("all original zero/LOWER anchors must remain available")
    for row in rows:
        bounded_offset(row["offset_root_m"])
        response_merit(row["score"])
    eligible = [i for i, r in enumerate(rows) if r["score"]["qualified"]]
    return min(eligible, key=lambda i: (rows[i]["score"]["cost"], i)) if eligible else 0
