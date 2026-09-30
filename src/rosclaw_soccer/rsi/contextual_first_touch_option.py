"""SIM_ONLY contextual action selection from audited counterfactual episodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

ACTION_NAMES = ("parent", "cap005", "cap015")
FEATURE_COUNT = 9
NEIGHBORS = 3
PRIOR_WEIGHT = 2.0
DISTANCE_EPSILON = 0.1
STD_FLOOR = 0.1
MAX_IN_PLAY_LATERAL_EXCURSION_M = 4.0


def first_touch_reward(arm: dict[str, Any]) -> float:
    """Score physical outcome; non-foot contacts cannot be bought with ball travel."""
    pelvis = arm.get("minimum_pelvis_z_m")
    bodies = arm.get("contact_body_indices")
    first = arm.get("first_contact_frame")
    if (
        not isinstance(pelvis, (int, float))
        or isinstance(pelvis, bool)
        or not math.isfinite(pelvis)
        or not isinstance(bodies, list)
        or any(type(body) is not int or body not in range(6) for body in bodies)
        or (
            first is not None
            and (not isinstance(first, int) or isinstance(first, bool) or not 0 <= first < 300)
        )
    ):
        raise ValueError("invalid authenticated first-touch outcome")
    if pelvis < 0.65:
        return -10.0
    if first is None or not bodies:
        return -3.0
    if not set(bodies) <= {0, 1}:
        return -2.0
    forward = arm.get("forward_60_m")
    lateral = arm.get("lateral_60_m")
    excursion = arm.get("max_lateral_excursion_m")
    if (
        not isinstance(forward, (int, float))
        or isinstance(forward, bool)
        or not isinstance(lateral, (int, float))
        or isinstance(lateral, bool)
        or not math.isfinite(forward)
        or not math.isfinite(lateral)
        or not isinstance(excursion, (int, float))
        or isinstance(excursion, bool)
        or not math.isfinite(excursion)
        or excursion < 0
    ):
        raise ValueError("missing measured 60-frame trajectory for foot-only contact")
    if excursion > MAX_IN_PLAY_LATERAL_EXCURSION_M:
        return -4.0
    ratio = abs(lateral) / max(forward, 0.01)
    return (
        2.0
        + min(max(forward, 0.0), 2.0)
        - 0.8 * abs(lateral)
        - 0.5 * min(ratio, 5.0)
        - (1.0 if forward <= 0 else 0.0)
    )


@dataclass(frozen=True)
class ContextualOptionSelector:
    """Small, regularized, measured-state option selector; no motor authority."""

    feature_mean: np.ndarray[Any, Any]
    feature_scale: np.ndarray[Any, Any]
    training_features: np.ndarray[Any, Any]
    training_rewards: np.ndarray[Any, Any]
    prior_rewards: np.ndarray[Any, Any]

    def value(self, query: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        observation = np.asarray(query, dtype=np.float64)
        if observation.shape != (FEATURE_COUNT,) or not np.isfinite(observation).all():
            raise ValueError("finite nine-feature precontact observation required")
        normalized = (observation - self.feature_mean) / self.feature_scale
        distance = np.linalg.norm(self.training_features - normalized, axis=1)
        nearest = np.argsort(distance, kind="stable")[:NEIGHBORS]
        weights = 1.0 / (DISTANCE_EPSILON + distance[nearest])
        return cast(
            np.ndarray[Any, Any],
            (weights @ self.training_rewards[nearest] + PRIOR_WEIGHT * self.prior_rewards)
            / (float(np.sum(weights)) + PRIOR_WEIGHT),
        )

    def choose(self, query: np.ndarray[Any, Any]) -> int:
        """Index 0 is parent and wins exact ties; all options remain SIM_ONLY."""
        return int(np.argmax(self.value(query)))


def fit_contextual_option(
    features: np.ndarray[Any, Any], rewards: np.ndarray[Any, Any]
) -> ContextualOptionSelector:
    """Fit only on the passed training fold; caller owns split integrity."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(rewards, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[0] < NEIGHBORS
        or x.shape[1] != FEATURE_COUNT
        or y.shape != (len(x), len(ACTION_NAMES))
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("finite independent training episodes and three arms required")
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), STD_FLOOR)
    return ContextualOptionSelector(mean, scale, (x - mean) / scale, y, np.mean(y, axis=0))
