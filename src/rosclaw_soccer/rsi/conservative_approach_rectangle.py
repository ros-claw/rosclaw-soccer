"""Learn a causal, abstaining SIM_ONLY approach gate from independent outcomes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

GUARDED_CV_HASH = "sha256:50e98c7810bf672d21a52cd6d33703dd729aa4e62f07757da0e4187c4ff1bd15"


@dataclass(frozen=True)
class ApproachRectangle:
    """No motor authority; caller may request only a bounded simulation option."""

    x_max_m: float | None
    y_min_m: float | None
    support: int
    training_gain: float

    def choose(
        self,
        forward_gap_m: float,
        lateral_gap_m: float,
        *,
        x_margin_m: float = 0.0,
        y_margin_m: float = 0.0,
    ) -> bool:
        if (
            not np.isfinite(forward_gap_m)
            or not np.isfinite(lateral_gap_m)
            or x_margin_m not in (0.0, 0.05)
            or y_margin_m not in (0.0, 0.02)
        ):
            raise ValueError("finite measured frame-zero ball context required")
        return bool(
            self.x_max_m is not None
            and self.y_min_m is not None
            and forward_gap_m <= self.x_max_m - x_margin_m
            and self.y_min_m + y_margin_m <= lateral_gap_m < 0.0
        )


def _cuts(values: np.ndarray[Any, Any]) -> tuple[float, ...]:
    unique = np.unique(values)
    cuts = np.r_[unique[0] - 1e-6, (unique[:-1] + unique[1:]) / 2, unique[-1] + 1e-6]
    return tuple(float(value) for value in cuts)


def fit_approach_rectangle(
    context: np.ndarray[Any, Any],
    reward_gain: np.ndarray[Any, Any],
    clean_loss: np.ndarray[Any, Any],
    new_out_of_play: np.ndarray[Any, Any],
) -> ApproachRectangle:
    """Enumerate two-coordinate rules using training-only rewards and safety labels."""
    x = np.asarray(context, dtype=float)
    gain = np.asarray(reward_gain, dtype=float)
    clean = np.asarray(clean_loss, dtype=bool)
    out = np.asarray(new_out_of_play, dtype=bool)
    if (
        x.ndim != 2
        or x.shape[1] != 2
        or len(x) < 12
        or gain.shape != (len(x),)
        or clean.shape != (len(x),)
        or out.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(gain).all()
    ):
        raise ValueError("finite independent causal paired courses required")
    negative = x[:, 1] < 0.0
    if np.count_nonzero(negative) < 3:
        return ApproachRectangle(None, None, 0, 0.0)
    best = ApproachRectangle(None, None, 0, 0.0)
    for x_max in _cuts(x[negative, 0]):
        for y_min in _cuts(x[negative, 1]):
            selected = negative & (x[:, 0] <= x_max) & (x[:, 1] >= y_min)
            support = int(np.count_nonzero(selected))
            if support < 3 or np.any(clean[selected]) or np.any(out[selected]):
                continue
            total_gain = float(np.sum(gain[selected]))
            if total_gain < 1.0:
                continue
            if (
                total_gain > best.training_gain + 1e-9
                or (
                    abs(total_gain - best.training_gain) <= 1e-9
                    and (best.support == 0 or support < best.support)
                )
                or (
                    abs(total_gain - best.training_gain) <= 1e-9
                    and support == best.support
                    and best.x_max_m is not None
                    and x_max < best.x_max_m
                )
            ):
                best = ApproachRectangle(x_max, y_min, support, total_gain)
    return best


def load_guarded_approach_policy(path: Path) -> tuple[ApproachRectangle, str]:
    """Load only the sealed simulation policy; never deserialize executable state."""
    policy = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema",
        "activation_ceiling",
        "cv_report_hash",
        "x_max_m",
        "y_min_m",
        "navigation_lateral_ball_gain",
        "promotion_authorized",
        "policy_hash",
    }
    x_max = policy.get("x_max_m")
    y_min = policy.get("y_min_m")
    if (
        set(policy) != expected_keys
        or policy.get("schema") != "rsi_isaac_guarded_approach_policy_v1"
        or policy.get("activation_ceiling") != "SIM_ONLY"
        or policy.get("cv_report_hash") != GUARDED_CV_HASH
        or policy.get("navigation_lateral_ball_gain") != 0.8
        or policy.get("promotion_authorized") is not False
        or not isinstance(x_max, (int, float))
        or not isinstance(y_min, (int, float))
        or isinstance(x_max, bool)
        or isinstance(y_min, bool)
        or not np.isfinite(x_max)
        or not np.isfinite(y_min)
        or not 2.0 <= x_max <= 2.6
        or not -0.15 <= y_min < 0.0
        or policy.get("policy_hash")
        != hash_json({key: value for key, value in policy.items() if key != "policy_hash"})
    ):
        raise ValueError("unsealed or unsafe SIM_ONLY guarded approach policy")
    return ApproachRectangle(float(x_max), float(y_min), 0, 0.0), policy["policy_hash"]
