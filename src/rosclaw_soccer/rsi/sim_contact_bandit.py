"""Bounded offline contextual contact-policy interface, simulation only.

The inputs use privileged simulated ball state; outputs select a fixed set of
research residuals and are not an executable or hardware-authorized policy.
"""

import numpy as np

ACTION_NAMES = ("parent", *(f"hg{i}-{side}" for i in range(4) for side in ("minus", "plus")))


def contact_features(ball_y: float, pelvis_y: float, ball_vy: float) -> np.ndarray:
    values = np.asarray([ball_y, pelvis_y, ball_vy], dtype=np.float64)
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 10:
        raise ValueError("finite bounded measured simulation state required")
    return np.asarray([(ball_y - pelvis_y - 0.16) / 0.1, (ball_vy - 0.12) / 0.05], dtype=np.float32)


def offline_contact_reward(outcome: dict) -> float:
    """Fixed bounded training score; the unmodified physical exam remains authoritative."""
    if (
        not isinstance(outcome, dict)
        or type(outcome.get("safe")) is not bool
        or type(outcome.get("task_pass")) is not bool
    ):
        raise ValueError("complete force-derived H300 outcome required")
    if outcome["task_pass"] and (not outcome["safe"] or outcome.get("forward_60_m") is None):
        raise ValueError("passing task requires safe observed outcome")
    if not outcome["safe"]:
        return -2.0
    forward = outcome.get("forward_60_m")
    if forward is None:
        return -0.2
    if (
        isinstance(forward, bool)
        or not isinstance(forward, (int, float))
        or not np.isfinite(forward)
    ):
        raise ValueError("finite observed forward distance required")
    return float(outcome["task_pass"]) + 0.1 * float(np.clip(forward, -1.0, 1.0))


def select_contact_action(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> int:
    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if (
        x.shape != (2,)
        or w.shape != (9, 2)
        or b.shape != (9,)
        or not np.isfinite(x).all()
        or not np.isfinite(w).all()
        or not np.isfinite(b).all()
        or np.max(np.abs(x)) > 10
    ):
        raise ValueError("finite bounded state and nine-action actor required")
    return int(np.argmax(w @ x + b))


def action_delta(action: int) -> np.ndarray:
    if type(action) is not int or not 0 <= action < len(ACTION_NAMES):
        raise ValueError("exact discrete research action required")
    delta = np.zeros(29, dtype=np.float64)
    if action:
        delta[(action - 1) // 2] = -0.08 if action % 2 else 0.08
    return delta
