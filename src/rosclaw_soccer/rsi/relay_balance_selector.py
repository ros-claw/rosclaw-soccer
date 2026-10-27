"""Offline development choice and score, not a safety gate or motor policy."""

import numpy as np

from rosclaw_soccer.rsi.relay_balance_context import relay_balance_context


def decision_rewards(success: np.ndarray, failure: np.ndarray, kick: np.ndarray) -> np.ndarray:
    if any(a.shape != (4,) or a.dtype != np.bool_ for a in (success, failure, kick)) or np.any(
        success & failure
    ):
        raise ValueError("four paired valid measured outcomes required")
    return (
        2.0 * success.astype(np.float32)
        + 0.25 * kick.astype(np.float32)
        - 2.0 * failure.astype(np.float32)
    )


def select_with_parent_fallback(values: np.ndarray, nearest_distance: float) -> tuple[int, str]:
    if (
        values.shape != (4,)
        or not np.isfinite(values).all()
        or not np.isfinite(nearest_distance)
        or nearest_distance < 0
    ):
        raise ValueError("finite four-action values and measured support distance required")
    if nearest_distance > 2.0:
        return 0, "OUTSIDE_TRAIN_SUPPORT"
    action = int(np.argmax(values))
    if action == 0 or values[action] < values[0] + 0.25:
        return 0, "PARENT_OR_SMALL_MARGIN"
    return action, "RESEARCH_TEACHER_CHOICE"


class RelayBalanceResearchSelector:
    """Frozen mathematical chooser; the caller must authenticate model and mode.

    An index is a SIM research proposal only. It grants no promotion, safety,
    real execution or authority to actuate a joint.
    """

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        names = ("center", "scale", "support", "weight1", "bias1", "weight2", "bias2")
        if set(arrays) != set(names):
            raise ValueError("exact trained model arrays required")
        values = [np.array(arrays[name], dtype=np.float32, copy=True) for name in names]
        center, scale, support, w1, b1, w2, b2 = values
        if (
            center.shape != (17,)
            or scale.shape != (17,)
            or support.ndim != 2
            or support.shape[1] != 17
            or support.shape[0] < 8
            or w1.shape != (32, 17)
            or b1.shape != (32,)
            or w2.shape != (4, 32)
            or b2.shape != (4,)
            or (scale < 0.05).any()
            or not all(np.isfinite(x).all() for x in values)
        ):
            raise ValueError("finite complete train-normalized selector required")
        for value in values:
            value.flags.writeable = False
        self.center, self.scale, self.support, self.w1, self.b1, self.w2, self.b2 = values

    def decide(self, q: np.ndarray, v: np.ndarray, return_goal: np.ndarray) -> dict:
        features = relay_balance_context(q, v, return_goal).astype(np.float32)
        z = np.clip((features - self.center) / self.scale, -8, 8).astype(np.float32)
        distance = float(np.sqrt(np.mean((self.support - z) ** 2, axis=1)).min())
        values = np.tanh(z @ self.w1.T + self.b1) @ self.w2.T + self.b2
        action, reason = select_with_parent_fallback(values, distance)
        return dict(
            action=action,
            teacher_index=action - 1,
            support_distance=distance,
            reason=reason,
            values=values.copy(),
            features=features.copy(),
            sim_research_only=True,
            hardware_authorized=False,
            promotion_authorized=False,
        )
