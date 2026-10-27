"""Separate historical relay rewards from observed nonfoot-contact evidence."""

import math

import numpy as np


def receiver_contact_prefix(physical_forces: np.ndarray, controls: int) -> dict:
    """Measure contact during actual receiver control, not after its handoff."""
    if (
        not isinstance(physical_forces, np.ndarray)
        or physical_forces.ndim != 4
        or physical_forces.shape[1:] != (10, 2, 3)
        or type(controls) is not int
        or not 1 <= controls <= len(physical_forces)
        or not np.isfinite(physical_forces).all()
        or (physical_forces < 0).any()
    ):
        raise ValueError("finite real receiver-control prefix and measured contact forces required")
    foot = physical_forces[:controls, :, 1, :2]
    other = physical_forces[:controls, :, 1, 2]
    return dict(
        foot_contact_over_1n=bool((foot > 1).any()),
        nonfoot_contact_over_1n=bool((other > 1).any()),
        first_nonfoot_substep=(
            int(np.flatnonzero(other.reshape(-1) > 1)[0]) if (other > 1).any() else None
        ),
        receiver_control_prefix=controls,
    )


def summarize_reward_purity(rows: list[dict]) -> dict:
    """Do not replace any original reward or imply clean contact beyond the prefix."""
    if not rows:
        raise ValueError("all sampled episodes required")
    counts = {
        key: {"total": 0, "nonfoot_in_prefix": 0, "no_nonfoot_observed_in_prefix": 0}
        for key in ("handoff", "precise_healthy", "positive_reward")
    }
    for row in rows:
        if (
            type(row.get("handoff")) is not bool
            or type(row.get("precise_healthy")) is not bool
            or type(row.get("nonfoot_contact_over_1n")) is not bool
            or not isinstance(row.get("reward"), (float, int))
            or isinstance(row["reward"], bool)
            or not math.isfinite(row["reward"])
        ):
            raise ValueError("explicit measured episode outcomes and reward required")
        for key, selected in (
            ("handoff", row["handoff"]),
            ("precise_healthy", row["precise_healthy"]),
            ("positive_reward", row["reward"] > 0),
        ):
            if selected:
                counts[key]["total"] += 1
                field = (
                    "nonfoot_in_prefix"
                    if row["nonfoot_contact_over_1n"]
                    else "no_nonfoot_observed_in_prefix"
                )
                counts[key][field] += 1
    return counts


def validate_reward_purity_profile(summary: dict, profile: str) -> None:
    """Explicit legacy use or reject positive examples with nonfoot prefix contact."""
    if profile not in ("legacy-mixed", "no-nonfoot-positive-prefix"):
        raise ValueError("explicit research contact profile required")
    try:
        positive = summary["positive_reward"]
        total = positive["total"]
        nonfoot = positive["nonfoot_in_prefix"]
        clear = positive["no_nonfoot_observed_in_prefix"]
    except (KeyError, TypeError) as error:
        raise ValueError("complete measured reward purity summary required") from error
    if (
        any(type(x) is not int or x < 0 for x in (total, nonfoot, clear))
        or total != nonfoot + clear
    ):
        raise ValueError("consistent non-negative reward purity counts required")
    if profile == "no-nonfoot-positive-prefix" and (not total or nonfoot):
        raise ValueError("positive episodes without nonfoot contact in the prefix required")
