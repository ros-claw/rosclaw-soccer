"""Do not award a clean-entry bonus to a body-first receiving window."""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.training.receiving_transition_credit import receiving_transition_window


def clean_receiving_transition_window(
    trace: dict[str, Any],
    *,
    agent_ids: tuple[str, ...],
    agent_id: str,
    start: int,
    frames: int,
    next_target_xy: tuple[float, float],
    required_frames: int = 125,
) -> tuple[np.ndarray, dict[str, Any]]:
    reward, outcome = receiving_transition_window(
        trace,
        agent_ids=agent_ids,
        agent_id=agent_id,
        start=start,
        frames=frames,
        next_target_xy=next_target_xy,
        required_frames=required_frames,
    )
    window = slice(start, start + frames)
    # Existing scoring already validates these measured columns. Inspect the
    # entire admitted window, including contacts before the first foot touch.
    nonfoot = (np.asarray(trace["ball_nonfoot_contact_agent_code"])[window] > 0) & (
        np.asarray(trace["ball_nonfoot_contact_force_n"])[window] > 0
    )
    clean = bool(outcome["controlled_reception"] and not nonfoot.any())
    withheld = 10.0 if outcome["controlled_reception"] and not clean else 0.0
    reward = reward.copy()
    reward[-1] -= withheld
    return reward, {
        **outcome,
        "schema": "soccer.clean_receiving_transition_credit.v1",
        "legacy_success_contract": outcome["success_contract"],
        "clean_controlled_reception": clean,
        "window_nonfoot_contact_frames": int(nonfoot.sum()),
        "withheld_capture_bonus": withheld,
        "shaped_return": float(reward.sum()),
    }
