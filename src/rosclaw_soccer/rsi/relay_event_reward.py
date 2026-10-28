"""Redistribute strict relay terminal outcomes over observed motor controls.

Offline credit assignment only. Events after the sampling horizon are placed
on its last action, never fabricated as observed earlier. Success stays at the
end because a clean full-horizon outcome depends on future contacts.
"""

import numpy as np

from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from rosclaw_soccer.rsi.receiver_strict_terminal_reward import strict_terminal_reward


def strict_relay_event_rewards(
    ball_forces: np.ndarray, body_failed: np.ndarray, *, handoff: bool, controls: int
) -> tuple[np.ndarray, dict]:
    """Keep the strict undiscounted score while locating measured failure events."""
    if (
        not isinstance(ball_forces, np.ndarray)
        or ball_forces.ndim != 4
        or ball_forces.shape[1:] != (10, 2, 3)
        or not isinstance(body_failed, np.ndarray)
        or body_failed.shape != (len(ball_forces), 2)
        or body_failed.dtype != np.bool_
        or not np.isfinite(ball_forces).all()
        or (ball_forces < 0).any()
        or type(handoff) is not bool
        or type(controls) is not int
        or not 1 <= controls <= min(300, len(ball_forces))
    ):
        raise ValueError(
            "finite full relay force and failure history with own control horizon required"
        )
    outcome = strict_pass_receive(ball_forces, body_failed, handoff=handoff)
    rewards = np.zeros(controls, dtype=np.float32)
    placements = {}
    for key in ("passer_nonfoot_substep", "receiver_nonfoot_substep"):
        native = outcome[key]
        if native is not None:
            frame = native // 10
            assigned = min(frame, controls - 1)
            rewards[assigned] -= np.float32(1.0)
            placements[key] = dict(
                observed_control=frame,
                assigned_control=assigned,
                after_sampling_horizon=frame >= controls,
            )
    fail_frames = np.flatnonzero(np.any(body_failed, axis=1))
    if len(fail_frames):
        frame = int(fail_frames[0])
        assigned = min(frame, controls - 1)
        rewards[assigned] -= np.float32(1.0)
        placements["either_body_failure"] = dict(
            observed_control=frame,
            assigned_control=assigned,
            after_sampling_horizon=frame >= controls,
        )
    if outcome["strict_both_clean_handoff"]:
        rewards[-1] += np.float32(2.0)
    score = strict_terminal_reward(outcome)
    if float(rewards.sum(dtype=np.float64)) != score:
        raise ValueError("redistribution must exactly preserve the strict terminal score")
    return rewards, dict(
        outcome=outcome,
        placements=placements,
        terminal_score=score,
        successful_outcome_retained_at_sample_horizon=True,
        physical_success_proven_by_reward=False,
    )
