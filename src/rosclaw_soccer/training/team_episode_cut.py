"""Sampled match-boundary and team-safety termination for SIM training only.

This helper does not certify safety, attribute blame, or authorize a policy.
Its 50 Hz inputs may miss failures and contacts between observations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.world.match_boundary import uninterrupted_in_play


@dataclass(frozen=True)
class TeamEpisodeCut:
    """First sampled terminal frame and any later failure within the episode."""

    first_safety_frame: int | None
    first_exit_frame: int | None
    first_terminal_frame: int | None
    sampled_only: bool = True
    promotion_eligible: bool = False


def sampled_team_episode_cut(
    trace: Mapping[str, Any], *, agent_ids: tuple[str, ...]
) -> TeamEpisodeCut:
    """Find first sampled team-body/collision failure or whole-ball exit.

    All declared agents are checked. Missing, malformed, nonfinite, or
    non-normalized evidence raises rather than creating a clean episode.
    Thresholds match the existing receiving-window sampled body guard.
    """
    if (
        not isinstance(trace, Mapping)
        or not agent_ids
        or tuple(sorted(set(agent_ids))) != agent_ids
    ):
        raise ValueError("explicit sorted unique team roster required")
    boundary = uninterrupted_in_play(dict(trace))
    time = np.asarray(trace["time"])
    n = len(time)

    def field(name: str, shape: tuple[int, ...]) -> NDArray[Any]:
        a = np.asarray(trace[name])
        if a.shape != shape or a.dtype.kind not in "biuf" or not np.isfinite(a).all():
            raise ValueError(f"finite aligned {name} evidence required")
        return a

    collision = field("robot_robot_contact_count", (n,))
    if (collision < 0).any():
        raise ValueError("nonnegative collision count required")
    unsafe = collision > 0
    for agent in agent_ids:
        if not isinstance(agent, str) or not agent:
            raise ValueError("named team agents required")
        prefix = agent.replace(".", "_")
        pose = field(prefix + "_pelvis_pose", (n, 7))
        margin = field(prefix + "_joint_safety_margin_rad", (n, 29))
        if not np.allclose(np.linalg.norm(pose[:, 3:7], axis=1), 1, atol=1e-4, rtol=0):
            raise ValueError("normalized measured team body quaternions required")
        upright = 1 - 2 * (pose[:, 4] ** 2 + pose[:, 5] ** 2)
        unsafe |= (pose[:, 2] < 0.55) | (upright < np.cos(0.8)) | (margin.min(axis=1) < 0)
    indices = np.flatnonzero(unsafe)
    safety = int(indices[0]) if len(indices) else None
    exit_frame = boundary["first_exit_frame"]
    terminal = min((x for x in (safety, exit_frame) if x is not None), default=None)
    return TeamEpisodeCut(safety, exit_frame, terminal)
