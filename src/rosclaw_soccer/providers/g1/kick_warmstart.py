"""SIM-only observation continuity at a mid-motion neural policy handoff."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def prepare_kick_handoff(policy: Any, *, entry_frame: int) -> None:
    """Rebase the reference, not physics; fill history from current measured state."""
    if policy.use_body_frame_ball or policy.runtime_mode != "sim":
        raise ValueError("kick handoff is simulation-only")
    module = importlib.import_module(type(policy).__module__)
    anchor_index = int(module.NPZ_ANCHOR_IDX)
    mapping = np.asarray(module.ISAAC_TO_MUJOCO)
    if mapping.shape != (29,) or sorted(mapping.tolist()) != list(range(29)):
        raise ValueError("kick joint mapping changed")
    if not 0 <= entry_frame < len(policy.motion_body_pos):
        raise ValueError("kick handoff frame outside reference")
    reference = np.asarray(policy.motion_body_pos[entry_frame, anchor_index], dtype=float)
    scale = np.asarray(policy.action_scale_mj, dtype=float)
    q = np.asarray(policy.state_cmd.q, dtype=float)
    if (
        reference.shape != (3,)
        or scale.shape != (29,)
        or q.shape != (29,)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(scale))
        or not np.all(np.isfinite(q))
        or np.any(np.abs(scale) < 1e-9)
    ):
        raise ValueError("invalid kick handoff reference or measured joints")
    policy._ref_anchor_world_origin = policy._init_to_world @ reference
    history_action = ((q - policy.default_q_mj) / scale)[mapping]
    policy.last_action_il = np.clip(
        history_action, policy.action_clip_lo_il, policy.action_clip_hi_il
    ).astype(np.float32)
    for _ in range(5):
        obs = policy._build_obs()
        if obs.shape != (547,) or not np.all(np.isfinite(obs)):
            raise ValueError("kick warmstart observation contract changed")
