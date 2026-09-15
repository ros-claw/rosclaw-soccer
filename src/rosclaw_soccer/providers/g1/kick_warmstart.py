"""SIM-only observation continuity at a mid-motion neural policy handoff."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def rebase_kick_reference(policy: Any, *, entry_frame: int) -> None:
    """Align only the reference origin to its admitted frame, never fill history.

    The existing world entry call binds the measured torso origin and yaw.
    This opt-in experiment removes skipped reference displacement, not actual
    robot displacement. It does not invoke inference or synthesize observations.
    """
    if policy.use_body_frame_ball or policy.runtime_mode != "sim":
        raise ValueError("kick reference rebasing is simulation-only")
    module = importlib.import_module(type(policy).__module__)
    anchor_index = int(module.NPZ_ANCHOR_IDX)
    motion = np.asarray(policy.motion_body_pos)
    rotation = np.asarray(policy._init_to_world, dtype=float)
    if (
        type(entry_frame) is not int
        or motion.ndim != 3
        or motion.shape[2] != 3
        or not 0 <= anchor_index < motion.shape[1]
        or not 0 <= entry_frame < len(motion)
        or rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("finite admitted frame and proper reference rotation required")
    reference = np.asarray(motion[entry_frame, anchor_index], dtype=float)
    if not np.all(np.isfinite(reference)):
        raise ValueError("finite admitted reference position required")
    policy._ref_anchor_world_origin = rotation @ reference


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
