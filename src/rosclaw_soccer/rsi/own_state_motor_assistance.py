"""Smooth teacher recovery proposals on the learner's ACTUAL observation path.

No state import/reset or physical integration. Assisted rollouts are DAgger
data, never pure-learner PPO or unassisted learner success evidence.
"""

import math
from typing import Any

import numpy as np


def assistance_share(frame: int, length: int, alpha: float) -> float:
    if (
        type(frame) is not int
        or type(length) is not int
        or not 96 <= length <= 300
        or not 0 <= frame < length
        or type(alpha) is not float
        or not math.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("bounded actual reference cursor and preregistered teacher share required")
    if frame < 80 or frame == length - 1:
        return 0.0
    lead = min(1.0, (frame - 80 + 1) / 8.0)
    tail = min(1.0, (length - 1 - frame) / 8.0)
    return alpha * 0.5 * (1 - math.cos(math.pi * lead)) * 0.5 * (1 - math.cos(math.pi * tail))


def blend_actual_motor(
    proposal: np.ndarray,
    oracle: np.ndarray,
    reference_nominal: np.ndarray,
    scale: np.ndarray,
    *,
    share: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        any(
            v.shape != (29,) or v.dtype != np.float32 or not np.isfinite(v).all()
            for v in (proposal, oracle, reference_nominal, scale)
        )
        or not (scale > 0).all()
        or type(share) is not float
        or not math.isfinite(share)
        or not 0 <= share <= 1
    ):
        raise ValueError(
            "finite actual policy29 proposal/oracle/reference in original units required"
        )
    proposed_delta = (proposal - reference_nominal) * scale
    if np.abs(proposed_delta).max() > 0.20002:
        raise ValueError("student proposal outside declared motor authority")
    oracle_delta = (oracle - reference_nominal) * scale
    bounded = np.clip(oracle_delta, -0.2, 0.2)
    bounded_oracle = reference_nominal + bounded / scale
    # Prefix must remain bit-exact to the separately audited failed learner.
    applied = (
        proposal.copy()
        if share == 0
        else (proposal + np.float32(share) * (bounded_oracle - proposal)).astype(np.float32)
    )
    actual_delta = (applied - reference_nominal) * scale
    if np.abs(actual_delta).max() > 0.20002:
        raise ValueError("applied recovery exceeded unchanged mean authority")
    return applied, dict(
        teacher_share=share,
        oracle_authority_clipped=bool(np.any(oracle_delta != bounded)),
        actual_delta_rad_policy=actual_delta.copy(),
        learned_proposal_executed_without_assistance=share == 0,
        PPO_actor_update_allowed=share == 0,
    )
