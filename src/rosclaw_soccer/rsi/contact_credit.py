"""Retrospective contact-time credit ablation; original task scoring is untouched.

Only the temporal placement of measured football reward changes. Undiscounted
episode return is preserved, not invented. This is NOT new physical experience
or a claim that off-contact delayed rewards are intrinsically incorrect.
"""

import math
from typing import Any

import numpy as np


def contact_return_credit(
    rewards: np.ndarray,
    terms: list[dict[str, float]],
    *,
    first_contact_frame: int | None,
    qualified: bool,
    causal: bool,
) -> np.ndarray:
    if (
        type(rewards) is not np.ndarray
        or rewards.shape != (300,)
        or rewards.dtype != np.float32
        or not np.isfinite(rewards).all()
        or len(terms) != 280
        or type(qualified) is not bool
        or type(causal) is not bool
        or (
            first_contact_frame is not None
            and (type(first_contact_frame) is not int or not 20 <= first_contact_frame < 300)
        )
    ):
        raise ValueError("complete finite native reward and owned post-prefix contact required")
    keys = {"time", "approach", "contact", "momentum", "forward", "lateral", "nonfoot", "balance"}
    for row in terms:
        if set(row) != keys or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in row.values()
        ):
            raise ValueError("complete finite original reward terms required")
    unsafe = any(t["nonfoot"] < 0 or t["balance"] < 0 for t in terms)
    expected = np.zeros(300, dtype=np.float64)
    expected[20:] = [sum(t.values()) for t in terms]
    expected[-1] += 5.0 if qualified and causal else -3.0 if unsafe else 0.0
    if (qualified and unsafe) or not np.allclose(rewards, expected, atol=1e-6, rtol=1e-6):
        raise ValueError("original measured terms/terminal reward commitment differs")
    football = np.zeros(300, dtype=np.float64)
    football[20:] = [
        sum(t[k] for k in ("contact", "momentum", "forward", "lateral")) for t in terms
    ]
    football[-1] += 5.0 if qualified and causal else 0.0
    if first_contact_frame is None:
        if np.any(football != 0):
            raise ValueError("football reward without measured foot contact")
        return rewards.copy()
    if np.any(football[:first_contact_frame] != 0):
        raise ValueError("football reward before measured foot contact")
    redistributed = rewards.astype(np.float64) - football
    redistributed[first_contact_frame] += football.sum()
    result = redistributed.astype(np.float32)
    if (
        not np.isfinite(result).all()
        or not np.array_equal(result[:20], rewards[:20])
        or abs(float(result.sum(dtype=np.float64) - rewards.sum(dtype=np.float64))) > 2e-5
    ):
        raise ValueError("contact credit must preserve total measured return and prefix")
    return result


def credit_audit(before: np.ndarray, after: np.ndarray, frame: int | None) -> dict[str, Any]:
    return dict(
        first_contact_frame=frame,
        original_return=float(before.sum(dtype=np.float64)),
        redistributed_return=float(after.sum(dtype=np.float64)),
        changed_frames=int(np.count_nonzero(before != after)),
        physical_experience_added=0,
        future_information_added_to_actor=False,
        original_exam_changed=False,
    )
