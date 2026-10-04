"""Source-bound preparation of complete successful-teacher distillation data.

The caller must independently authenticate all physical episodes and teacher
eligibility. This adapter checks original behavior, ordered frames and actual
conditional densities. It runs no physics and issues no authority.
"""

import re
from typing import Any

import numpy as np
from rosclaw.growth.correlated_residual_gradient import conditional_means

from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
from rosclaw_soccer.rsi.proposal_memory_motor import make_preview, validate_model


def prepare_imitation_batch(
    parent: dict[str, Any],
    arrays: Any,
    *,
    selected_teacher_groups: list[int],
    behavior_model_hash: str,
    teacher_selection_hash: str,
) -> dict[str, Any]:
    validate_model(parent)
    if behavior_model_hash != parent["model_hash"]:
        raise ValueError("exact actual immediate source behavior required")
    if (
        type(teacher_selection_hash) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", teacher_selection_hash) is None
    ):
        raise ValueError("externally authenticated teacher selection commitment required")
    values = [
        np.asarray(arrays[k])
        for k in (
            "observation",
            "phase_index",
            "latent_action",
            "old_log_probability",
            "terminal_return",
            "std_raw",
            "trajectory_index",
        )
    ]
    x, phase, action, old_logp, returns, std, groups = values
    n = len(x)
    if (
        x.shape != (n, 134)
        or not 1080 <= n <= 200000
        or n % 270
        or action.shape != (n, 12)
        or any(v.shape != (n,) for v in (phase, old_logp, returns, std, groups))
        or phase.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or set(phase.tolist()) != {0, 1, 2}
        or any(
            v.dtype.kind not in "fiu" or not np.isfinite(v).all()
            for v in (x, action, old_logp, returns, std)
        )
        or not np.array_equal(groups, np.repeat(np.arange(n // 270), 270))
        or not np.all(std == 0.1)
        or type(selected_teacher_groups) is not list
        or not 4 <= len(selected_teacher_groups) <= n // 270
        or any(type(v) is not int or not 0 <= v < n // 270 for v in selected_teacher_groups)
        or selected_teacher_groups != sorted(set(selected_teacher_groups))
    ):
        raise ValueError(
            "all original ordered source frames and distinct complete teacher episodes required"
        )
    if any(not np.all(returns[groups == g] == returns[groups == g][0]) for g in range(n // 270)):
        raise ValueError("one original terminal return per complete source episode required")
    decoder = select_proposal_decoder(make_preview(parent), implementation="bounded_snapshot")
    features = np.stack([decoder.features(v) for v in x])
    context = np.column_stack((features[:, :134], phase))
    current = np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    baseline = np.stack(
        [decoder._parent.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)]
    )
    first = np.arange(n) % 270 == 0
    scale = std * np.where(first, 1.0, np.sqrt(1 - 0.9**2))
    mean = conditional_means(current, action, first, 0.9)
    density = np.sum(
        -0.5 * ((action - mean) / scale[:, None]) ** 2
        - np.log(scale[:, None])
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(density, old_logp, atol=1e-8, rtol=0):
        raise ValueError("exact original actually sampled conditional behavior density required")
    return dict(
        context=context,
        baseline=baseline,
        gates=decoder._guard.gates(context),
        targets=action.copy(),
        training_weights=np.isin(groups, selected_teacher_groups).astype(np.float64),
    )
