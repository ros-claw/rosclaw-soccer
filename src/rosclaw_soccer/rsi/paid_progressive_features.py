"""Recompute own learned-parent targets, never relabel old actions as R2 zeros."""

from typing import Any, Literal

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_contact_residual import ContactResidual, DecoupledContactResidual
from rosclaw_soccer.rsi.paid_progressive_contact import ProgressiveContactResidual


def progressive_features(
    base: Any,
    parent: ContactResidual | ProgressiveContactResidual,
    raw: dict[str, np.ndarray],
    scale: torch.Tensor,
    *,
    recorded_policy: Literal["base", "parent"],
    protect: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return own288, correction RELATIVE TO PARENT, parent's R2-relative delta.

    Admission of full physical success and TRAIN provenance belongs to the caller.
    This function proves only causal network/action identity on each supplied row.
    Parent trajectories require actual action == own parent proposal; teacher
    trajectories require the recorded nominal to equal own R2 proposal. Protection
    is legal only if the actual executed action was exactly the parent's action.
    """
    if (
        recorded_policy not in ("base", "parent")
        or type(protect) is not bool
        or type(parent)
        not in (ContactResidual, DecoupledContactResidual, ProgressiveContactResidual)
        or parent.training
        or any(p.requires_grad or p.device.type != "cpu" for p in parent.parameters())
        or scale.shape != (29,)
        or scale.dtype != torch.float32
        or scale.device.type != "cpu"
        or not torch.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError(
            "frozen CPU learned parent, explicit provenance and original scales required"
        )
    n = len(raw["NN_obs"])
    shapes = dict(
        NN_obs=(n, 160),
        NN_action=(n, 29),
        NN_nominal_action=(n, 29),
        NN_h_before=(n, 2, 1, 128),
        NN_c_before=(n, 2, 1, 128),
        NN_h_after=(n, 2, 1, 128),
        NN_c_after=(n, 2, 1, 128),
    )
    if not 1 <= n <= 300 or any(
        k not in raw
        or raw[k].shape != shape
        or raw[k].dtype != np.float32
        or not np.isfinite(raw[k]).all()
        for k, shape in shapes.items()
    ):
        raise ValueError("complete bounded float32 recorded own neural trajectory required")
    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    features, corrections, priors = [], [], []
    with torch.no_grad():
        for i, item in enumerate(raw["NN_obs"]):
            if any(
                not np.array_equal(t.numpy(), raw[k][i])
                for t, k in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
            ):
                raise ValueError("own parent memory-before causal chain differs")
            obs = torch.from_numpy(item.copy()).reshape(1, 1, 160)
            nominal, state, hidden = base.forward_with_features(obs, state)
            if any(
                not np.array_equal(t.numpy(), raw[k][i])
                for t, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
            ):
                raise ValueError("own parent memory-after causal chain differs")
            f = torch.cat([(obs - base.mean) / base.denominator, hidden], dim=-1)[0, 0]
            prior = parent(f.reshape(1, 288))[0]
            parent_action = nominal[0, 0] + prior / scale
            expected = parent_action if recorded_policy == "parent" else nominal[0, 0]
            if not np.array_equal(expected.numpy(), raw["NN_nominal_action"][i]):
                raise ValueError("recorded policy is not own recomputed base/parent proposal")
            actual = torch.from_numpy(raw["NN_action"][i].copy())
            if recorded_policy == "parent" and not torch.equal(actual, parent_action):
                raise ValueError("learned-parent witness contains external action edits")
            if protect and not torch.equal(actual, parent_action):
                raise ValueError("cannot protect an action not actually emitted by parent")
            total = (actual - nominal[0, 0]) * scale
            correction = (actual - parent_action) * scale
            if (
                not torch.isfinite(f).all()
                or not torch.isfinite(correction).all()
                or prior.abs().max() > 0.2
                or total.abs().max() > 0.20001
                or correction.abs().max() > 0.40002
            ):
                raise ValueError("original .2rad final authority or .4rad difference violated")
            features.append(f.numpy().copy())
            corrections.append(correction.numpy().copy())
            priors.append(prior.numpy().copy())
    return np.stack(features), np.stack(corrections), np.stack(priors)
