"""Measure cross-policy action-coordinate compatibility, without installing history."""

from typing import Any

import numpy as np


def describe_previous_target_encoding(
    target_hg: np.ndarray,
    default_lab: np.ndarray,
    scale_lab: np.ndarray,
    mj2lab: np.ndarray,
    lab2mj: np.ndarray,
) -> dict[str, Any]:
    """Invert the target codec; this is not the old policy's actual action.

    Target is the previously *executed* PD target, not the last proposal. The
    returned reconstructed target is diagnostic only; no clipping, history
    installation, physical action, or exact-equivalence claim is performed.
    """
    vectors = (target_hg, default_lab, scale_lab)
    permutations = (mj2lab, lab2mj)
    if (
        any(
            type(x) is not np.ndarray
            or x.shape != (29,)
            or x.dtype.kind != "f"
            or not np.isfinite(x).all()
            for x in vectors
        )
        or any(
            type(x) is not np.ndarray
            or x.shape != (29,)
            or x.dtype.kind not in "iu"
            or sorted(x.tolist()) != list(range(29))
            for x in permutations
        )
        or not np.array_equal(mj2lab[lab2mj], np.arange(29))
        or np.any(scale_lab <= 0)
    ):
        raise ValueError(
            "finite target, positive scales and inverse 29-joint permutations required"
        )
    encoded64 = (target_hg[mj2lab] - default_lab) / scale_lab
    encoded = encoded64.astype(np.float32)
    if not np.isfinite(encoded).all():
        raise ValueError("finite representable float32 action required")
    # Reproduce the source codec's float32 arithmetic and HG permutation.
    reconstructed = (encoded * scale_lab.astype(np.float32) + default_lab.astype(np.float32))[
        lab2mj
    ]
    return dict(
        implied_action_lab=encoded,
        reconstructed_target_hg=reconstructed,
        maximum_target_roundtrip_error_rad=float(np.max(np.abs(reconstructed - target_hg))),
        maximum_absolute_implied_action=float(np.max(np.abs(encoded))),
        prior_policy_action_proven=False,
        historical_observation_installed=False,
        clipping_applied=False,
        physical_state_modified=False,
        hardware_authorized=False,
    )
