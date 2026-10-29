"""Source-codec-derived, differentiable SIM kick target envelope."""

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


def kick_target_bounds(config):
    """Compile source FreeKick action bounds into canonical HG29 target order."""
    default = np.asarray(config["default_joint_pos"], dtype=np.float32)
    scale = np.asarray(config["action_scale"], dtype=np.float32)
    fallback = float(config["clip_actions"])
    if (
        default.shape != (29,)
        or scale.shape != (29,)
        or not np.isfinite(default).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or not np.isfinite(fallback)
        or fallback <= 0
        or not isinstance(config.get("action_clip"), dict)
    ):
        raise ValueError("finite positive source codec required")
    lo = np.full(29, -fallback, dtype=np.float32)
    hi = np.full(29, fallback, dtype=np.float32)
    matched = set()
    for pattern, bounds in config["action_clip"].items():
        if (
            not isinstance(pattern, str)
            or not isinstance(bounds, (tuple, list))
            or len(bounds) != 2
        ):
            raise ValueError("source action bounds required")
        lower, upper = map(float, bounds)
        if not np.isfinite(lower + upper) or lower >= upper:
            raise ValueError("finite increasing action bounds required")
        hits = [
            i
            for i, name in enumerate(G1_DDS_JOINT_NAMES)
            if pattern == name or (pattern.startswith(".*_") and name.endswith(pattern[3:]))
        ]
        if not hits or any(i in matched for i in hits):
            raise ValueError("unique action pattern required")
        for i in hits:
            lo[i], hi[i] = lower, upper
            matched.add(i)
    return default + scale * lo, default + scale * hi


def decode_bounded(logits, lower, upper):
    """Differentiably map unconstrained logits to fixed joint target bounds."""
    import torch

    return (lower + upper) * 0.5 + (upper - lower) * 0.5 * torch.tanh(logits)
