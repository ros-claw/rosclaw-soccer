"""Common-prefix safety and causal first-touch attribution; not an exam change."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.bilateral_entry import observed_contact_foot


def diagnose_entry_prefix(model: Any, prefix: dict[str, np.ndarray]) -> dict[str, Any]:
    f = prefix["ball_contact_force_by_body_n"]
    q = prefix["target_body_canonical_qpos_after_substep"]
    if (
        f.shape != (20, 10, model.nbody)
        or q.shape != (20, 10, 43)
        or not np.isfinite(f).all()
        or not np.isfinite(q).all()
        or np.any(f < 0)
    ):
        raise ValueError("complete finite native20 prefix required")
    root = model.body_rootid[model.body("pelvis").id]
    feet = [
        model.body(side + suffix).id
        for side in ("left", "right")
        for suffix in ("_ankle_roll_link", "_ankle_pitch_link")
    ]
    nonfeet = np.flatnonzero((model.body_rootid == root) & ~np.isin(np.arange(model.nbody), feet))
    if root == 0 or not len(nonfeet):
        raise ValueError("full robot nonfoot body classification required")
    peak = float(f[:, :, nonfeet].max())
    height = float(q[:, :, 2].min())
    upright = float((1 - 2 * (q[:, :, 4] ** 2 + q[:, :, 5] ** 2)).min())
    foot = observed_contact_foot(model, prefix, global_start_frame=0)
    safe = bool(peak <= 1 and height >= 0.65 and upright >= 0.9)
    return dict(
        prefix_physical_safe=safe,
        prefix_maximum_nonfoot_force_n=peak,
        prefix_minimum_pelvis_z_m=height,
        prefix_minimum_upright=upright,
        prefix_foot_contact=foot,
        post_boundary_control_can_cause_first_touch=foot["first_contact_native_offset"] is None,
        hardware_authorized=False,
        promotion_authorized=False,
    )
