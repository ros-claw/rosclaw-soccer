"""Compare paired first episodes without counting simulator auto-resets."""

import numpy as np

_INITIAL = (
    "initial_actor_obs",
    "initial_root_pose",
    "initial_root_velocity",
    "initial_joint_pos",
    "initial_joint_vel",
    "initial_motion_id",
    "initial_motion_step",
)
_ERRORS = ("error_body_pos", "error_joint_pos")


def compare_first_episodes(parent, candidate):
    """Require exact measured starts and compare only jointly valid post-steps."""
    for key in _INITIAL:
        a, b = np.asarray(parent[key]), np.asarray(candidate[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.array_equal(a, b):
            raise ValueError(f"unmatched initial physical or reference field: {key}")
    valid_masks, done_masks = [], []
    for trace in (parent, candidate):
        valid = np.asarray(trace["valid_before"])
        done = np.asarray(trace["done"])
        if (
            valid.ndim != 2
            or valid.dtype != np.bool_
            or done.shape != valid.shape
            or not np.isin(done, (0, 1)).all()
        ):
            raise ValueError("aligned first-episode masks required")
        previous = np.ones(valid.shape[1], dtype=bool)
        for index in range(len(valid)):
            if not np.array_equal(valid[index], previous):
                raise ValueError("first-episode mask contradicts done flags")
            previous &= ~done[index].astype(bool)
        valid_masks.append(valid)
        done_masks.append(done.astype(bool))
    if valid_masks[0].shape != valid_masks[1].shape or valid_masks[0].shape[1] != len(
        parent["initial_motion_id"]
    ):
        raise ValueError("matched evaluation dimensions required")
    common = valid_masks[0] & valid_masks[1] & ~done_masks[0] & ~done_masks[1]
    rows = []
    for world in range(common.shape[1]):
        indices = common[:, world]
        row = dict(
            world=world,
            motion_id=int(parent["initial_motion_id"][world]),
            initial_length=int(valid_masks[0][:, world].sum()),
            candidate_length=int(valid_masks[1][:, world].sum()),
            initial_censored=bool(not np.any(valid_masks[0][:, world] & done_masks[0][:, world])),
            candidate_censored=bool(not np.any(valid_masks[1][:, world] & done_masks[1][:, world])),
            common_valid_post_steps=int(indices.sum()),
        )
        for key in _ERRORS:
            a, b = np.asarray(parent[key]), np.asarray(candidate[key])
            if (
                a.shape != common.shape
                or b.shape != common.shape
                or not np.isfinite(a[indices, world]).all()
                or not np.isfinite(b[indices, world]).all()
            ):
                raise ValueError(f"finite aligned metric required: {key}")
            row[key] = (
                None
                if not indices.any()
                else dict(
                    initial_mean=float(a[indices, world].mean()),
                    candidate_mean=float(b[indices, world].mean()),
                )
            )
        rows.append(row)
    return dict(
        worlds=len(rows),
        rows=rows,
        development_training_motions=True,
        soccer_verified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
