"""Bounded static reference-sequence gate, never physical execution evidence."""

from typing import Any

import numpy as np


def assess_reference_sequence(
    roots: np.ndarray,
    joints: np.ndarray,
    ankles: np.ndarray,
    support_masks: np.ndarray,
    candidate_valid: np.ndarray,
    trust_boundary_active: np.ndarray,
    nonfoot_floor_distance: np.ndarray,
    *,
    declared_fps: float,
) -> dict[str, Any]:
    """Assess consecutive reference samples without filling censored frames."""
    arrays = (
        roots,
        joints,
        ankles,
        support_masks,
        candidate_valid,
        trust_boundary_active,
        nonfoot_floor_distance,
    )
    if (
        any(type(x) is not np.ndarray for x in arrays)
        or roots.ndim != 2
        or roots.shape[1] != 7
        or not 2 <= len(roots) <= 1000
        or joints.shape != (len(roots), 29)
        or ankles.shape != (len(roots), 2, 3)
        or support_masks.shape != (len(roots), 2)
        or candidate_valid.shape != (len(roots),)
        or trust_boundary_active.shape != (len(roots),)
        or nonfoot_floor_distance.shape != (len(roots),)
        or any(x.dtype != np.bool_ for x in (support_masks, candidate_valid, trust_boundary_active))
        or any(x.dtype.kind != "f" for x in (roots, joints, ankles, nonfoot_floor_distance))
        or type(declared_fps) is not float
        or not np.isfinite(declared_fps)
        or not 1 <= declared_fps <= 240
        or np.any(candidate_valid & ~support_masks.any(axis=1))
    ):
        raise ValueError("bounded sequence, explicit support and validity arrays required")
    for value in (roots, joints, ankles, nonfoot_floor_distance):
        if not np.isfinite(value[candidate_valid]).all():
            raise ValueError("all admitted reference frames must be finite")
        if np.isfinite(value[~candidate_valid]).any():
            raise ValueError("missing or rejected frames must not be padded with numeric values")
    consecutive = candidate_valid[:-1] & candidate_valid[1:]
    z_step = np.abs(np.diff(roots[:, 2]))[consecutive]
    joint_step = np.max(np.abs(np.diff(joints, axis=0)), axis=1)[consecutive]
    shared_support = support_masks[:-1] & support_masks[1:] & consecutive[:, None]
    ankle_step = np.linalg.norm(np.diff(ankles[:, :, :2], axis=0), axis=2)[shared_support]
    max_z = float(z_step.max()) if len(z_step) else None
    max_joint = float(joint_step.max()) if len(joint_step) else None
    max_ankle = float(ankle_step.max()) if len(ankle_step) else None
    all_frames_valid = bool(candidate_valid.all())
    root_step_limit = 0.3 / declared_fps
    joint_step_limit = 3.0 / declared_fps
    ankle_step_limit = 0.3 / declared_fps
    gate = bool(
        all_frames_valid
        and not trust_boundary_active.any()
        and max_z is not None
        and max_joint is not None
        and max_ankle is not None
        and max_z <= root_step_limit
        and max_joint <= joint_step_limit
        and max_ankle <= ankle_step_limit
        and np.min(roots[:, 2]) >= 0.65
        and np.min(nonfoot_floor_distance) >= -0.002
    )
    return dict(
        declared_frames=len(roots),
        valid_frames=int(candidate_valid.sum()),
        missing_or_rejected_frames=int((~candidate_valid).sum()),
        trust_boundary_frames=int(trust_boundary_active.sum()),
        consecutive_valid_pairs=int(consecutive.sum()),
        same_support_ankle_pairs=int(shared_support.sum()),
        maximum_root_z_step_m=max_z,
        maximum_joint_step_rad=max_joint,
        maximum_same_support_ankle_xy_step_m=max_ankle,
        minimum_nonfoot_floor_distance_m=(
            float(nonfoot_floor_distance[candidate_valid].min()) if candidate_valid.any() else None
        ),
        declared_fps=declared_fps,
        root_z_step_limit_m=root_step_limit,
        joint_step_limit_rad=joint_step_limit,
        ankle_xy_step_limit_m=ankle_step_limit,
        nonfoot_penetration_limit_m=0.002,
        reference_sequence_gate_passed=gate,
        original_labels_not_changed=True,
        source_contact_truth_not_proven=True,
        kinematic_gate_is_not_dynamics_or_motor_qualification=True,
        physical_execution_qualified=False,
        hardware_authorized=False,
    )
