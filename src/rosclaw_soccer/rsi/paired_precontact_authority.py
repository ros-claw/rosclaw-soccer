"""Time-aligned physical effect before a receiver's first real ball contact.

Comparisons are paired by initial condition, not fixed-state interventions:
the two closed-loop trajectories may already have diverged.
"""

import numpy as np


def paired_precontact_authority(
    candidate_command: np.ndarray,
    baseline_command: np.ndarray,
    candidate_qpos: np.ndarray,
    baseline_qpos: np.ndarray,
    candidate_forces: np.ndarray,
    *,
    joint_indices: np.ndarray,
    root_index: int,
) -> dict:
    """Measure the prior completed control, never a mismatched terminal state."""
    arrays = (candidate_command, baseline_command, candidate_qpos, baseline_qpos, candidate_forces)
    if (
        any(not isinstance(x, np.ndarray) or not np.isfinite(x).all() for x in arrays)
        or candidate_command.ndim != 2
        or baseline_command.ndim != 2
        or candidate_command.shape[1] != 3
        or baseline_command.shape[1] != 3
        or candidate_qpos.ndim != 3
        or baseline_qpos.ndim != 3
        or candidate_qpos.shape[1:] != baseline_qpos.shape[1:]
        or candidate_qpos.shape[1] != 10
        or candidate_forces.shape != (len(candidate_qpos), 10, 2, 3)
        or (candidate_forces < 0).any()
        or type(root_index) is not int
        or not 0 <= root_index < candidate_qpos.shape[2] - 1
        or not isinstance(joint_indices, np.ndarray)
        or joint_indices.ndim != 1
        or not np.issubdtype(joint_indices.dtype, np.integer)
        or len(joint_indices) == 0
        or np.any(joint_indices < 0)
        or np.any(joint_indices >= candidate_qpos.shape[2])
    ):
        raise ValueError("aligned, finite native paired trajectories required")
    n = min(len(candidate_command), len(baseline_command))
    changed = np.flatnonzero(
        np.any(np.abs(candidate_command[:n] - baseline_command[:n]) > 1e-6, axis=1)
    )
    # Use the same force thresholds as the strict pass/receive contract.
    receiver = candidate_forces[:, :, 1, :].reshape(-1, 3)
    touched = np.flatnonzero((np.max(receiver[:, :2], axis=1) > 10) | (receiver[:, 2] > 1))
    first = int(touched[0]) if len(touched) else None
    prior = None if first is None else first // 10 - 1
    aligned = prior is not None and 0 <= prior < min(len(candidate_qpos), len(baseline_qpos))
    result = dict(
        first_command_change_control=int(changed[0]) if len(changed) else None,
        first_receiver_contact_substep=first,
        prior_control=prior if aligned else None,
        prior_control_time_aligned=bool(aligned),
        precontact_max_joint_delta_rad=None,
        precontact_root_xy_delta_m=None,
        fixed_state_causal_effect=False,
    )
    if aligned:
        assert prior is not None
        a, b = candidate_qpos[prior, -1], baseline_qpos[prior, -1]
        result["precontact_max_joint_delta_rad"] = float(
            np.max(np.abs(a[joint_indices] - b[joint_indices]))
        )
        result["precontact_root_xy_delta_m"] = float(
            np.linalg.norm(a[root_index : root_index + 2] - b[root_index : root_index + 2])
        )
    return result
