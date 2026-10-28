"""Time-aligned physical effect before a receiver's first real ball contact.

Comparisons are paired by initial condition, not fixed-state interventions:
the two closed-loop trajectories may already have diverged.
"""

import numpy as np


def precontact_tracking_error(
    targets: np.ndarray,
    qpos: np.ndarray,
    forces: np.ndarray,
    *,
    joint_indices: np.ndarray,
    preceding_controls: int = 5,
) -> dict:
    """Measure target/actual error at completed controls strictly before contact.

    This is not a torque-saturation estimate, a prediction, or a real-time
    feature: first contact is only known retrospectively.
    """
    if (
        not isinstance(targets, np.ndarray)
        or targets.ndim != 2
        or targets.shape[1] != 29
        or not isinstance(qpos, np.ndarray)
        or qpos.ndim != 3
        or qpos.shape[1] != 10
        or not isinstance(forces, np.ndarray)
        or forces.shape != (len(qpos), 10, 2, 3)
        or len(targets) != len(qpos)
        or len(qpos) < 1
        or any(not np.isfinite(x).all() for x in (targets, qpos, forces))
        or (forces < 0).any()
        or not isinstance(joint_indices, np.ndarray)
        or joint_indices.shape != (29,)
        or not np.issubdtype(joint_indices.dtype, np.integer)
        or np.any(joint_indices < 0)
        or np.any(joint_indices >= qpos.shape[2])
        or type(preceding_controls) is not int
        or not 1 <= preceding_controls <= 20
    ):
        raise ValueError("finite aligned native target, state and contact histories required")
    receiver = forces[:, :, 1, :].reshape(-1, 3)
    contact = np.flatnonzero((np.max(receiver[:, :2], axis=1) > 10) | (receiver[:, 2] > 1))
    if not len(contact):
        return dict(
            first_contact_control=None, controls_observed=0, max_leg_target_tracking_error_rad=None
        )
    end = int(contact[0]) // 10
    begin = max(0, end - preceding_controls)
    if begin == end:
        return dict(
            first_contact_control=end, controls_observed=0, max_leg_target_tracking_error_rad=None
        )
    error = targets[begin:end, :12] - qpos[begin:end, -1, joint_indices][:, :12]
    return dict(
        first_contact_control=end,
        controls_observed=end - begin,
        max_leg_target_tracking_error_rad=float(np.max(np.abs(error))),
    )


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
