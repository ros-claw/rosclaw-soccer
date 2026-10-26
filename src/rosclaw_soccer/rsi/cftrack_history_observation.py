"""Causal CFtrack proprioception assembly; never install or execute a policy."""

import importlib
from typing import Any

import numpy as np

HISTORY_SLICES = (
    (0, 15),
    (15, 18),
    (18, 21),
    (21, 50),
    (50, 79),
    (79, 108),
    (108, 111),
    (111, 117),
    (117, 141),
)


def assemble_proprioception(
    ee_position: np.ndarray,
    angular_velocity: np.ndarray,
    gravity: np.ndarray,
    joint_position_hg: np.ndarray,
    joint_velocity_hg: np.ndarray,
    previous_action_lab: np.ndarray,
    object_position: np.ndarray,
    object_rotation_6d: np.ndarray,
    bbox: np.ndarray,
    *,
    mj2lab: np.ndarray,
    default_lab: np.ndarray,
) -> np.ndarray:
    """Assemble source-order features; inputs must describe the same measured boundary.

    The caller must retain the provenance of previous_action_lab: a conversion
    of another policy's executed target is not that policy's original action.
    """
    parts = (
        (ee_position, 15),
        (angular_velocity, 3),
        (gravity, 3),
        (joint_position_hg, 29),
        (joint_velocity_hg, 29),
        (previous_action_lab, 29),
        (object_position, 3),
        (object_rotation_6d, 6),
        (bbox, 24),
        (default_lab, 29),
    )
    if (
        any(
            type(value) is not np.ndarray
            or value.shape != (length,)
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
            for value, length in parts
        )
        or type(mj2lab) is not np.ndarray
        or mj2lab.shape != (29,)
        or mj2lab.dtype.kind not in "iu"
        or sorted(mj2lab.tolist()) != list(range(29))
    ):
        raise ValueError("finite measured feature vectors and HG-to-lab permutation required")
    observation = np.concatenate(
        [
            ee_position.astype(np.float32),
            angular_velocity.astype(np.float32),
            gravity.astype(np.float32),
            (joint_position_hg[mj2lab] - default_lab).astype(np.float32),
            joint_velocity_hg[mj2lab].astype(np.float32),
            previous_action_lab.astype(np.float32),
            object_position.astype(np.float32),
            object_rotation_6d.astype(np.float32),
            bbox.astype(np.float32),
        ]
    )
    if observation.shape != (141,) or not np.isfinite(observation).all():
        raise ValueError("finite float32 141-vector required")
    return observation


def flatten_five_frame_history(prior_four: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Flatten measured oldest-to-newest history in the source's feature-major order."""
    if (
        type(prior_four) is not np.ndarray
        or prior_four.shape != (4, 141)
        or prior_four.dtype != np.float32
        or not np.isfinite(prior_four).all()
        or type(current) is not np.ndarray
        or current.shape != (141,)
        or current.dtype != np.float32
        or not np.isfinite(current).all()
    ):
        raise ValueError("four prior and one current finite float32 observations required")
    history = np.concatenate((prior_four, current[None, :]), axis=0)
    return np.concatenate([history[:, a:b].reshape(-1) for a, b in HISTORY_SLICES])


def assemble_current_ball_observation(policy: Any) -> np.ndarray:
    """Read a scratch policy's already measured and FK-updated ball features."""
    if policy.task != "kickball" or policy.active_object_name != "ball":
        raise ValueError("qualified scratch kickball policy required")
    module = importlib.import_module(type(policy).__module__)
    heading = module.yaw_quat(policy.torso_quat)
    object_position, _ = module.subtract_frame_transforms(
        policy.torso_pos, heading, policy.state_cmd.obj_pos, policy.state_cmd.obj_quat
    )
    object_position = policy._clip_norm(object_position)
    object_rotation = module.matrix_from_quat(np.array([1, 0, 0, 0], dtype=np.float32))[
        :, :2
    ].reshape(-1)
    return assemble_proprioception(
        policy.ee_pos,
        policy.state_cmd.ang_vel,
        policy.state_cmd.gravity_ori,
        policy.state_cmd.q,
        policy.state_cmd.dq,
        policy.action,
        object_position,
        object_rotation,
        policy._build_bbox_rel(heading),
        mj2lab=policy.mj2lab,
        default_lab=policy.default_angles_lab,
    )


def audit_current_ball_observation(policy: Any, full_observation: np.ndarray) -> None:
    """Compare source inference input with independently assembled current features.

    Called at its inference boundary after source history was rolled. It reads
    the current policy and does not install a history or invoke the network.
    The source FK and quaternion helpers are still shared dependencies.
    """
    if (
        type(full_observation) is not np.ndarray
        or full_observation.shape != (1, 1244)
        or full_observation.dtype != np.float32
        or not np.isfinite(full_observation).all()
        or policy.obs_history_buffer.shape != (5, 141)
    ):
        raise ValueError("qualified source kickball inference boundary required")
    current = assemble_current_ball_observation(policy)
    history = policy.obs_history_buffer
    if not np.array_equal(current, history[-1]) or not np.array_equal(
        flatten_five_frame_history(history[:4], current), full_observation[0, -705:]
    ):
        raise ValueError("measured current proprioception or source history layout mismatch")
