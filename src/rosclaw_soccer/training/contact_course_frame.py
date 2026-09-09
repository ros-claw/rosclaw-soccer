"""Validate a G1 contact curriculum's physical frame before simulator creation.

An inference-only canonical copy is not a physical curriculum transform. If an
episode is translated, body, ball, physical goal and reward target must move
together. This numerical check proves neither dynamics parity nor promotion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class ContactCourseFrameBinding:
    translation_xy_m: tuple[float, float]
    binding_hash: str
    activation_ceiling: str = "SIM_ONLY"


def bind_contact_course_frame(
    *,
    source_qpos: np.ndarray,
    training_qpos: np.ndarray,
    source_goal_center_m: tuple[float, float, float],
    training_goal_center_m: tuple[float, float, float],
    source_reward_target_m: tuple[float, float, float],
    training_reward_target_m: tuple[float, float, float],
) -> ContactCourseFrameBinding:
    """Bind equal planar translations; reject mismatched or partial transforms.

    Inputs are measured/declared initial poses, not writable simulator handles.
    G1 qpos has 36 body and 7 ball coordinates. Rotations, joint angles and Z
    must remain unchanged; rotations and morphology transfer are out of scope.
    """
    states = (source_qpos, training_qpos)
    for state in states:
        if (
            not isinstance(state, np.ndarray)
            or state.shape != (43,)
            or state.dtype.kind not in "fiu"
            or not np.isfinite(state).all()
        ):
            raise ValueError("finite real G1 body/ball initial poses required")
        if any(
            abs(float(np.linalg.norm(state[s])) - 1) > 1e-4 for s in (slice(3, 7), slice(39, 43))
        ):
            raise ValueError("unit body and ball orientations required")
    points = (
        source_goal_center_m,
        training_goal_center_m,
        source_reward_target_m,
        training_reward_target_m,
    )
    for point in points:
        if (
            type(point) is not tuple
            or len(point) != 3
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in point)
        ):
            raise ValueError("finite immutable goal and reward coordinates required")
    source, training = (s.astype(np.float64) for s in states)
    delta = training[:2] - source[:2]
    unchanged = np.ones(43, dtype=bool)
    unchanged[[0, 1, 36, 37]] = False
    if not np.allclose(training[unchanged], source[unchanged], rtol=0, atol=1e-10):
        raise ValueError("contact frame transform must preserve height, joints and orientation")
    expected = np.array([*delta, 0.0])
    translations = (
        training[36:39] - source[36:39],
        np.asarray(training_goal_center_m) - np.asarray(source_goal_center_m),
        np.asarray(training_reward_target_m) - np.asarray(source_reward_target_m),
    )
    if not np.isfinite(expected).all() or any(
        not np.allclose(value, expected, rtol=0, atol=1e-8) for value in translations
    ):
        raise ValueError("body, ball, physical goal and reward target frames differ")
    binding = hash_json(
        {
            "schema": "rosclaw_soccer.g1_contact_course_frame.v1",
            "source_qpos": source.tolist(),
            "training_qpos": training.tolist(),
            "source_goal_center_m": source_goal_center_m,
            "training_goal_center_m": training_goal_center_m,
            "source_reward_target_m": source_reward_target_m,
            "training_reward_target_m": training_reward_target_m,
            "translation_xy_m": delta.tolist(),
            "activation_ceiling": "SIM_ONLY",
        }
    )
    return ContactCourseFrameBinding((float(delta[0]), float(delta[1])), str(binding))
