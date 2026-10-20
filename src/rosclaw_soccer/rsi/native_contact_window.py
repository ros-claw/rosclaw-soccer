"""Read-only outcome window for the new native reference-motion pilot.

This variable-horizon, 10 N profile is NOT the legacy H300/1 N first-touch exam.
Missing 1.2 s ball outcomes remain censored; no synthetic tail or task pass.
"""

import numpy as np


def native_contact_window(
    *,
    force: np.ndarray,
    canonical_qpos: np.ndarray,
    canonical_qvel: np.ndarray,
    goal: np.ndarray,
    intended_foot: int,
) -> dict[str, object]:
    """Aligned 2 ms post-step samples, canonical robot-first q43/v41."""
    if (
        type(intended_foot) is not int
        or intended_foot not in (0, 1)
        or not isinstance(force, np.ndarray)
        or force.ndim != 2
        or force.shape[1] != 2
        or len(force) == 0
    ):
        raise ValueError("aligned actual two-foot samples and explicit kicking foot required")
    for value, shape in (
        (force, (len(force), 2)),
        (canonical_qpos, (len(force), 43)),
        (canonical_qvel, (len(force), 41)),
        (goal, (3,)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite aligned actual native q43/v41/goal arrays required")
    if (force < 0).any():
        raise ValueError("nonnegative actual contact force required")
    indices = np.flatnonzero((force > 10).any(axis=1))
    result: dict[str, object] = dict(
        profile="native_reference_motion.contact10N.post1p2s.v1",
        timestep_s=0.002,
        required_post_contact_substeps=600,
        available_native_substeps=len(force),
        first_foot_contact_substep=None,
        correct_foot_first=False,
        post_contact_status="NO_FOOT_CONTACT",
        post_contact_quality_observed=False,
        goal_progress_1p2s_m=None,
        goal_lateral_1p2s_m=None,
        other_foot_contact_in_window=None,
        maximum_pelvis_retreat_after_contact_m=None,
        minimum_post_contact_upright=None,
        minimum_post_contact_pelvis_height_m=None,
        formal_exam_unchanged=True,
        causal_kick_proven=False,
        football_skill_proven=False,
    )
    if not len(indices):
        return result
    first = int(indices[0])
    at_first = force[first] > 10
    correct = bool(at_first.sum() == 1 and at_first[intended_foot])
    q = canonical_qpos
    target = goal[:2] - q[first, 36:38]
    norm = float(np.linalg.norm(target))
    if norm <= 1e-9:
        raise ValueError("nondegenerate measured horizontal target direction required")
    direction = target / norm
    end = first + 600
    observed = end < len(force)
    root_travel = (q[first:, :2] - q[first, :2]) @ direction
    result.update(
        first_foot_contact_substep=first,
        correct_foot_first=correct,
        post_contact_status="OBSERVED" if observed else "RIGHT_CENSORED",
        post_contact_quality_observed=observed,
        maximum_pelvis_retreat_after_contact_m=float(max(0.0, -root_travel.min())),
        minimum_post_contact_upright=float((1 - 2 * (q[first:, 4] ** 2 + q[first:, 5] ** 2)).min()),
        minimum_post_contact_pelvis_height_m=float(q[first:, 2].min()),
    )
    if observed:
        travel = q[end, 36:38] - q[first, 36:38]
        result.update(
            goal_progress_1p2s_m=float(travel @ direction),
            goal_lateral_1p2s_m=float(direction[0] * travel[1] - direction[1] * travel[0]),
            other_foot_contact_in_window=bool(
                (force[first : end + 1, 1 - intended_foot] > 10).any()
            ),
        )
    return result
