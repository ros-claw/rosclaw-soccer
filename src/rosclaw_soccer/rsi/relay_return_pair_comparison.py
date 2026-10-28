"""Compare paired SIM return trials without equating proximity with a pass."""

import numpy as np


def compare_return_pair(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    prefix_controls: int,
    ball_qpos: int,
    return_goal: np.ndarray,
) -> dict[str, object]:
    """Verify identical pre-intervention physics and report descriptive differences.

    The closest ball position is an exploratory metric, not evidence of a pass,
    possession, fitness or successful teamwork. It is only measured after the
    intervention in handoff cases, and is undefined when no handoff occurred.
    """
    if type(prefix_controls) is not int or prefix_controls < 1:
        raise ValueError("positive matching prefix required")
    if type(ball_qpos) is not int or ball_qpos < 0:
        raise ValueError("valid ball position address required")
    goal = np.asarray(return_goal)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("finite 3D goal required")
    for data in (baseline, candidate):
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            dimensions = 4 if key == "ball_forces" else 3
            if key not in data or data[key].ndim != dimensions or len(data[key]) < prefix_controls:
                raise ValueError("complete native substep trajectory required")
            if data[key].shape[1] != 10 or not np.isfinite(data[key]).all():
                raise ValueError("finite 10-substep native trajectory required")
        n = len(data["qpos"])
        if any(len(data[k]) != n for k in ("qvel", "torque", "ball_forces")):
            raise ValueError("physical trajectory lengths differ")
        if ball_qpos + 3 > data["qpos"].shape[2]:
            raise ValueError("ball address outside qpos")
        if data["ball_forces"].shape[2:] != (2, 3):
            raise ValueError("two bodies and three contact groups required")
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        if not np.array_equal(baseline[key][:prefix_controls], candidate[key][:prefix_controls]):
            raise ValueError("matched physical prefix differs before intervention")

    def minimum(data: dict[str, np.ndarray]) -> float | None:
        ball = data["qpos"][prefix_controls:, :, ball_qpos : ball_qpos + 3]
        if not ball.size:
            return None
        return float(np.linalg.norm(ball - goal, axis=-1).min())

    def contact_counts(data: dict[str, np.ndarray]) -> dict[str, int]:
        # Count native substeps with >1 N contact, not distinct kicks or passes.
        forces = data["ball_forces"][prefix_controls:, :, 1, :]
        return {
            label: int(np.count_nonzero(forces[:, :, i] > 1.0))
            for i, label in enumerate(("left_foot", "right_foot", "nonfoot"))
        }

    first, second = minimum(baseline), minimum(candidate)
    return {
        "pre_intervention_controls_exact": prefix_controls,
        "baseline_post_handoff_controls": len(baseline["qpos"]) - prefix_controls,
        "candidate_post_handoff_controls": len(candidate["qpos"]) - prefix_controls,
        "baseline_min_ball_goal_distance_m": first,
        "candidate_min_ball_goal_distance_m": second,
        "distance_delta_m": None if first is None or second is None else second - first,
        "baseline_receiver_contact_substeps": contact_counts(baseline),
        "candidate_receiver_contact_substeps": contact_counts(candidate),
        "contact_force_threshold_n": 1.0,
        "goal_proximity_is_not_pass_success": True,
    }
