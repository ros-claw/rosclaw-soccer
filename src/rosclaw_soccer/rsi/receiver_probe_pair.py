"""Measured same-boundary SIM probe comparison, not a reception certificate."""

import numpy as np

from rosclaw_soccer.rsi.receiver_first_contact_dynamics import first_contact_dynamics


def compare_receiver_probe_pair(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    ball_q: int,
    ball_v: int,
    receiver_q: int,
    receiver_v: int,
) -> dict:
    """Compare only equal exposure; an early stop is never a success."""
    shapes = {"qpos": (10, 79), "qvel": (10, 76), "torque": (10, 58), "ball_forces": (10, 2, 3)}
    for run in (baseline, candidate):
        if type(run) is not dict or not all(k in run for k in shapes):
            raise ValueError("both physical trajectories required")
        lengths = set()
        for name, shape in shapes.items():
            a = run[name]
            if (
                type(a) is not np.ndarray
                or a.ndim != len(shape) + 1
                or a.shape[1:] != shape
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("finite aligned shared native substeps required")
            lengths.add(len(a))
        if len(lengths) != 1 or not 1 <= lengths.pop() <= 36:
            raise ValueError("bounded internally aligned physical history required")
    if any(not np.array_equal(baseline[k][0], candidate[k][0]) for k in shapes):
        raise ValueError("identical first actual shared control required")
    if any(type(i) is not int for i in (ball_q, ball_v, receiver_q, receiver_v)):
        raise ValueError("explicit model-derived addresses required")
    common = min(len(baseline["qpos"]), len(candidate["qpos"]))

    def measured(run: dict[str, np.ndarray]) -> dict:
        q, v = run["qpos"][:common], run["qvel"][:common]
        contact = first_contact_dynamics(
            q,
            v,
            run["ball_forces"][:common],
            ball_q=ball_q,
            ball_v=ball_v,
            receiver_q=receiver_q,
            receiver_v=receiver_v,
        )
        return dict(
            first_contact=contact,
            final_relative_speed_mps=float(
                np.linalg.norm(
                    v[-1, -1, ball_v : ball_v + 2] - v[-1, -1, receiver_v : receiver_v + 2]
                )
            ),
            final_root_ball_distance_m=float(
                np.linalg.norm(
                    q[-1, -1, ball_q : ball_q + 2] - q[-1, -1, receiver_q : receiver_q + 2]
                )
            ),
            minimum_receiver_height_m=float(np.min(q[:, :, receiver_q + 2])),
        )

    divergences = [
        i
        for i in range(common)
        if any(not np.array_equal(baseline[k][i], candidate[k][i]) for k in shapes)
    ]
    return dict(
        common_controls=common,
        baseline_controls=len(baseline["qpos"]),
        candidate_controls=len(candidate["qpos"]),
        same_exposure=len(baseline["qpos"]) == len(candidate["qpos"]),
        first_different_control=None if not divergences else divergences[0],
        baseline=measured(baseline),
        candidate=measured(candidate),
        clean_reception_certified=False,
        promotion_authorized=False,
    )
