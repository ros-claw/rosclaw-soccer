"""Compare full native SIM trajectories from the same restored control."""

import numpy as np

from rosclaw_soccer.rsi.receiver_first_contact_dynamics import first_contact_dynamics


def compare_long_pair(baseline: dict, candidate: dict, *, layout: dict) -> dict:
    """Report matched common-exposure measurements without certifying possession."""
    shapes = {"qpos": (10, 79), "qvel": (10, 76), "torque": (10, 58), "ball_forces": (10, 2, 3)}
    for run in (baseline, candidate):
        lengths = set()
        for key, shape in shapes.items():
            array = run[key]
            if (
                type(array) is not np.ndarray
                or array.dtype != np.float64
                or array.ndim != len(shape) + 1
                or array.shape[1:] != shape
                or not np.isfinite(array).all()
            ):
                raise ValueError("finite actual shared native substeps required")
            lengths.add(len(array))
        if len(lengths) != 1 or not 1 <= lengths.pop() <= 460:
            raise ValueError("aligned bounded native history required")
        action, obs = run["executed_receiver_action"], run["receiver_observation"]
        if (
            action.dtype != np.float32
            or obs.dtype != np.float32
            or action.ndim != 2
            or action.shape[1:] != (29,)
            or obs.shape != (len(action), 98)
            or not 1 <= len(action) <= len(run["qpos"])
            or not np.isfinite(action).all()
            or not np.isfinite(obs).all()
            or not np.array_equal(obs[1:, 69:98], action[:-1])
        ):
            raise ValueError("each velocity observation must use its own previous applied action")
    if any(not np.array_equal(baseline[k][0], candidate[k][0]) for k in shapes):
        raise ValueError("the first physical control must match exactly")
    common = min(len(baseline["qpos"]), len(candidate["qpos"]))
    metrics = []
    for run in (baseline, candidate):
        q = run["qpos"][:common]
        v = run["qvel"][:common]
        metrics.append(first_contact_dynamics(q, v, run["ball_forces"][:common], **layout))
    return dict(
        baseline_controls=len(baseline["qpos"]),
        candidate_controls=len(candidate["qpos"]),
        common_controls=common,
        same_exposure=len(baseline["qpos"]) == len(candidate["qpos"]),
        baseline=metrics[0],
        candidate=metrics[1],
        clean_reception_certified=False,
    )
