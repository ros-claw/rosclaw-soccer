"""Descriptive SIM teammate/ball state following an ordered relay contact."""

import numpy as np


def teammate_successor(
    qpos: np.ndarray,
    qvel: np.ndarray,
    forces: np.ndarray,
    *,
    return_start: int,
    ball_q: int,
    ball_v: int,
    teammate_q: int,
    teammate_v: int,
    timestep: float,
) -> dict[str, object]:
    """Measure relative planar state, never label root proximity as possession."""
    if (
        qpos.ndim != 3
        or qvel.ndim != 3
        or forces.ndim != 4
        or qpos.shape[:2] != qvel.shape[:2]
        or qpos.shape[:2] != forces.shape[:2]
        or forces.shape[1:] != (10, 2, 3)
        or any(not np.isfinite(a).all() for a in (qpos, qvel, forces))
        or type(return_start) is not int
        or not 0 <= return_start <= len(qpos)
        or not np.isfinite(timestep)
        or timestep <= 0
    ):
        raise ValueError("finite consistent native shared trajectories required")
    for address, width in (
        (ball_q, qpos.shape[2]),
        (teammate_q, qpos.shape[2]),
        (ball_v, qvel.shape[2]),
        (teammate_v, qvel.shape[2]),
    ):
        if type(address) is not int or address < 0 or address + 2 > width:
            raise ValueError("valid planar state addresses required")
    q = qpos[return_start:].reshape(-1, qpos.shape[-1])
    v = qvel[return_start:].reshape(-1, qvel.shape[-1])
    f = forces[return_start:].reshape(-1, 2, 3)
    b = np.flatnonzero(np.any(f[:, 1, :2] > 1, axis=1))
    if not b.size:
        return {"status": "NO_RETURNER_FOOT_CONTACT", "samples": []}
    a = np.flatnonzero(np.any(f[:, 0, :2] > 1, axis=1) & (np.arange(len(f)) > b[0]))
    if not a.size:
        return {"status": "NO_LATER_TEAMMATE_FOOT_CONTACT", "samples": []}
    first = int(a[0])
    samples = []
    for seconds in (0.0, 0.5, 1.0, 2.0):
        index = first + int(round(seconds / timestep))
        if index >= len(q):
            samples.append({"seconds_after_contact": seconds, "status": "RIGHT_CENSORED"})
            continue
        samples.append(
            {
                "seconds_after_contact": seconds,
                "status": "OBSERVED",
                "ball_to_pelvis_planar_m": float(
                    np.linalg.norm(
                        q[index, ball_q : ball_q + 2] - q[index, teammate_q : teammate_q + 2]
                    )
                ),
                "relative_planar_speed_mps": float(
                    np.linalg.norm(
                        v[index, ball_v : ball_v + 2] - v[index, teammate_v : teammate_v + 2]
                    )
                ),
            }
        )
    return {"status": "ORDERED_FOOT_CONTACT_NOT_POSSESSION", "samples": samples}
