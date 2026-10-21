"""Declared SIM practice ball drop; never reset the robot or disguise a match."""

from typing import Any

import numpy as np


def snapshot(data: Any) -> dict[str, np.ndarray]:
    return {
        **{
            k: getattr(data, k).copy()
            for k in (
                "qpos",
                "qvel",
                "ctrl",
                "qacc_warmstart",
                "mocap_pos",
                "mocap_quat",
                "qfrc_applied",
                "xfrc_applied",
                "act",
            )
        },
        "time": np.array(data.time),
    }


def practice_ball_drop(ep: Any, *, foot: str) -> dict[str, np.ndarray]:
    """Only an out-of-drill-zone ball may be returned by this explicit fixture.

    The new ball falls under gravity from .4m at a declared .8m forward/.15m
    lateral position. Scratch collision admission precedes the ball-only write.
    No robot, clock, warmstart, controls or goal-marker writes are permitted.
    This is NOT a teammate pass or uninterrupted autonomous game.
    """
    import mujoco

    if foot not in ("left", "right") or ep.termination_reason != "REFERENCE_COMPLETE":
        raise ValueError("healthy completed reference and explicit next foot required")
    m, d, layout = ep.env.model, ep.env.data, ep.layout
    before = snapshot(d)
    q, _ = layout.canonical(d)
    if np.linalg.norm(q[36:38] - q[:2]) < 2:
        return {"performed": np.array(False), **{"before_" + k: v for k, v in before.items()}}
    forward = ep.goal[:2] - q[:2]
    norm = np.linalg.norm(forward)
    if not np.isfinite(norm) or norm < 1:
        raise ValueError("practice requires an unchanged distant world goal")
    forward /= norm
    lateral = np.array([-forward[1], forward[0]])
    point = np.r_[q[:2] + 0.8 * forward + (0.15 if foot == "left" else -0.15) * lateral, 0.4]
    trial = mujoco.MjData(m)
    mujoco.mj_copyData(trial, m, d)
    trial.qpos[layout.bq : layout.bq + 7] = np.r_[point, 1.0, 0, 0, 0]
    trial.qvel[layout.bv : layout.bv + 6] = 0
    mujoco.mj_forward(m, trial)
    for contact in trial.contact:
        x, y = (int(m.geom_bodyid[g]) for g in contact.geom)
        if (x == layout.ball and y in ep.robot) or (y == layout.ball and x in ep.robot):
            raise ValueError("declared practice drop overlaps body; retain failure, no resample")
    d.qpos[layout.bq : layout.bq + 7] = trial.qpos[layout.bq : layout.bq + 7]
    d.qvel[layout.bv : layout.bv + 6] = 0
    after = snapshot(d)
    for k in before:
        actual = after[k].copy()
        if k == "qpos":
            actual[layout.bq : layout.bq + 7] = before[k][layout.bq : layout.bq + 7]
        if k == "qvel":
            actual[layout.bv : layout.bv + 6] = before[k][layout.bv : layout.bv + 6]
        if not np.array_equal(before[k], actual):
            raise ValueError("practice fixture changed non-ball integration state")
    mujoco.mj_forward(m, d)
    return {
        "performed": np.array(True),
        **{"before_" + k: v for k, v in before.items()},
        **{"after_" + k: v for k, v in after.items()},
    }
