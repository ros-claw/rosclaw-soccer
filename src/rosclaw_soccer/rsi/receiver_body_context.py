"""Actual body-frame receiving cues, not future flight or course identifiers."""

import numpy as np


def receiver_context14(
    qpos: np.ndarray,
    qvel: np.ndarray,
    anchor_world: np.ndarray,
    *,
    foot_contact_seen: bool,
    other_body_contact_seen: bool,
) -> np.ndarray:
    """Canonical robot-first43/41 state at the current PRE-action boundary.

    Channels: ball displacement3, relative translational velocity3, anchor
    displacement3, root linear velocity3, past foot/other-body contact flags2.
    The latter are measured contact categories, NOT future legal/success labels.
    Vectors are in the measured root frame; relative velocity is NOT labelled
    as the derivative of the rotating-frame relative position.
    """
    if (
        qpos.shape != (43,)
        or qvel.shape != (41,)
        or anchor_world.shape != (3,)
        or any(
            x.dtype != np.float64 or not np.isfinite(x).all() for x in (qpos, qvel, anchor_world)
        )
        or abs(np.linalg.norm(qpos[3:7]) - 1) > 1e-5
        or type(foot_contact_seen) is not bool
        or type(other_body_contact_seen) is not bool
    ):
        raise ValueError(
            "actual canonical finite float64 state, unit root and past contact flags required"
        )
    w, x, y, z = qpos[3:7]
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.concatenate(
        (
            rotation.T @ (qpos[36:39] - qpos[:3]),
            rotation.T @ (qvel[35:38] - qvel[:3]),
            rotation.T @ (anchor_world - qpos[:3]),
            rotation.T @ qvel[:3],
            np.array([foot_contact_seen, other_body_contact_seen], dtype=np.float64),
        )
    ).astype(np.float32)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError("finite float32 own body cues required, never padded or sanitized")
    return result
