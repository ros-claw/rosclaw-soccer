"""Offline diagnostic learning of return-foot events, not a control policy."""

import numpy as np


def causal_features(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    start: int,
    ball_q: int,
    ball_v: int,
    passer_q: int,
    passer_v: int,
    returner_q: int,
    returner_v: int,
) -> np.ndarray:
    """Three past/current control-end samples only, with no contact-force input."""
    if (
        qpos.ndim != 3
        or qvel.ndim != 3
        or qpos.shape[:2] != qvel.shape[:2]
        or qpos.shape[1] != 10
        or type(start) is not int
        or start < 4
        or start > len(qpos)
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
    ):
        raise ValueError("finite control-aligned history with three past samples required")
    for address, width, size in (
        (ball_q, qpos.shape[-1], 2),
        (passer_q, qpos.shape[-1], 2),
        (returner_q, qpos.shape[-1], 2),
        (ball_v, qvel.shape[-1], 3),
        (passer_v, qvel.shape[-1], 2),
        (returner_v, qvel.shape[-1], 2),
    ):
        if type(address) is not int or address < 0 or address + size > width:
            raise ValueError("valid native state addresses required")
    q, v = qpos[:, -1], qvel[:, -1]

    def feature(t: int) -> np.ndarray:
        return np.concatenate(
            (
                q[t, ball_q : ball_q + 2] - q[t, passer_q : passer_q + 2],
                q[t, ball_q : ball_q + 2] - q[t, returner_q : returner_q + 2],
                v[t, ball_v : ball_v + 2],
                v[t, passer_v : passer_v + 2],
                v[t, returner_v : returner_v + 2],
                v[t, ball_v : ball_v + 3] - v[t - 1, ball_v : ball_v + 3],
            )
        )

    return np.asarray(
        [
            np.concatenate([feature(i - 3), feature(i - 2), feature(i - 1)])
            for i in range(start, len(q))
        ],
        dtype=np.float32,
    ).reshape(-1, 39)


def event_labels(forces: np.ndarray, *, start: int) -> tuple[np.ndarray, int | None, bool]:
    """Only mark an event observable at the control boundary after contact."""
    if (
        forces.ndim != 4
        or forces.shape[1:] != (10, 2, 3)
        or not np.isfinite(forces).all()
        or not 0 <= start <= len(forces)
    ):
        raise ValueError("finite native contact force sequence required")
    events = np.flatnonzero(np.any(forces[start:, :, 1, :2] > 1, axis=(1, 2)))
    length = len(forces) - start
    if not events.size:
        return np.zeros(length, dtype=np.float32), None, False
    onset = int(events[0]) + 1
    if onset >= length:
        return np.zeros(length, dtype=np.float32), None, True
    labels = np.zeros(length, dtype=np.float32)
    labels[onset:] = 1
    return labels, onset, False


def classify_first(
    predictions: np.ndarray, onset: int | None, censored: bool, *, grace: int = 5
) -> str:
    """Case-level outcome, no invented event for a censored terminal contact."""
    if (
        predictions.ndim != 1
        or predictions.dtype != np.bool_
        or type(grace) is not int
        or grace < 0
    ):
        raise ValueError("boolean sequential predictions required")
    hit = np.flatnonzero(predictions)
    if censored:
        return "CENSORED"
    if onset is None:
        return "FALSE_ALARM" if hit.size else "CORRECT_NO_EVENT"
    if not 0 <= onset < len(predictions):
        raise ValueError("observable onset required")
    if not hit.size:
        return "MISSED"
    if hit[0] < onset:
        return "EARLY_FALSE_ALARM"
    return "TIMELY" if hit[0] <= onset + grace else "LATE"
