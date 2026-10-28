"""Retrospective legacy feature proximity, not a live policy or transfer proof."""

import numpy as np


def legacy_proximity(
    qpos: np.ndarray, qvel: np.ndarray, past_contact: np.ndarray, anchors: np.ndarray
) -> dict:
    """Search actual precontact states with a hypothetical old local frame=19.

    Absolute historical frame and body/action semantics differ across worlds.
    The phase override relaxes the old clock, but the search is restricted to
    states preceding any observed foot or nonfoot contact. It is neither an
    overall upper bound nor an event clock, controller or intervention.
    """
    n = len(qpos)
    if (
        qpos.shape != (n, 43)
        or qvel.shape != (n, 41)
        or past_contact.shape != (n, 2)
        or anchors.shape != (3, 10)
        or not 1 <= n <= 300
        or past_contact.dtype != np.bool_
        or any(not np.isfinite(x).all() for x in (qpos, qvel, anchors))
        or np.any(np.diff(past_contact.astype(np.int8), axis=0) < 0)
    ):
        raise ValueError(
            "complete finite actual canonical states and persistent past contact required"
        )
    delta = qpos[:, 36:39] - qpos[:, :3]
    rel = qvel[:, 35:38] - qvel[:, :3]
    features = np.stack(
        (
            delta[:, 0],
            delta[:, 1] / 0.2,
            delta[:, 2] / 0.5,
            rel[:, 0] / 2.0,
            rel[:, 1] / 2.0,
            qvel[:, 0] / 2.0,
            qvel[:, 1] / 2.0,
            (qpos[:, 2] - 0.7) / 0.3,
            np.full(n, 0.08),
            np.zeros(n),
        ),
        axis=1,
    )
    np.clip(features, -2.0, 2.0, out=features)
    eligible = ~past_contact.any(axis=1)
    if not eligible.any():
        return dict(
            precontact_states=0,
            nearest_distance=None,
            nearest_control=None,
            any_relaxed_gate_match=False,
        )
    distances = np.linalg.norm(features[:, None, :] - anchors[None, :, :], axis=2).min(axis=1)
    indices = np.flatnonzero(eligible)
    index = int(indices[np.argmin(distances[indices])])
    matches = eligible & (distances <= 0.017) & (features[:, 3] >= -0.48)
    return dict(
        precontact_states=len(indices),
        nearest_distance=float(distances[index]),
        nearest_control=index,
        nearest_world_relative_vx_mps=float(rel[index, 0]),
        any_relaxed_gate_match=bool(matches.any()),
        phase19_is_hypothetical=True,
        physical_transfer_proven=False,
    )
