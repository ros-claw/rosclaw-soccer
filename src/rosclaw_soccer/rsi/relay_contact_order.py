"""Describe measured contact ordering; chronology alone is not causal attribution."""

import numpy as np


def relay_contact_order(forces: np.ndarray, handoff_control: int | None) -> dict:
    if (
        forces.ndim != 4
        or forces.shape[1:] != (10, 2, 3)
        or not forces.shape[0]
        or not np.isfinite(forces).all()
        or (forces < 0).any()
        or (
            handoff_control is not None
            and (type(handoff_control) is not int or not 0 <= handoff_control < forces.shape[0])
        )
    ):
        raise ValueError("finite relay substeps and own valid handoff control required")
    flat = forces.reshape(-1, 2, 3)
    source = np.flatnonzero(np.max(flat[:, 0, :2], axis=1) > 10)
    receiver = np.flatnonzero(np.max(flat[:, 1, :], axis=1) > 1)
    first_source = int(source[0]) if source.size else None
    first_receiver = int(receiver[0]) if receiver.size else None
    ordered = bool(
        first_source is not None
        and first_receiver is not None
        and handoff_control is not None
        and first_source < first_receiver < handoff_control * 10
    )
    return {
        "first_source_foot_substep": first_source,
        "first_receiver_any_substep": first_receiver,
        "ordered_source_then_receiver_before_handoff": ordered,
        "causal_attribution_proven": False,
    }
