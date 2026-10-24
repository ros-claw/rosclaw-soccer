"""Separate applicability, actual neural action change and physical success."""

from typing import Any

import numpy as np


def supported_action_effects(
    activation: np.ndarray,
    actual: np.ndarray,
    parent_on_own_states: np.ndarray,
    scale: np.ndarray,
    *,
    first_foot_substep: int | None,
) -> dict[str, Any]:
    n = len(actual)
    if (
        not 1 <= n <= 512
        or actual.shape != (n, 29)
        or parent_on_own_states.shape != actual.shape
        or activation.shape != (n, 1)
        or scale.shape != (29,)
        or any(
            x.dtype != np.float32 or not np.isfinite(x).all()
            for x in (activation, actual, parent_on_own_states, scale)
        )
        or (scale <= 0).any()
        or ((activation < 0) | (activation > 1)).any()
        or (
            first_foot_substep is not None
            and (type(first_foot_substep) is not int or first_foot_substep < 0)
        )
    ):
        raise ValueError("finite actual own-state action/support/contact arrays required")
    active = activation[:, 0] > 0
    changed = np.any(actual != parent_on_own_states, axis=1)
    if (changed & ~active).any():
        raise ValueError("zero applicability must preserve exact entire parent action")
    delta = (actual.astype(np.float64) - parent_on_own_states.astype(np.float64)) * scale
    indices = np.flatnonzero(active).tolist()
    segments: list[list[int]] = []
    for i in indices:
        if not segments or i != segments[-1][1] + 1:
            segments.append([i, i])
        else:
            segments[-1][1] = i
    contact_control = None if first_foot_substep is None else first_foot_substep // 10
    last = indices[-1] if indices else None
    before_contact = [i for i in indices if contact_control is not None and i <= contact_control]
    last_before_contact = before_contact[-1] if before_contact else None
    return dict(
        rows=n,
        support_active_rows=int(active.sum()),
        actual_changed_rows=int(changed.sum()),
        material_changed_rows_1e5rad=int((np.max(np.abs(delta), axis=1) > 1e-5).sum()),
        maximum_action_change_vs_own_parent_rad=float(np.abs(delta).max()),
        support_segments_in_control_rows=segments,
        first_support_control=indices[0] if indices else None,
        last_support_control=last,
        last_support_before_contact_control=last_before_contact,
        first_foot_contact_control=contact_control,
        support_active_at_first_contact=(
            bool(active[contact_control])
            if contact_control is not None and contact_control < n
            else None
        ),
        gap_after_last_support_before_contact_controls=(
            max(0, contact_control - last_before_contact - 1)
            if contact_control is not None and last_before_contact is not None
            else None
        ),
        maximum_support_change_per_control=float(np.abs(np.diff(activation[:, 0])).max())
        if n > 1
        else 0.0,
        maximum_executed_action_step_rad=float(
            np.abs(np.diff(actual.astype(np.float64), axis=0) * scale).max()
        )
        if n > 1
        else 0.0,
        parent_query_is_on_candidate_observed_stream=True,
        parent_counterfactual_world_outcome_proven=False,
        physical_success_inferred_from_activation=False,
    )
