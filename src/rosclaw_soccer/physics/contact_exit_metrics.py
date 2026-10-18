"""Read-only contact/linear-momentum diagnostics; never a policy reward or pass gate."""

from typing import Any

import numpy as np


def contact_exit_metrics(
    arrays: dict[str, Any], events: list[dict[str, Any]], *, mass_kg: float, dt_s: float
) -> dict[str, Any]:
    if not np.isfinite(mass_kg) or mass_kg <= 0 or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("positive finite mass and timestep required")
    expected = {
        "canonical_qvel_before_substep": (3000, 41),
        "canonical_qvel_after_substep": (3000, 41),
        "all_contact_force_on_ball_world_n": (3000, 3),
        "robot_contact_force_on_ball_world_n": (3000, 3),
    }
    for key, shape in expected.items():
        a = np.asarray(arrays.get(key))
        if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError("complete finite dense canonical/force arrays required")
    before = arrays["canonical_qvel_before_substep"][:, 35:38]
    after = arrays["canonical_qvel_after_substep"][:, 35:38]
    if not np.array_equal(before[1:], after[:-1]):
        raise ValueError("contiguous before/after ball translation clock required")
    total = arrays["all_contact_force_on_ball_world_n"]
    robot = arrays["robot_contact_force_on_ball_world_n"]
    # Applicable only to the pinned replay world's free ball: no damping/external force.
    residual = mass_kg * (after - before) / dt_s - total - mass_kg * np.array([0, 0, -9.81])
    steps: set[int] = set()
    nonfoot: set[int] = set()
    for event in events:
        step, force = event["substep_index"], event["force_norm_n"]
        if type(step) is not int or not 0 <= step < 3000 or not np.isfinite(force) or force <= 1:
            raise ValueError("actual over-1N contact events and dense indices required")
        target = (
            steps
            if event["other_body_name"] in ("left_ankle_roll_link", "right_ankle_roll_link")
            else nonfoot
        )
        target.add(step)
    ordered = sorted(steps)
    longest = current = 0
    previous = None
    for step in ordered:
        current = current + 1 if previous is not None and step == previous + 1 else 1
        longest = max(longest, current)
        previous = step
    first = ordered[0] if ordered else None

    def interval(start: int, end: int) -> dict[str, Any]:
        return dict(
            start_substep=start,
            end_substep_exclusive=end,
            integrated_duration_s=(end - start) * dt_s,
            robot_impulse_world_n_s=(robot[start:end].sum(axis=0) * dt_s).tolist(),
            floor_impulse_world_n_s=(
                (total[start:end] - robot[start:end]).sum(axis=0) * dt_s
            ).tolist(),
            gravity_impulse_world_n_s=[0.0, 0.0, -9.81 * mass_kg * (end - start) * dt_s],
            ball_momentum_change_world_kg_m_s=(mass_kg * (after[end - 1] - before[start])).tolist(),
        )

    early = None if first is None else interval(first, min(first + 10, 3000))
    reversal, slowing = None, None
    if first is not None and first + 10 < 3000 and after[first + 9, 0] > 0:
        crossings = np.flatnonzero((before[:, 0] >= 0) & (after[:, 0] < 0))
        crossings = crossings[crossings >= first + 10]
        if len(crossings):
            reversal = int(crossings[0])
            slowing = interval(first + 10, reversal + 1)
    return dict(
        schema="soccer.physics.contact_exit_metrics.v1",
        first_foot_contact_substep=first,
        last_foot_contact_substep=None if not ordered else ordered[-1],
        unique_active_foot_substeps=len(ordered),
        summed_active_foot_duration_s=len(ordered) * dt_s,
        first_to_last_foot_contact_span_s=None
        if not ordered
        else (ordered[-1] - ordered[0]) * dt_s,
        longest_contiguous_active_foot_substeps=longest,
        longest_contiguous_active_foot_duration_s=longest * dt_s,
        first_nonfoot_robot_contact_substep=min(nonfoot) if nonfoot else None,
        first_forward_to_backward_crossing_after_initial_20ms_substep=reversal,
        total_robot_downward_impulse_n_s=float(np.maximum(-robot[:, 2], 0).sum() * dt_s),
        maximum_robot_downward_force_n=float(np.maximum(-robot[:, 2], 0).max()),
        initial_20ms=early,
        slowing_until_reversal=slowing,
        maximum_linear_newton_residual_n=float(np.max(np.abs(residual))),
        world_gravity_m_s2=[0.0, 0.0, -9.81],
        angular_velocity_interpreted=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        skill_success_inferred=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
