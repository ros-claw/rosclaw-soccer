"""Build one contact reference from an OWN current-state native NN future.

No recorded future, course identity or outcome is accepted as a control input.
Only scratch FK references are modified; every execution stays a separate step.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.absolute_contact_reference import AbsoluteContactReferencePlan
from rosclaw_soccer.rsi.contact_reference_ik import first_contact_foot, solve_contact_reference_pose
from rosclaw_soccer.rsi.contact_reference_plan import ContactReferencePlan
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer


def generate_contact_reference(
    player: SharedRoboNaldoPlayer,
    forecaster: Any,
    *,
    absolute_waypoint: bool = False,
    bounded_absolute_curve: bool = False,
) -> tuple[
    ContactReferencePlan | AbsoluteContactReferencePlan | None, dict[str, Any], dict[str, Any]
]:
    import mujoco

    from rosclaw_soccer.rsi.shared_nn_forecast import SharedNNForecaster, policy_state_hash

    world = player.world
    world._require_current_boundary()
    if (
        type(absolute_waypoint) is not bool
        or type(bounded_absolute_curve) is not bool
        or (bounded_absolute_curve and not absolute_waypoint)
    ):
        raise ValueError("explicit absolute-reference experiment switch required")
    if (
        type(forecaster) is not SharedNNForecaster
        or forecaster.player is not player
        or player.pending is not None
        or player.proposals
        or player.failed
        or player.retired
        or player.measured_entry_receipt is None
    ):
        raise ValueError(
            "own current admitted pre-proposal SIM policy and bound native forecaster required"
        )
    before = policy_state_hash(player.policy)
    parent = forecaster.forecast(controls=128, before_proposal=True)
    report: dict[str, Any] = dict(
        schema="soccer.current_nn_contact_reference_generation.v1",
        parent_controls=parent["controls"],
        parent_body_failed=parent["body_failed"],
        recorded_future_used=False,
        source_case_id_used=False,
        hardware_authorized=False,
        absolute_waypoint=absolute_waypoint,
    )
    if bounded_absolute_curve:
        report["bounded_absolute_curve"] = True
    plan = None
    if parent["controls"] != 128 or parent["body_failed"]:
        report["status"] = "ABSTAIN_INCOMPLETE_PARENT_FUTURE"
    else:
        contact = first_contact_foot(parent["ball_forces"][:, 0, :2])
        if contact is None:
            report["status"] = "ABSTAIN_NO_PREDICTED_FIRST_FOOT"
        else:
            foot, native = contact
            peak = native // 10 - 1
            report.update(contact_foot=foot, first_foot_native_index=native, peak_control=peak)
            if not 1 <= peak < 48:
                report["status"] = "ABSTAIN_NO_BOUNDED_CONTACT_REFERENCE_WINDOW"
            else:
                seed = mujoco.MjData(world.model)
                seed.qpos[:], seed.qvel[:] = (
                    parent["qpos"][peak * 10 - 1],
                    parent["qvel"][peak * 10 - 1],
                )
                seed.time = float(world.data.time) + 0.02 * peak
                ik = solve_contact_reference_pose(
                    world.model,
                    seed,
                    player.layout,
                    goal=player.goal.astype(np.float64),
                    active_foot=foot,
                )
                report["ik"] = ik
                if ik["status"] != "KINEMATIC_CONTACT_CONFIGURATION_FEASIBLE":
                    report["status"] = "ABSTAIN_CONTACT_CONFIGURATION_INFEASIBLE"
                else:
                    from importlib import import_module

                    mapping = tuple(
                        int(v)
                        for v in import_module(type(player.policy).__module__).MUJOCO_TO_ISAAC
                    )
                    delta = np.array(ik["reference_joints_hg"]) - seed.qpos[player.layout.qi]
                    start = int(player.policy.time_step) - int(player.policy.WARMUP_STEPS)
                    candidate: ContactReferencePlan | AbsoluteContactReferencePlan
                    candidate = (
                        AbsoluteContactReferencePlan(
                            start,
                            mapping,
                            tuple(float(v) for v in ik["reference_joints_hg"]),
                            peak,
                            bounded_curve=bounded_absolute_curve,
                        )
                        if absolute_waypoint
                        else ContactReferencePlan(
                            start, mapping, tuple(float(v) for v in delta), peak
                        )
                    )
                    try:
                        pos, vel = (
                            candidate.offsets(
                                player.policy,
                                limits_hg=world.model.jnt_range[player.layout.joints].astype(
                                    np.float64
                                ),
                            )
                            if isinstance(candidate, AbsoluteContactReferencePlan)
                            else candidate.offsets()
                        )
                    except ValueError as error:
                        report.update(
                            status="ABSTAIN_REFERENCE_TRAJECTORY_LIMIT", reason=str(error)
                        )
                    else:
                        plan = candidate
                        report.update(
                            status="REFERENCE_PROPOSED_NOT_PHYSICALLY_QUALIFIED",
                            start_frame=candidate.start_frame,
                            delta_hg=tuple(float(v) for v in pos[peak, mapping]),
                            maximum_reference_offset_rad=float(abs(pos).max()),
                            maximum_additional_reference_velocity_rad_s=float(abs(vel).max()),
                        )
    world._require_current_boundary()
    if policy_state_hash(player.policy) != before:
        player.failed = True
        raise ValueError("contact reference generation mutated actual policy history")
    return plan, report, parent
