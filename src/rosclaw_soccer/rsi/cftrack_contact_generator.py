"""Current OWN CFtrack future -> absolute contact IK -> ankle-pitch reference.

Saved episodes, case IDs and source outcomes are not accepted as control inputs.
IK's ankle-roll geometry is NOT a promise the ankle-pitch tracker will realize it.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_contact_reference import (
    CFtrackContactPlan,
    condition_cftrack_contact_reference,
)
from rosclaw_soccer.rsi.contact_reference_ik import first_contact_foot, solve_absolute_contact_pose
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash, copy_cftrack_policy
from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


def generate_cftrack_contact_plan(
    player: SharedOmniContactPlayer,
    forecaster: SharedCFtrackForecaster,
) -> tuple[CFtrackContactPlan | None, dict[str, Any], dict[str, Any]]:
    import mujoco

    world = player.world
    world._require_current_boundary()
    if (
        type(player) is not SharedOmniContactPlayer
        or type(forecaster) is not SharedCFtrackForecaster
        or forecaster.player is not player
        or player.failed
        or player.pending is not None
        or player.boundary != world.control_steps
    ):
        raise ValueError("current own pre-proposal CFtrack player/forecaster required")
    before = cftrack_state_hash(player.policy)
    parent = forecaster.forecast(controls=160)
    report: dict[str, Any] = dict(
        schema="soccer.own_cftrack_contact_generation.v1",
        parent_controls=parent["controls"],
        recorded_future_used=False,
        source_case_identity_used=False,
        task_reference_link="ankle_pitch_not_direct_ankle_roll",
        hardware_authorized=False,
    )
    plan = None
    try:
        if (
            parent["controls"] != 160
            or parent["body_failed"]
            or parent["runtime_failure"] is not None
        ):
            report["status"] = "ABSTAIN_INCOMPLETE_PARENT"
            return plan, report, parent
        contact = first_contact_foot(parent["ball_forces"][:, 0, :2])
        if contact is None:
            report["status"] = "ABSTAIN_NO_PREDICTED_FOOT_CONTACT"
            return plan, report, parent
        side, native = contact
        ahead = native // 10 - 1
        origin = int(player.policy.counter_step)
        peak_frame = origin + ahead
        start = max(origin, peak_frame - 24)
        peak = peak_frame - start
        report.update(
            active_foot=side,
            first_foot_native=native,
            peak_ahead=ahead,
            start_frame=start,
            peak_control=peak,
        )
        if not 1 <= peak < 48 or start + 48 >= len(player.policy.ref_left_ankle_future_pos):
            report["status"] = "ABSTAIN_INCOMPLETE_REFERENCE_WINDOW"
            return plan, report, parent
        scratch = mujoco.MjData(world.model)
        scratch.qpos[:], scratch.qvel[:] = (
            parent["qpos"][ahead * 10 - 1],
            parent["qvel"][ahead * 10 - 1],
        )
        scratch.time = float(world.data.time) + 0.02 * ahead
        ik = solve_absolute_contact_pose(
            world.model,
            scratch,
            player.layout,
            goal=player.policy.goal_pos.astype(np.float64),
            active_foot=side,
        )
        report["ik"] = ik
        if ik["status"] != "KINEMATIC_CONTACT_CONFIGURATION_FEASIBLE":
            report["status"] = "ABSTAIN_ABSOLUTE_CONTACT_GEOMETRY"
            return plan, report, parent
        scratch.qpos[player.layout.qi] = np.array(ik["reference_joints_hg"])
        mujoco.mj_kinematics(world.model, scratch)
        body = int(world.model.body(player.prefix + side + "_ankle_pitch_link").id)
        candidate = CFtrackContactPlan(
            origin,
            start,
            peak,
            side,
            tuple(float(v) for v in scratch.xpos[body]),
            tuple(float(v) for v in scratch.xquat[body]),
            before,
        )
        try:
            receipt = condition_cftrack_contact_reference(
                copy_cftrack_policy(player.policy), candidate
            )
        except ValueError as error:
            report.update(status="ABSTAIN_TASK_REFERENCE_ENVELOPE", reason=str(error))
        else:
            plan = candidate
            report.update(
                status="TASK_REFERENCE_PROPOSED_NOT_PHYSICALLY_QUALIFIED",
                prospective_installation=receipt,
            )
        return plan, report, parent
    finally:
        world._require_current_boundary()
        if cftrack_state_hash(player.policy) != before:
            player.failed = True
            raise ValueError("CFtrack contact generation changed actual policy/FK state")
