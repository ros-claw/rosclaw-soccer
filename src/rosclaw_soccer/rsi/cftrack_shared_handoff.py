"""Experimental SIM-only own-history transfer into a shared CFtrack player.

This constructs no policy or world, integrates no physics and does not
qualify or promote the resulting controller. The caller must supply an
already pinned fresh policy, an actual shared body and its own live history.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
from rosclaw_soccer.rsi.cftrack_measured_history_install import install_owned_cftrack_history
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


def handoff_shared_cftrack(
    world: SharedNativeMotorWorld,
    prefix: str,
    history: CFtrackLiveHistory,
    policy: Any,
    *,
    expected_policy_type: type[Any],
    goal: np.ndarray,
) -> tuple[SharedOmniContactPlayer, dict[str, Any]]:
    """Switch only one body's gains and install its own five measured controls.

    Physical integration never advances here. If construction or history
    installation fails before any physical step, restore the prior PD profile
    and remove only the newly registered private owner; propagate the failure.
    Restoring gains is not an assertion that running the old motor is safe.
    """
    if (
        type(world) is not SharedNativeMotorWorld
        or type(history) is not CFtrackLiveHistory
        or not history.belongs_to(world, prefix)
        or prefix not in world.parameters
        or prefix in world._private_policy_owners
        or prefix in world._torque_layer_owners
        or type(policy) is not expected_policy_type
        or not isinstance(goal, np.ndarray)
        or goal.shape != (3,)
        or goal.dtype.kind != "f"
        or not np.isfinite(goal).all()
    ):
        raise ValueError("fresh private policy and matching actual shared body history required")
    world._require_current_boundary()
    samples = history.at_current_boundary()
    if len(samples) != 5:
        raise ValueError("five consecutive actual boundaries required")
    before = snapshot(world.data)
    layout = world.layouts[prefix]
    ball = world.data.xpos[layout.ball].copy(), world.data.xquat[layout.ball].copy()
    old = world.parameters[prefix].owned_copy()
    identities = (id(policy), id(policy.state_cmd), id(policy.policy_output))
    try:
        order = np.asarray(policy.lab2mj)
        if (
            order.shape != (29,)
            or order.dtype.kind not in "iu"
            or sorted(order.tolist()) != list(range(29))
        ):
            raise ValueError("valid CFtrack joint order required before gain switch")
        gains = NativePDParameters(
            np.asarray(policy.kps_lab)[order], np.asarray(policy.kds_lab)[order], old.effort
        )
        switch = world.switch_pd_profile(prefix, gains)
        player = SharedOmniContactPlayer(
            world, prefix, policy, expected_policy_type=expected_policy_type, goal=goal
        )
        receipt = install_owned_cftrack_history(player, history)
        after = snapshot(world.data)
        if (
            any(not np.array_equal(before[k], after[k]) for k in before)
            or not np.array_equal(ball[0], world.data.xpos[layout.ball])
            or not np.array_equal(ball[1], world.data.xquat[layout.ball])
        ):
            raise ValueError("shared body, ball and physical clock must remain unchanged")
        return player, dict(
            control_step=world.control_steps,
            time_sec=float(world.data.time),
            previous_target=world.last_applied_target(prefix).tolist(),
            gain_switch={k: v for k, v in switch.items() if k not in ("before", "after")},
            measured_history=receipt,
            body_ball_clock_unchanged=True,
            physics_success_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
    except BaseException:
        # Only a matching owner created by this exact attempted handoff can be
        # removed. Never steal or overwrite another player's live ownership.
        if world._private_policy_owners.get(prefix) == identities:
            del world._private_policy_owners[prefix]
        world.switch_pd_profile(prefix, old)
        raise
