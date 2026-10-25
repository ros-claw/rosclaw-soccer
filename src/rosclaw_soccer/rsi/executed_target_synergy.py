"""Bounded multiframe SIM target authority with truthful subsequent NN action history.

Unlike reference conditioning, this changes the executed target after the
original NN call. It is an explicit experimental motor controller, not a learned
policy, transport, torque bypass, or hardware authorization.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.motion_synergy_reference import MotionSynergyPlan
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer


@dataclass(frozen=True)
class ExecutedTargetSynergy:
    start_control: int
    basis_hg: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[float, ...], tuple[float, ...]]

    def offsets(self) -> np.ndarray:
        # Reuse the qualified C1 temporal basis, in HG order this time. This
        # mathematical helper does NOT install/change the policy reference.
        return MotionSynergyPlan(self.start_control, self.basis_hg, self.coefficients).offsets()[0]


def apply_executed_target_synergy(
    player: SharedRoboNaldoPlayer, plan: ExecutedTargetSynergy
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode a pending target and its next-history action together, atomically.

    Physical joint limits are checked only for changed targets; zero is the
    exact original floating path, never a re-encoding of the frozen prior.
    An empty feasible intersection falls back for this entire control tick.
    This bounds target offsets, NOT actual torque or torque slew.
    """
    world, policy = player.world, player.policy
    world._require_current_boundary()
    if (
        type(plan) is not ExecutedTargetSynergy
        or player.failed
        or player.pending is None
        or player.boundary != world.control_steps
        or policy.runtime_mode != "sim"
        or policy.use_body_frame_ball
        or policy.time_step <= policy.WARMUP_STEPS
        or getattr(player, "_target_edit_boundary", None) == world.control_steps
    ):
        raise ValueError("explicit SIM plan at one admitted pending NN boundary required")
    offsets = plan.offsets()
    origin = getattr(player, "_forecast_control_origin", 0)
    if type(origin) is not int or origin < 0:
        raise ValueError("explicit nonnegative copied-world control origin required")
    control = origin + world.control_steps
    local = control - plan.start_control
    delta = offsets[local] if 0 <= local < len(offsets) else np.zeros(29)
    nominal = player.pending.copy()
    receipt = dict(
        schema="soccer.executed_target_synergy.v1",
        control_step=control,
        owned_control_step=world.control_steps,
        plan_start_control=plan.start_control,
        requested_offset_hg_rad=delta.tolist(),
        applied=False,
        fallback=False,
        physical_state_modified=False,
        reference_modified=False,
        neural_weights_modified=False,
        hardware_authorized=False,
    )
    # A failed encoder check cannot be retried as another edit on this proposal.
    # This is software state on the adapter, not on the copied pure policy.
    player._target_edit_boundary = world.control_steps
    if not np.any(delta):
        return nominal, dict(receipt, zero_branch_exact=True)
    mapping = np.asarray(import_module(type(policy).__module__).MUJOCO_TO_ISAAC)
    if mapping.shape != (29,) or sorted(mapping.tolist()) != list(range(29)):
        raise ValueError("qualified HG-to-Isaac action bijection required")
    raw = policy.last_action_il
    default, scale = policy.default_q_mj, policy.action_scale_mj
    if (
        any(
            a.shape != (29,) or a.dtype != np.float32 or not np.isfinite(a).all()
            for a in (raw, default, scale)
        )
        or (scale <= 0).any()
    ):
        raise ValueError("finite original float32 action encoder with positive scale required")
    decoded = (default + scale * raw[mapping]).astype(np.float64)
    if not np.array_equal(decoded, nominal) or not np.array_equal(
        policy.policy_output.actions, nominal
    ):
        raise ValueError("pending original target must exactly decode from original NN history")
    limits = world.model.jnt_range[player.layout.joints]
    action_lo, action_hi = policy.action_clip_lo_il, policy.action_clip_hi_il
    if (
        limits.shape != (29, 2)
        or action_lo.shape != (29,)
        or action_hi.shape != (29,)
        or not np.isfinite(limits).all()
        or not np.isfinite(action_lo).all()
        or not np.isfinite(action_hi).all()
        or (action_lo > action_hi).any()
    ):
        raise ValueError("finite original action and actual joint bounds required")
    changed = delta != 0
    lo = np.maximum(action_lo[mapping], (limits[:, 0] - default) / scale)
    hi = np.minimum(action_hi[mapping], (limits[:, 1] - default) / scale)
    if (lo[changed] > hi[changed]).any():
        return nominal, dict(receipt, fallback=True, reason="EMPTY_ENCODER_JOINT_INTERSECTION")
    # Round bounds inwards in the actual codec dtype, never round outside them.
    low, high = lo.astype(np.float32), hi.astype(np.float32)
    low = np.where(low < lo, np.nextafter(low, np.float32(np.inf)), low)
    high = np.where(high > hi, np.nextafter(high, np.float32(-np.inf)), high)
    if (low[changed] > high[changed]).any():
        return nominal, dict(receipt, fallback=True, reason="NO_FLOAT32_FEASIBLE_ACTION")
    desired = raw[mapping].copy()
    desired[changed] = np.clip(
        (raw[mapping][changed].astype(np.float64) + delta[changed] / scale[changed]).astype(
            np.float32
        ),
        low[changed],
        high[changed],
    )
    target = (default + scale * desired).astype(np.float64)
    # Codec rounding is measured explicitly; original trust radius is 0.06rad,
    # with at most 1e-6rad declared numerical tolerance, not a larger search.
    if (
        not np.isfinite(target).all()
        or (abs(target - nominal) > 0.060001).any()
        or (target[changed] < limits[changed, 0]).any()
        or (target[changed] > limits[changed, 1]).any()
    ):
        return nominal, dict(receipt, fallback=True, reason="ENCODED_TARGET_OUTSIDE_TRUST_OR_JOINT")
    if np.array_equal(target, nominal):
        return nominal, dict(receipt, zero_branch_exact=True)
    new_raw = raw.copy()
    new_raw[mapping] = desired
    # All validation precedes the joint pending-target/history commitment.
    policy.last_action_il = new_raw
    policy.policy_output.actions = target.astype(np.float32)
    player.pending = target.copy()
    return target, dict(
        receipt,
        applied=True,
        applied_offset_hg_rad=(target - nominal).tolist(),
        maximum_target_offset_rad=float(abs(target - nominal).max()),
        next_action_history_matches_executed_codec=True,
    )
