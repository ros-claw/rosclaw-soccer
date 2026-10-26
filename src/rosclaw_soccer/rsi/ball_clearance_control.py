"""A finite SIM contact-aware teacher hypothesis, not a learned motor policy.

Measured lower-leg/ball signed-distance gradients request small HG leg offsets.
Only the existing exact pending-target/action-history codec changes motor goals.
Scratch kinematics never reset or advance the actual body. A physical forecast
must pass before this hypothesis can be selected by a teacher.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.executed_target_synergy import _encode_pending_offset
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer


@dataclass(frozen=True)
class BallClearancePlan:
    activation_m: float = 0.08
    anticipatory_velocity: bool = False

    def validate(self) -> None:
        if type(self.activation_m) is not float or self.activation_m != 0.08:
            raise ValueError("one preregistered 8cm clearance-feedback hypothesis required")
        if type(self.anticipatory_velocity) is not bool:
            raise ValueError("explicit measured-velocity anticipation switch required")


def clearance_offset(
    distances: np.ndarray, gradients: np.ndarray, previous: np.ndarray
) -> np.ndarray:
    if (
        any(
            type(a) is not np.ndarray
            or a.shape != shape
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            for a, shape in ((distances, (2,)), (gradients, (2, 12)), (previous, (29,)))
        )
        or (abs(previous) > 0.060001).any()
        or np.any(previous[12:])
    ):
        raise ValueError("finite two-leg measured geometry and bounded prior offset required")
    desired = np.zeros(29)
    for distance, gradient in zip(distances, gradients, strict=True):
        norm = float(abs(gradient).max())
        if distance < 0.08 and norm > 1e-8:
            weight = float(np.clip((0.08 - distance) / 0.08, 0.0, 1.0))
            desired[:12] += 0.06 * weight * gradient / norm
    desired = np.clip(desired, -0.06, 0.06)
    desired *= min(1.0, 0.1 / max(float(np.linalg.norm(desired)), 1e-20))
    # Explicit smooth request, NOT a bound on actual joint/torque slew.
    result: np.ndarray = previous + np.clip(desired - previous, -0.008, 0.008)
    return result


class BallClearanceController:
    def __init__(self, player: SharedRoboNaldoPlayer, plan: BallClearancePlan) -> None:
        import mujoco

        if type(plan) is not BallClearancePlan:
            raise ValueError("explicit immutable clearance plan required")
        plan.validate()
        self.player, self.plan = player, plan
        self.scratch = mujoco.MjData(player.world.model)
        model = player.world.model
        ball_geoms = np.flatnonzero(model.geom_bodyid == player.world.ball)
        if len(ball_geoms) != 1:
            raise ValueError("one qualified sphere ball geometry required")
        self.ball = int(ball_geoms[0])
        self.groups = []
        for side in ("left", "right"):
            body = model.body(player.prefix + side + "_knee_link").id
            geoms = np.flatnonzero(model.geom_bodyid == body)
            if not len(geoms):
                raise ValueError("both named lower-leg collision groups required")
            self.groups.append(tuple(int(g) for g in geoms))
        self.previous = np.zeros(29)
        self.calls = 0

    def apply(self) -> tuple[np.ndarray, dict[str, Any]]:
        import mujoco

        player = self.player
        world, model = player.world, player.world.model
        world._require_current_boundary()
        if player.pending is None:
            raise ValueError("one pending NN proposal inside bounded128 clearance episode required")
        if self.calls >= 128:
            return player.pending.copy(), dict(applied=False, inactive_after_bounded128=True)
        qpos = world.data.qpos.copy()
        projected = qpos.copy()
        if self.plan.anticipatory_velocity:
            # 8 periods are the existing .06/.008 target-request ramp rounded
            # up, NOT a searched prediction horizon. Kinematic extrapolation
            # is a risk cue, never a dynamic rollout or a forced actual state.
            mujoco.mj_integratePos(model, projected, world.data.qvel.copy(), 0.16)
        self.scratch.qpos[:] = qpos
        mujoco.mj_kinematics(model, self.scratch)
        distances, gradients = np.ones(2), np.zeros((2, 12))
        risk_distances = np.ones(2)
        selected = []
        for side, group in enumerate(self.groups):
            geom = min(
                group,
                key=lambda g: mujoco.mj_geomDistance(model, self.scratch, self.ball, g, 1.0, None),
            )
            distance = mujoco.mj_geomDistance(model, self.scratch, self.ball, geom, 1.0, None)
            distances[side] = distance
            linearization = qpos
            if self.plan.anticipatory_velocity:
                self.scratch.qpos[:] = projected
                mujoco.mj_kinematics(model, self.scratch)
                ahead_geom = min(
                    group,
                    key=lambda g: mujoco.mj_geomDistance(
                        model, self.scratch, self.ball, g, 1.0, None
                    ),
                )
                ahead = mujoco.mj_geomDistance(
                    model, self.scratch, self.ball, ahead_geom, 1.0, None
                )
                if ahead < distance:
                    distance, geom, linearization = ahead, ahead_geom, projected
                self.scratch.qpos[:] = qpos
                mujoco.mj_kinematics(model, self.scratch)
            risk_distances[side] = distance
            selected.append(geom)
            if distance >= self.plan.activation_m:
                continue
            for axis, address in enumerate(player.layout.qi[:12]):
                values = []
                for sign in (-1, 1):
                    self.scratch.qpos[:] = linearization
                    self.scratch.qpos[address] += sign * 1e-4
                    mujoco.mj_kinematics(model, self.scratch)
                    values.append(
                        mujoco.mj_geomDistance(model, self.scratch, self.ball, geom, 1.0, None)
                    )
                gradients[side, axis] = (values[1] - values[0]) / 2e-4
            # Do not let one side's finite difference contaminate the next side.
            self.scratch.qpos[:] = qpos
            mujoco.mj_kinematics(model, self.scratch)
        delta = clearance_offset(risk_distances, gradients, self.previous)
        if self.calls >= 120:
            delta = self.previous + np.clip(-self.previous, -0.008, 0.008)
        receipt = dict(
            schema="soccer.measured_ball_clearance_target.v1",
            control_step=world.control_steps,
            distances_m=distances.tolist(),
            decision_distances_m=risk_distances.tolist(),
            anticipatory_velocity=self.plan.anticipatory_velocity,
            kinematic_extrapolation_sec=0.16 if self.plan.anticipatory_velocity else 0.0,
            extrapolation_is_not_physical_prediction=True,
            gradient_hg12=gradients.tolist(),
            selected_geometry=selected,
            previous_requested_offset=self.previous.tolist(),
            requested_offset_hg_rad=delta.tolist(),
            projection_mode="per_axis",
            original_nn_target_hg_rad=player.pending.tolist(),
            original_raw_action_il=player.policy.last_action_il.tolist(),
            applied=False,
            fallback=False,
            physical_state_modified=False,
            reference_modified=False,
            neural_weights_modified=False,
            hardware_authorized=False,
        )
        result = _encode_pending_offset(player, delta, receipt, projection_mode="per_axis")
        self.previous = delta.copy()
        self.calls += 1
        return result
