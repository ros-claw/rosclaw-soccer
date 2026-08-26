"""Whole-body goalkeeper combat adapter over the stable locomotion prior.

The learnable policy does not emit arbitrary leg targets.  Its first channel
gates a pinned, human-motion-constrained 29-DoF goalkeeper teacher; the other
channels retain ROSClaw's bounded waist/arm residual.  A zero action is exactly
the frozen stable prior, making safety comparison and rollback unambiguous.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rosclaw_soccer.training.goalkeeper_combat_teacher import (
    OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE,
    OFFICIAL_GOALKEEPER_DEFAULT_QPOS,
    load_official_goalkeeper_teacher,
)
from rosclaw_soccer.training.goalkeeper_mjwarp import (
    GoalkeeperMJWarpBatch,
    GoalkeeperMJWarpConfig,
)
from rosclaw_soccer.training.goalkeeper_mobility_option import (
    MOBILE_UPPER_BODY_KD,
    MOBILE_UPPER_BODY_KP,
    GoalkeeperMobilityOptionConfig,
)

COMBAT_SIGNED_LATERAL_GATE_LIMIT = 0.35
COMBAT_RECOVERY_LATERAL_GATE_LIMIT = 0.25
COMBAT_WAIST_RESIDUAL_LIMIT = 0.20
COMBAT_ARM_RESIDUAL_LIMIT = 0.25
COMBAT_SECOND_SHOT_ARM_RESIDUAL_LIMIT = 0.30
COMBAT_RECOVERY_CAPTURE_HORIZON_SEC = 0.35
COMBAT_RECOVERY_CENTER_DEADBAND_M = 0.025


class GoalkeeperCombatMJWarpBatch(GoalkeeperMJWarpBatch):
    """Blend a frozen athlete base with a position-conditioned motion teacher."""

    lower_body_authority = "BOUNDED_FROZEN_GOALKEEPER_TEACHER_BLEND"
    learned_residual_authority = (
        "SIGNED_STABLE_LATERAL_TEACHER_GATE_AND_BOUNDED_UPPER_BODY_RESIDUAL"
    )

    def __init__(
        self,
        *,
        asset_root: Path,
        locomotion_policy_path: Path,
        teacher_checkout: Path,
        teacher_checkpoint: Path,
        device: Any,
        config: GoalkeeperMJWarpConfig,
        maximum_teacher_blend: float = 0.35,
        intercept_conditioning_enabled: bool = False,
        mobility_option_enabled: bool = False,
        mobility_option_config: GoalkeeperMobilityOptionConfig | None = None,
        runtime_reach_atlas: Any | None = None,
        runtime_reach_blend: float = 0.0,
    ) -> None:
        maximum_allowed_blend = 1.0 if mobility_option_enabled else 0.50
        if (
            not isinstance(intercept_conditioning_enabled, bool)
            or not math.isfinite(maximum_teacher_blend)
            or not 0.05 <= maximum_teacher_blend <= maximum_allowed_blend
        ):
            raise ValueError(
                f"goalkeeper combat teacher blend must be in [0.05, {maximum_allowed_blend:.2f}]"
            )
        super().__init__(
            asset_root=asset_root,
            locomotion_policy_path=locomotion_policy_path,
            device=device,
            config=config,
        )
        torch = self.torch
        self.mobility_option_enabled = mobility_option_enabled
        self.mobility_option_config = mobility_option_config or GoalkeeperMobilityOptionConfig()
        self.teacher, self.teacher_report = load_official_goalkeeper_teacher(
            checkout=teacher_checkout,
            checkpoint=teacher_checkpoint,
            device=self.device,
        )
        self.maximum_teacher_blend = maximum_teacher_blend
        self.intercept_conditioning_enabled = intercept_conditioning_enabled
        if runtime_reach_atlas is None:
            if runtime_reach_blend != 0.0:
                raise ValueError("goalkeeper runtime reach blend requires an atlas")
        elif (
            not self.mobility_option_enabled
            or not math.isfinite(runtime_reach_blend)
            or not 0.05 <= runtime_reach_blend <= 0.85
            or bool(getattr(runtime_reach_atlas, "physics_authority", True))
            or getattr(runtime_reach_atlas, "activation_ceiling", None) != "SIM_ONLY"
        ):
            raise ValueError("goalkeeper runtime reach atlas boundary is invalid")
        self.runtime_reach_atlas = runtime_reach_atlas
        self.runtime_reach_blend = runtime_reach_blend
        self._teacher_default = torch.tensor(
            OFFICIAL_GOALKEEPER_DEFAULT_QPOS,
            dtype=torch.float32,
            device=self.device,
        )
        # The official teacher was trained on a 10-frame causal history.
        self._teacher_history = torch.zeros((self.count, 10, 96), device=self.device)
        self._previous_teacher_action = torch.zeros((self.count, 29), device=self.device)
        teacher_group_scale = (
            self.mobility_option_config.teacher_group_scale
            if self.mobility_option_enabled
            else OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE
        )
        self._teacher_group_scale = torch.tensor(
            teacher_group_scale,
            dtype=torch.float32,
            device=self.device,
        )
        self._mobility_teacher_gate = torch.zeros(self.count, device=self.device)
        self._teacher_recovery_gate = torch.zeros(self.count, device=self.device)
        self._teacher_recovery_age_steps = torch.full(
            (self.count,), -1, dtype=torch.long, device=self.device
        )
        self._previous_teacher_shot_active = torch.zeros(
            self.count, dtype=torch.bool, device=self.device
        )
        self._teacher_recovery_active = torch.zeros(
            self.count, dtype=torch.bool, device=self.device
        )
        if self.mobility_option_enabled:
            self._kp[12:] = torch.tensor(
                MOBILE_UPPER_BODY_KP,
                dtype=torch.float32,
                device=self.device,
            )
            self._kd[12:] = torch.tensor(
                MOBILE_UPPER_BODY_KD,
                dtype=torch.float32,
                device=self.device,
            )
        self._torso_body = int(self.cpu_model.body("torso_link").id)
        self._maximum_applied_teacher_blend = torch.zeros(self.count, device=self.device)
        self._maximum_applied_runtime_reach_blend = torch.zeros(self.count, device=self.device)
        self._previous_teacher_target_delta = torch.zeros((self.count, 29), device=self.device)
        self._maximum_applied_teacher_target_step = torch.zeros(
            (self.count, 3), device=self.device
        )
        self._teacher_active_steps = torch.zeros(self.count, device=self.device)
        self._runtime_reach_ready = torch.zeros(29, device=self.device)
        self._runtime_reach_ready[self._loco_to_motor] = self._loco_default
        self._runtime_reach_limits = (
            None
            if runtime_reach_atlas is None
            else torch.tensor(
                tuple(runtime_reach_atlas.effective_arm_limits_rad) * 2,
                dtype=torch.float32,
                device=self.device,
            )
        )

    def reset(self, *, seed: int) -> Any:
        self._teacher_history.zero_()
        self._previous_teacher_action.zero_()
        self._mobility_teacher_gate.zero_()
        self._teacher_recovery_gate.zero_()
        self._teacher_recovery_age_steps.fill_(-1)
        self._previous_teacher_shot_active.zero_()
        self._teacher_recovery_active.zero_()
        self._maximum_applied_teacher_blend.zero_()
        self._maximum_applied_runtime_reach_blend.zero_()
        self._previous_teacher_target_delta.zero_()
        self._maximum_applied_teacher_target_step.zero_()
        self._teacher_active_steps.zero_()
        return super().reset(seed=seed)

    def _shape_actor_action(self, requested_action: Any) -> tuple[Any, Any]:
        """Gate plastic authority by shot/recovery phase and angular stability.

        The upper body normally remains unavailable whenever no ball is in
        flight.  With a visible causal intent cue, bounded arm residuals may
        pre-shape before launch.  A separately bound predictive-teacher switch
        may also warm the frozen option from that same cue; otherwise the
        waist and whole-body teacher remain locked.
        A bounded lateral channel stays available after the first release so
        the learned policy can reposition for a second save instead of merely
        waiting for the previous command to decay.  Its sign is projected only
        at the goal boundary; no fixed-centre command is imposed.
        """

        torch = self.torch
        mobility_enabled = bool(getattr(self, "mobility_option_enabled", False))
        mobility = getattr(
            self,
            "mobility_option_config",
            GoalkeeperMobilityOptionConfig(),
        )
        angular_speed = torch.linalg.vector_norm(self.qvel[:, 3:6], dim=1)
        onset = self.config.agility.angular_guard_onset_rad_s
        ceiling = self.config.agility.angular_guard_ceiling_rad_s
        authority = torch.clamp(
            (ceiling - angular_speed) / max(ceiling - onset, 1.0e-6),
            min=self.config.agility.minimum_upper_body_scale,
            max=1.0,
        )
        shaped = requested_action * authority.unsqueeze(1)
        if mobility_enabled:
            # Stability-plasticity staging: first learn stable footwork and the
            # frozen whole-body option gate.  Residual waist/arm authority is
            # explicitly and reproducibly thawed by later checkpoint stages.
            shaped[:, 2:4] *= mobility.effective_waist_plasticity_scale
            shaped[:, 4:] *= mobility.effective_arm_plasticity_scale
            if mobility.counter_rotation_enabled:
                guard_fraction = torch.clamp(
                    (angular_speed - onset) / max(ceiling - onset, 1.0e-6),
                    min=0.0,
                    max=1.0,
                )
                gain = self.config.agility.counter_rotation_gain
                shaped[:, 2] -= gain * guard_fraction * self.qvel[:, 3]
                shaped[:, 3] -= gain * guard_fraction * self.qvel[:, 4]
        lateral_limit = (
            mobility.lateral_command_limit if mobility_enabled else COMBAT_SIGNED_LATERAL_GATE_LIMIT
        )
        waist_limit = (
            mobility.waist_residual_limit if mobility_enabled else COMBAT_WAIST_RESIDUAL_LIMIT
        )
        shaped[:, 0] = torch.clamp(shaped[:, 0], -lateral_limit, lateral_limit)
        waist_start = 2 if mobility_enabled else 1
        shaped[:, waist_start:4] = torch.clamp(
            shaped[:, waist_start:4],
            -waist_limit,
            waist_limit,
        )
        arm_limit = torch.where(
            (self._shot_index == 2).unsqueeze(1),
            torch.full_like(
                shaped[:, 4:],
                mobility.second_arm_residual_limit
                if mobility_enabled
                else COMBAT_SECOND_SHOT_ARM_RESIDUAL_LIMIT,
            ),
            torch.full_like(
                shaped[:, 4:],
                mobility.first_arm_residual_limit
                if mobility_enabled
                else COMBAT_ARM_RESIDUAL_LIMIT,
            ),
        )
        shaped[:, 4:] = torch.clamp(shaped[:, 4:], -arm_limit, arm_limit)
        shot_active = self._shot_index > 0
        first_release_step = int(
            round(self.config.first_shot_release_sec / self.config.control_dt_sec)
        )
        recovery_active = (~shot_active) & (self._step_index >= first_release_step)
        anticipation_active = (
            (~shot_active)
            & (self._step_index < first_release_step)
            & self.config.shot_intent_cue_enabled
        )
        predictive_ready = torch.zeros_like(shot_active)
        if self.config.shot_intent_cue_enabled and mobility.anticipatory_arm_reach_enabled:
            first_end_step = int(round(self.config.first_shot_end_sec / self.config.control_dt_sec))
            second_release_step = int(
                round(self.config.second_shot_release_sec / self.config.control_dt_sec)
            )
            cue_visible = torch.zeros_like(shot_active)
            if self._step_index < first_release_step:
                cue_visible = self._intent_cue_one[:, 2] > 0.5
            elif first_end_step <= self._step_index < second_release_step:
                cue_visible = self._intent_cue_two[:, 2] > 0.5
            predictive_ready = (~shot_active) & cue_visible
        if mobility_enabled:
            previous_shot_active = getattr(
                self,
                "_previous_teacher_shot_active",
                torch.zeros_like(shot_active),
            )
            recovery_age = getattr(
                self,
                "_teacher_recovery_age_steps",
                torch.full_like(self._shot_index, -1),
            )
            recovery_gate = getattr(
                self,
                "_teacher_recovery_gate",
                torch.zeros_like(shaped[:, 1]),
            )
            ended_shot = previous_shot_active & ~shot_active
            if mobility.teacher_recovery_latch_enabled:
                recovery_age[ended_shot] = 0
                recovery_gate[ended_shot] = self._mobility_teacher_gate[ended_shot]
            hold_steps = int(round(mobility.teacher_recovery_hold_sec / self.config.control_dt_sec))
            decay_steps = int(
                round(mobility.teacher_recovery_decay_sec / self.config.control_dt_sec)
            )
            total_recovery_steps = hold_steps + decay_steps
            recovery_latch = (
                mobility.teacher_recovery_latch_enabled
                & (recovery_age >= 0)
                & (recovery_age < total_recovery_steps)
                & ~shot_active
                & ~predictive_ready
            )
            decay_fraction = torch.clamp(
                (total_recovery_steps - recovery_age).to(torch.float32) / max(decay_steps, 1),
                min=0.0,
                max=1.0,
            )
            recovery_gate_request = recovery_gate * decay_fraction
            live_gate_request = torch.where(
                shot_active | (predictive_ready & mobility.predictive_teacher_warmstart_enabled),
                torch.clamp(shaped[:, 1], 0.0, 1.0),
                torch.zeros_like(shaped[:, 1]),
            )
            # Optional curriculum authority: hard, visible intercepts may force
            # a bounded frozen-teacher floor while the actor learns to request
            # the same option itself.  The floor is explicit in the signed
            # config/report and never bypasses target slew, joint, torque or
            # posture guards.  Zero preserves all historical checkpoints.
            teacher_floor_active = shot_active | (
                predictive_ready & mobility.predictive_teacher_warmstart_enabled
            )
            live_gate_request = torch.where(
                teacher_floor_active,
                torch.maximum(
                    live_gate_request,
                    torch.full_like(
                        live_gate_request,
                        mobility.predictive_teacher_gate_floor,
                    ),
                ),
                live_gate_request,
            )
            desired_teacher_gate = torch.where(
                recovery_latch,
                recovery_gate_request,
                live_gate_request,
            )
            gate_delta = torch.clamp(
                desired_teacher_gate - self._mobility_teacher_gate,
                -mobility.teacher_gate_step,
                mobility.teacher_gate_step,
            )
            self._mobility_teacher_gate += mobility.teacher_gate_filter_fraction * gate_delta
            recovery_age[recovery_latch] += 1
            reset_recovery = shot_active | predictive_ready | (recovery_age >= total_recovery_steps)
            recovery_age[reset_recovery] = -1
            recovery_gate[reset_recovery] = 0.0
            if hasattr(self, "_teacher_recovery_active"):
                self._teacher_recovery_active.copy_(recovery_latch)
                self._previous_teacher_shot_active.copy_(shot_active)
            shaped[:, 1] = 0.0
            shaped[:, 0] = torch.where(
                recovery_active & ~predictive_ready,
                torch.clamp(
                    shaped[:, 0],
                    -mobility.recovery_command_limit,
                    mobility.recovery_command_limit,
                ),
                shaped[:, 0],
            )
            capture = self.qpos[:, 1].clone()
            capture += mobility.capture_horizon_sec * self.qvel[:, 1]
            outward = (
                (torch.abs(capture) >= mobility.goal_boundary_m)
                & (capture * shaped[:, 0] < 0.0)
                & recovery_active
            )
            shaped[:, 0] = torch.where(
                outward,
                torch.sign(capture) * torch.abs(shaped[:, 0]),
                shaped[:, 0],
            )
            if mobility.lateral_velocity_guard_enabled:
                lateral_speed = torch.abs(self.qvel[:, 1])
                continuing = self.qvel[:, 1] * shaped[:, 0] < 0.0
                span = (
                    mobility.lateral_velocity_guard_ceiling_mps
                    - mobility.lateral_velocity_guard_onset_mps
                )
                velocity_authority = torch.clamp(
                    (mobility.lateral_velocity_guard_ceiling_mps - lateral_speed) / span,
                    min=0.0,
                    max=1.0,
                )
                faded = shaped[:, 0] * velocity_authority
                brake = torch.sign(self.qvel[:, 1]) * torch.clamp(
                    torch.abs(shaped[:, 0]), min=0.20, max=0.35
                )
                guarded = torch.where(
                    lateral_speed >= mobility.lateral_velocity_guard_ceiling_mps,
                    brake,
                    faded,
                )
                shaped[:, 0] = torch.where(
                    continuing & (lateral_speed > mobility.lateral_velocity_guard_onset_mps),
                    guarded,
                    shaped[:, 0],
                )
        else:
            capture_error = self.qpos[:, 1].clone()
            capture_error += COMBAT_RECOVERY_CAPTURE_HORIZON_SEC * self.qvel[:, 1]
            recovery_direction = torch.sign(capture_error)
            recovery_direction = torch.where(
                torch.abs(capture_error) >= COMBAT_RECOVERY_CENTER_DEADBAND_M,
                recovery_direction,
                torch.zeros_like(recovery_direction),
            )
            recovery_gate = torch.clamp(
                torch.abs(shaped[:, 0]),
                0.0,
                COMBAT_RECOVERY_LATERAL_GATE_LIMIT,
            )
            shaped[:, 0] = torch.where(
                recovery_active,
                recovery_direction * recovery_gate,
                shaped[:, 0],
            )
        shaped[:, 0] = torch.where(
            shot_active | recovery_active | anticipation_active,
            shaped[:, 0],
            torch.zeros_like(shaped[:, 0]),
        )
        shaped[:, 1:4] = torch.where(
            shot_active.unsqueeze(1),
            shaped[:, 1:4],
            torch.zeros_like(shaped[:, 1:4]),
        )
        shaped[:, 4:] = torch.where(
            (shot_active | predictive_ready).unsqueeze(1),
            shaped[:, 4:],
            torch.zeros_like(shaped[:, 4:]),
        )
        return shaped, authority

    def _locomotion_target(self, signed_combat_action: Any) -> Any:
        torch = self.torch
        # The sign commands the qualified locomotion prior toward the intercept;
        # magnitude opens the bounded position-conditioned goalkeeper teacher.
        stable_target = super()._locomotion_target(signed_combat_action)
        quaternion = self.qpos[:, 3:7]
        torso = self.xpos[:, self._torso_body]
        ball_relative = self._rotate_inverse(quaternion, self.qpos[:, 36:39] - torso)
        visible = self._shot_index > 0
        predictive = torch.zeros_like(visible)
        if self.intercept_conditioning_enabled:
            # Upstream trains actor channel 0:3 on the predicted catch-plane
            # target (``end_target_local``), not the instantaneous ball pose.
            # ROSClaw's estimate uses only current position/velocity and is
            # therefore causal while restoring the teacher's exact semantics.
            intercept = self._causal_intercept()
            ball_relative = self._rotate_inverse(quaternion, intercept - torso)
        mobility = getattr(
            self,
            "mobility_option_config",
            GoalkeeperMobilityOptionConfig(),
        )
        if (
            bool(getattr(self, "mobility_option_enabled", False))
            and self.config.shot_intent_cue_enabled
            and mobility.predictive_teacher_warmstart_enabled
        ):
            first_release_step = int(
                round(self.config.first_shot_release_sec / self.config.control_dt_sec)
            )
            first_end_step = int(round(self.config.first_shot_end_sec / self.config.control_dt_sec))
            second_release_step = int(
                round(self.config.second_shot_release_sec / self.config.control_dt_sec)
            )
            cue = None
            if self._step_index < first_release_step:
                cue = self._intent_cue_one
            elif first_end_step <= self._step_index < second_release_step:
                cue = self._intent_cue_two
            if cue is not None:
                predictive = cue[:, 2] > 0.5
                proxy_x = (
                    self.config.keeper_x_m - 0.08
                    if self.intercept_conditioning_enabled
                    else self.config.keeper_x_m - 1.0
                )
                proxy_world = torch.stack(
                    (torch.full_like(cue[:, 0], proxy_x), cue[:, 0], cue[:, 1]), dim=1
                )
                proxy_relative = self._rotate_inverse(quaternion, proxy_world - torso)
                ball_relative = torch.where(predictive.unsqueeze(1), proxy_relative, ball_relative)
                visible |= predictive
        recovery_visible = getattr(
            self,
            "_teacher_recovery_active",
            torch.zeros_like(visible),
        )
        visible |= recovery_visible
        if self.intercept_conditioning_enabled:
            # Upstream explicitly masks the target after the flight.  Feeding
            # ROSClaw's parked ball (-20 m) during a recovery latch previously
            # produced an out-of-distribution command and violent falls.
            teacher_target_visible = (self._shot_index > 0) | predictive
            ball_relative = torch.where(
                teacher_target_visible.unsqueeze(1),
                ball_relative,
                torch.zeros_like(ball_relative),
            )
        else:
            ball_relative = torch.where(
                visible.unsqueeze(1), ball_relative, torch.zeros_like(ball_relative)
            )
        angular_velocity = self._rotate_inverse(quaternion, self.qvel[:, 3:6]) * 0.25
        qw, qx, qy, qz = (quaternion[:, index] for index in range(4))
        gravity = torch.stack(
            (
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ),
            dim=1,
        )
        observation = torch.cat(
            (
                ball_relative,
                angular_velocity,
                gravity,
                self.qpos[:, 7:36] - self._teacher_default,
                self.qvel[:, 6:35] * 0.05,
                self._previous_teacher_action,
            ),
            dim=1,
        )
        if tuple(observation.shape) != (self.count, 96):
            raise RuntimeError("goalkeeper combat teacher observation contract changed")
        self._teacher_history[:, :-1].copy_(self._teacher_history[:, 1:].clone())
        self._teacher_history[:, -1].copy_(torch.clamp(observation, -100.0, 100.0))
        with torch.inference_mode():
            teacher_action = self.teacher(self._teacher_history.flatten(1))
        teacher_action = torch.clamp(teacher_action, -4.0, 4.0)
        self._previous_teacher_action.copy_(teacher_action)
        teacher_target = self._teacher_default + 0.25 * teacher_action

        blend = (
            self._mobility_teacher_gate
            if self.mobility_option_enabled
            else torch.clamp(torch.abs(signed_combat_action), 0.0, 1.0)
        )
        # Never scale the stateful mobility gate in place.  ``blend`` aliases
        # ``self._mobility_teacher_gate`` on this path; the old ``*=`` wrote
        # the external blend ceiling back into the controller state every
        # step, creating a hidden fixed point and making stronger predictive
        # requests ineffective.
        blend = blend * self.maximum_teacher_blend
        blend = torch.where(visible, blend, torch.zeros_like(blend))
        maximum_joint_blend = 1.0 if self.mobility_option_enabled else 0.50
        joint_blend = torch.clamp(
            blend.unsqueeze(1) * self._teacher_group_scale.unsqueeze(0),
            0.0,
            maximum_joint_blend,
        )
        target = stable_target + joint_blend * (teacher_target - stable_target)
        if self.runtime_reach_atlas is not None:
            from rosclaw_soccer.training.goalkeeper_reach import (
                task_space_reach_from_target_torch,
            )

            normalized_reach = task_space_reach_from_target_torch(
                torch=torch,
                target_relative=ball_relative,
                model=self.runtime_reach_atlas,
            )
            reach_target = self._runtime_reach_ready[15:29].unsqueeze(0)
            reach_target = reach_target + normalized_reach * self._runtime_reach_limits
            reach_visible = (self._shot_index > 0) | predictive
            reach_blend = self.runtime_reach_blend * self._mobility_teacher_gate
            reach_blend = torch.where(
                reach_visible,
                reach_blend,
                torch.zeros_like(reach_blend),
            )
            target[:, 15:29] += reach_blend.unsqueeze(1) * (reach_target - target[:, 15:29])
            self._maximum_applied_runtime_reach_blend.copy_(
                torch.maximum(
                    self._maximum_applied_runtime_reach_blend,
                    reach_blend,
                )
            )
        raw_teacher_delta = target - stable_target
        filtered_groups: list[Any] = []
        group_contracts = mobility.teacher_target_filter_contracts
        for group_index, (start, end, step_limit, filter_fraction) in enumerate(
            group_contracts
        ):
            previous = self._previous_teacher_target_delta[:, start:end]
            delta_step = torch.clamp(
                raw_teacher_delta[:, start:end] - previous,
                -step_limit,
                step_limit,
            )
            filtered = previous + filter_fraction * delta_step
            applied_step = torch.max(torch.abs(filtered - previous), dim=1).values
            self._maximum_applied_teacher_target_step[:, group_index].copy_(
                torch.maximum(
                    self._maximum_applied_teacher_target_step[:, group_index],
                    applied_step,
                )
            )
            filtered_groups.append(filtered)
        filtered_teacher_delta = torch.cat(filtered_groups, dim=1)
        target = stable_target + filtered_teacher_delta
        self._previous_teacher_target_delta.copy_(filtered_teacher_delta)
        target = torch.clamp(target, self._joint_ranges[:, 0], self._joint_ranges[:, 1])
        self._maximum_applied_teacher_blend.copy_(
            torch.maximum(self._maximum_applied_teacher_blend, torch.max(joint_blend, dim=1).values)
        )
        self._teacher_active_steps += (blend > 1.0e-4).to(torch.float32)
        return target

    def _rotate_inverse(self, quaternion: Any, vector: Any) -> Any:
        """Rotate world vectors into the MuJoCo free-joint body frame (wxyz)."""

        torch = self.torch
        qw, qx, qy, qz = (quaternion[:, index] for index in range(4))
        vx, vy, vz = (vector[:, index] for index in range(3))
        return torch.stack(
            (
                (1.0 - 2.0 * (qy * qy + qz * qz)) * vx
                + 2.0 * (qx * qy + qz * qw) * vy
                + 2.0 * (qx * qz - qy * qw) * vz,
                2.0 * (qx * qy - qz * qw) * vx
                + (1.0 - 2.0 * (qx * qx + qz * qz)) * vy
                + 2.0 * (qy * qz + qx * qw) * vz,
                2.0 * (qx * qz + qy * qw) * vx
                + 2.0 * (qy * qz - qx * qw) * vy
                + (1.0 - 2.0 * (qx * qx + qy * qy)) * vz,
            ),
            dim=1,
        )

    def summary(self) -> dict[str, Any]:
        report = super().summary()
        report.update(
            {
                "schema_version": "rosclaw_soccer.goalkeeper_combat_mjwarp_summary.v4",
                "external_teacher": self.teacher_report,
                "maximum_declared_teacher_blend": self.maximum_teacher_blend,
                "teacher_conditioning": (
                    "CAUSAL_CATCH_PLANE_TARGET"
                    if self.intercept_conditioning_enabled
                    else "LEGACY_INSTANTANEOUS_BALL_POSITION"
                ),
                "mobility_option_enabled": self.mobility_option_enabled,
                "mobility_option": (
                    {
                        "config_hash": self.mobility_option_config.config_hash,
                        "separate_teacher_gate_channel": 1,
                        "fixed_center_recovery": False,
                        "residual_plasticity_scale": (
                            self.mobility_option_config.residual_plasticity_scale
                        ),
                        "effective_waist_plasticity_scale": (
                            self.mobility_option_config.effective_waist_plasticity_scale
                        ),
                        "effective_arm_plasticity_scale": (
                            self.mobility_option_config.effective_arm_plasticity_scale
                        ),
                        "counter_rotation_enabled": (
                            self.mobility_option_config.counter_rotation_enabled
                        ),
                        "teacher_recovery_latch_enabled": (
                            self.mobility_option_config.teacher_recovery_latch_enabled
                        ),
                    }
                    if self.mobility_option_enabled
                    else None
                ),
                "maximum_applied_teacher_blend": float(self._maximum_applied_teacher_blend.max()),
                "maximum_applied_teacher_target_step_rad": {
                    "lower_body": float(self._maximum_applied_teacher_target_step[:, 0].max()),
                    "waist": float(self._maximum_applied_teacher_target_step[:, 1].max()),
                    "arms": float(self._maximum_applied_teacher_target_step[:, 2].max()),
                },
                "runtime_task_space_reach": (
                    None
                    if self.runtime_reach_atlas is None
                    else {
                        "atlas_hash": self.runtime_reach_atlas.model_hash,
                        "maximum_declared_blend": self.runtime_reach_blend,
                        "maximum_applied_blend": float(
                            self._maximum_applied_runtime_reach_blend.max()
                        ),
                        "actor_gate_channel": 1,
                        "target_conditioning": "CAUSAL_INTERCEPT_RELATIVE_XYZ",
                    }
                ),
                "mean_teacher_active_fraction": float(
                    (self._teacher_active_steps / max(self._step_index, 1)).mean()
                ),
                "learned_action_envelope": {
                    "signed_lateral_teacher_gate": COMBAT_SIGNED_LATERAL_GATE_LIMIT,
                    "recovery_lateral_gate": COMBAT_RECOVERY_LATERAL_GATE_LIMIT,
                    "recovery_capture_horizon_sec": (COMBAT_RECOVERY_CAPTURE_HORIZON_SEC),
                    "recovery_center_deadband_m": (COMBAT_RECOVERY_CENTER_DEADBAND_M),
                    "waist_residual": COMBAT_WAIST_RESIDUAL_LIMIT,
                    "arm_residual": COMBAT_ARM_RESIDUAL_LIMIT,
                    "second_shot_arm_residual": (COMBAT_SECOND_SHOT_ARM_RESIDUAL_LIMIT),
                },
            }
        )
        return report


__all__ = [
    "COMBAT_ARM_RESIDUAL_LIMIT",
    "COMBAT_RECOVERY_CAPTURE_HORIZON_SEC",
    "COMBAT_RECOVERY_CENTER_DEADBAND_M",
    "COMBAT_RECOVERY_LATERAL_GATE_LIMIT",
    "COMBAT_SECOND_SHOT_ARM_RESIDUAL_LIMIT",
    "COMBAT_SIGNED_LATERAL_GATE_LIMIT",
    "COMBAT_WAIST_RESIDUAL_LIMIT",
    "GoalkeeperCombatMJWarpBatch",
]
