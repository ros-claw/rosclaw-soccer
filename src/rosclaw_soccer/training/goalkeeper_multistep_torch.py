"""Torch-native state machine for batched goalkeeper physics training.

This mirrors :mod:`goalkeeper_multistep` without moving simulator tensors to
the host.  Torch remains an optional training dependency and is imported only
when the accumulator is instantiated.
"""

from __future__ import annotations

from typing import Any

from rosclaw_soccer.training.goalkeeper_multistep import (
    GoalkeeperEpisodePhase,
    GoalkeeperMultiStepConfig,
)


class TorchGoalkeeperMultiStepAccumulator:
    """GPU-resident long-horizon reward and event accumulator."""

    def __init__(
        self,
        environment_count: int,
        *,
        device: Any,
        config: GoalkeeperMultiStepConfig | None = None,
        validate_each_step: bool = True,
    ) -> None:
        import torch

        if not 1 <= environment_count <= 262_144:
            raise ValueError("goalkeeper environment count is outside [1, 262144]")
        self.torch = torch
        self.environment_count = environment_count
        self.device = device
        self.config = config or GoalkeeperMultiStepConfig()
        self.validate_each_step = validate_each_step
        self.phase = torch.full(
            (environment_count,),
            int(GoalkeeperEpisodePhase.READY),
            dtype=torch.long,
            device=device,
        )
        self.first_contact = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self.first_save = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self.first_hand_save = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self.recovered_after_first = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self.second_contact = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self.second_attempt_save = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self.second_attempt_hand_save = torch.zeros(
            environment_count, dtype=torch.bool, device=device
        )
        self.second_save = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self.second_hand_save = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self._previous_contact = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self._previous_hand_contact = torch.zeros(
            environment_count, dtype=torch.bool, device=device
        )
        self._previous_save = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self._stable_steps = torch.zeros(environment_count, dtype=torch.long, device=device)
        self._bimanual_reach_steps = torch.zeros(environment_count, dtype=torch.long, device=device)
        self._active_flight_steps = torch.zeros(environment_count, dtype=torch.long, device=device)
        self._previous_hard_height_potential = torch.zeros(environment_count, device=device)
        self._previous_reach_potential = torch.zeros(environment_count, device=device)
        self._previous_bimanual_potential = torch.zeros(environment_count, device=device)
        self._previous_task_motion_potential = torch.zeros(environment_count, device=device)
        self._previous_task_motion_arrival_window = torch.zeros(
            environment_count, dtype=torch.bool, device=device
        )
        self._previous_recovery_potential = torch.zeros(environment_count, device=device)
        self._previous_shot = torch.zeros(environment_count, dtype=torch.long, device=device)

    def reset(self, environment_ids: Any | None = None) -> None:
        torch = self.torch
        ids = (
            torch.arange(self.environment_count, dtype=torch.long, device=self.device)
            if environment_ids is None
            else torch.as_tensor(environment_ids, dtype=torch.long, device=self.device)
        )
        if ids.ndim != 1 or bool(torch.any((ids < 0) | (ids >= self.environment_count))):
            raise ValueError("goalkeeper reset ids are out of range")
        self.phase[ids] = int(GoalkeeperEpisodePhase.READY)
        self.first_contact[ids] = False
        self.first_save[ids] = False
        self.first_hand_save[ids] = False
        self.recovered_after_first[ids] = False
        self.second_contact[ids] = False
        self.second_attempt_save[ids] = False
        self.second_attempt_hand_save[ids] = False
        self.second_save[ids] = False
        self.second_hand_save[ids] = False
        self._previous_contact[ids] = False
        self._previous_hand_contact[ids] = False
        self._previous_save[ids] = False
        self._stable_steps[ids] = 0
        self._bimanual_reach_steps[ids] = 0
        self._active_flight_steps[ids] = 0
        self._previous_hard_height_potential[ids] = 0.0
        self._previous_reach_potential[ids] = 0.0
        self._previous_bimanual_potential[ids] = 0.0
        self._previous_task_motion_potential[ids] = 0.0
        self._previous_task_motion_arrival_window[ids] = False
        self._previous_recovery_potential[ids] = 0.0
        self._previous_shot[ids] = 0

    def step(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Update state from simulator tensors and return decomposed rewards."""

        torch = self.torch
        # Full finite/shape validation synchronizes a CUDA stream.  Keep it on
        # by default for public callers and tests, while the physics trainer
        # validates its fixed tensor contract once and performs a finite-state
        # audit at every rollout boundary.
        if self.validate_each_step:
            self._validate(sample)
        cfg = self.config
        time = sample["time_sec"]
        contact = sample["ball_contact"]
        hand_contact = sample["hand_contact"]
        save = sample["true_save"]
        shot = sample["shot_index"]
        new_contact = contact & ~self._previous_contact
        new_hand_contact = hand_contact & ~self._previous_hand_contact
        new_save = save & ~self._previous_save
        posture_unsafe = (sample["pelvis_height_m"] < cfg.minimum_pelvis_height_m) | (
            sample["upright_projection"] < cfg.minimum_upright_projection
        )
        posture_exception = sample.get(
            "posture_exception_granted",
            torch.zeros(self.environment_count, dtype=torch.bool, device=self.phase.device),
        )
        unsafe = posture_unsafe & ~posture_exception
        terminal_phase = (self.phase == int(GoalkeeperEpisodePhase.COMPLETE)) | (
            self.phase == int(GoalkeeperEpisodePhase.FAILED)
        )
        active = ~terminal_phase
        self.phase[active & (shot == 1)] = int(GoalkeeperEpisodePhase.FIRST_FLIGHT)
        self.phase[active & (shot == 2)] = int(GoalkeeperEpisodePhase.SECOND_FLIGHT)

        first_contact_event = active & new_contact & (shot == 1)
        first_save_event = active & new_save & (shot == 1)
        first_hand_save_event = first_save_event & hand_contact
        self.first_contact |= first_contact_event
        self.first_save |= first_save_event
        self.first_hand_save |= first_hand_save_event
        self.phase[first_contact_event | first_save_event] = int(
            GoalkeeperEpisodePhase.FIRST_IMPACT
        )
        recovering = (
            active & self.first_save & ~self.recovered_after_first & ~first_save_event & (shot != 2)
        )
        root_speed = torch.linalg.vector_norm(sample["root_linear_velocity_mps"], dim=1)
        angular_speed = torch.linalg.vector_norm(sample["root_angular_velocity_rad_s"], dim=1)
        stable = (
            recovering
            & ~posture_unsafe
            & (root_speed <= cfg.maximum_recovered_linear_speed_mps)
            & (angular_speed <= cfg.maximum_recovered_angular_speed_rad_s)
        )
        self._stable_steps = torch.where(stable, self._stable_steps + 1, 0)
        just_recovered = (
            recovering
            & ~self.recovered_after_first
            & (self._stable_steps >= cfg.recovery_hold_steps)
        )
        self.recovered_after_first |= just_recovered
        self.phase[recovering & ~just_recovered] = int(GoalkeeperEpisodePhase.FIRST_RECOVERY)

        second_attempt_contact_event = active & new_contact & (shot == 2)
        second_attempt_save_event = active & new_save & (shot == 2)
        second_attempt_hand_save_event = second_attempt_save_event & hand_contact
        self.second_attempt_save |= second_attempt_save_event
        self.second_attempt_hand_save |= second_attempt_hand_save_event
        second_eligible = active & self.recovered_after_first & (shot == 2)
        second_contact_event = second_eligible & new_contact
        second_save_event = second_eligible & new_save
        second_hand_save_event = second_save_event & hand_contact
        self.second_contact |= second_contact_event
        self.second_save |= second_save_event
        self.second_hand_save |= second_hand_save_event
        self.phase[second_attempt_contact_event | second_attempt_save_event] = int(
            GoalkeeperEpisodePhase.SECOND_IMPACT
        )

        left_distance = torch.linalg.vector_norm(
            sample["left_hand_position_m"] - sample["intercept_target_m"], dim=1
        )
        right_distance = torch.linalg.vector_norm(
            sample["right_hand_position_m"] - sample["intercept_target_m"], dim=1
        )
        hand_distance = torch.minimum(left_distance, right_distance)
        phase_reach_multiplier = torch.where(
            shot == 2,
            torch.full_like(hand_distance, cfg.second_shot_reach_reward_multiplier),
            torch.ones_like(hand_distance),
        )
        reach_potential = (shot > 0).to(torch.float32) * torch.exp(
            -4.0 * torch.square(hand_distance)
        )
        continued_flight = (shot > 0) & (shot == self._previous_shot)
        reach_signal = (
            torch.where(
                continued_flight,
                reach_potential - self._previous_reach_potential,
                torch.zeros_like(reach_potential),
            )
            if cfg.reach_reward_semantics == "POTENTIAL_PROGRESS_ONLY"
            else reach_potential
        )
        reach = cfg.reach_reward_scale * phase_reach_multiplier * reach_signal
        target = sample["intercept_target_m"]
        hard_height = (shot > 0) & (target[:, 2] >= cfg.hard_height_reach_threshold_m)
        hard_potential = hard_height.to(torch.float32) * torch.exp(
            -cfg.hard_height_reach_distance_decay * torch.square(hand_distance)
        )
        continued_shot = hard_height & (shot == self._previous_shot)
        hard_progress = torch.where(
            continued_shot,
            hard_potential - self._previous_hard_height_potential,
            torch.zeros_like(hard_potential),
        )
        reach += cfg.hard_height_reach_reward_scale * phase_reach_multiplier * hard_progress
        hand_midpoint = 0.5 * (sample["left_hand_position_m"] + sample["right_hand_position_m"])
        bilateral_opportunity = (
            (shot > 0)
            & (torch.abs(target[:, 1] - hand_midpoint[:, 1]) <= 0.42)
            & (target[:, 2] >= 0.62)
        )
        bimanual_distance = torch.maximum(left_distance, right_distance)
        bimanual_potential = bilateral_opportunity.to(torch.float32) * torch.exp(
            -3.0 * torch.square(bimanual_distance)
        )
        bimanual_signal = (
            torch.where(
                continued_flight,
                bimanual_potential - self._previous_bimanual_potential,
                torch.zeros_like(bimanual_potential),
            )
            if cfg.reach_reward_semantics == "POTENTIAL_PROGRESS_ONLY"
            else bimanual_potential
        )
        bimanual_reach = cfg.bimanual_reach_reward_scale * phase_reach_multiplier * bimanual_signal
        pelvis_position = sample.get("pelvis_position_m")
        if cfg.task_motion_reward_scale > 0.0 and pelvis_position is None:
            raise ValueError("goalkeeper task-motion reward requires pelvis position")
        if pelvis_position is None:
            task_motion_potential = torch.zeros_like(hand_distance)
        else:
            lateral_remaining = torch.clamp(
                torch.abs(target[:, 1] - pelvis_position[:, 1])
                - cfg.task_motion_lateral_standoff_m,
                min=0.0,
            )
            pelvis_target_height = torch.clamp(
                target[:, 2] + cfg.task_motion_vertical_offset_m,
                min=cfg.task_motion_minimum_pelvis_target_m,
                max=cfg.task_motion_maximum_pelvis_target_m,
            )
            task_motion_error_sq = torch.square(lateral_remaining) + torch.square(
                pelvis_position[:, 2] - pelvis_target_height
            )
            task_motion_potential = (shot > 0).to(torch.float32) * torch.exp(
                -cfg.task_motion_distance_decay * task_motion_error_sq
            )
        task_motion_progress = torch.where(
            continued_flight,
            task_motion_potential - self._previous_task_motion_potential,
            torch.zeros_like(task_motion_potential),
        )
        forward_velocity = sample["ball_velocity_mps"][:, 0]
        valid_arrival_clock = (shot > 0) & (torch.abs(forward_velocity) >= 0.10)
        safe_forward_velocity = torch.where(
            valid_arrival_clock,
            forward_velocity,
            torch.ones_like(forward_velocity),
        )
        time_to_intercept = (target[:, 0] - sample["ball_position_m"][:, 0]) / safe_forward_velocity
        valid_arrival_clock &= time_to_intercept >= 0.0
        arrival_progress = torch.where(
            valid_arrival_clock,
            torch.clamp(
                1.0 - time_to_intercept / cfg.task_motion_approach_horizon_sec,
                min=0.0,
                max=1.0,
            ),
            torch.zeros_like(time_to_intercept),
        )
        progress_weight = (
            cfg.task_motion_minimum_progress_weight
            + (1.0 - cfg.task_motion_minimum_progress_weight) * arrival_progress
        )
        arrival_window = valid_arrival_clock & (
            time_to_intercept <= cfg.task_motion_arrival_horizon_sec
        )
        arrival_event = arrival_window & ~self._previous_task_motion_arrival_window
        arrival_coupling = torch.sqrt(
            torch.clamp(task_motion_potential * reach_potential, min=0.0, max=1.0)
        )
        qualified_arrival_coupling = torch.clamp(
            (arrival_coupling - cfg.task_motion_arrival_readiness_threshold)
            / (1.0 - cfg.task_motion_arrival_readiness_threshold),
            min=0.0,
            max=1.0,
        )
        task_motion = (
            cfg.task_motion_reward_scale
            * phase_reach_multiplier
            * (
                progress_weight * task_motion_progress
                + cfg.task_motion_arrival_bonus_scale
                * arrival_event.to(torch.float32)
                * qualified_arrival_coupling
            )
        )
        self._active_flight_steps += (shot > 0).to(torch.long)
        self._bimanual_reach_steps += (bilateral_opportunity & (bimanual_distance <= 0.42)).to(
            torch.long
        )
        upright = cfg.upright_reward_scale * torch.clamp(
            (sample["upright_projection"] - cfg.minimum_upright_projection)
            / (1.0 - cfg.minimum_upright_projection),
            0.0,
            1.0,
        )
        normalized_upright = torch.clamp(
            (sample["upright_projection"] - cfg.minimum_upright_projection)
            / (1.0 - cfg.minimum_upright_projection),
            0.0,
            1.0,
        )
        recovery_potential = normalized_upright * torch.exp(
            -cfg.recovery_progress_linear_speed_decay * root_speed.square()
            - cfg.recovery_progress_angular_speed_decay * angular_speed.square()
        )
        recovery_progress = cfg.recovery_progress_reward_scale * torch.where(
            recovering,
            recovery_potential - self._previous_recovery_potential,
            torch.zeros_like(recovery_potential),
        )
        action_rate = torch.mean(torch.square(sample["action"] - sample["previous_action"]), dim=1)
        acceleration = torch.mean(torch.square(sample["joint_acceleration_rad_s2"]), dim=1)
        torque = torch.mean(torch.square(sample["applied_torque_nm"]), dim=1)
        root_linear_speed = torch.sum(torch.square(sample["root_linear_velocity_mps"]), dim=1)
        root_angular_speed = torch.sum(torch.square(sample["root_angular_velocity_rad_s"]), dim=1)
        action_magnitude = torch.mean(torch.square(sample["action"]), dim=1)
        action_rate_penalty = cfg.action_rate_penalty_scale * action_rate
        joint_acceleration_penalty = cfg.joint_acceleration_penalty_scale * acceleration
        root_linear_speed_penalty = cfg.root_linear_speed_penalty_scale * root_linear_speed
        # Use the environment-owned, time-bounded dynamic-skill clock so the
        # goalkeeper may anticipate and land, while later recovery still pays
        # the full angular stability cost.
        angular_phase_scale = torch.where(
            posture_exception,
            torch.full_like(root_angular_speed, cfg.flight_root_angular_penalty_scale),
            torch.ones_like(root_angular_speed),
        )
        root_angular_speed_penalty = (
            angular_phase_scale * cfg.root_angular_speed_penalty_scale * root_angular_speed
        )
        root_angular_excess = torch.clamp(
            torch.sqrt(root_angular_speed) - cfg.root_angular_speed_soft_limit_rad_s,
            min=0.0,
        )
        root_angular_excess_penalty = (
            angular_phase_scale
            * cfg.root_angular_speed_excess_penalty_scale
            * root_angular_excess.square()
        )
        action_magnitude_penalty = cfg.action_magnitude_penalty_scale * action_magnitude
        smoothness_penalty = (
            action_rate_penalty
            + joint_acceleration_penalty
            + root_linear_speed_penalty
            + root_angular_speed_penalty
            + root_angular_excess_penalty
            + action_magnitude_penalty
        )
        effort_penalty = cfg.torque_penalty_scale * torque
        false_contact = (
            new_contact
            & ~save
            & (torch.linalg.vector_norm(sample["ball_velocity_mps"], dim=1) < 0.10)
        )
        event_bonus = cfg.contact_bonus * new_contact.to(torch.float32)
        event_bonus += cfg.hand_contact_bonus * new_hand_contact.to(torch.float32)
        event_bonus += cfg.true_save_bonus * first_save_event.to(torch.float32)
        event_bonus += cfg.hand_save_bonus * first_hand_save_event.to(torch.float32)
        event_bonus += cfg.true_save_bonus * second_attempt_save_event.to(torch.float32)
        event_bonus += cfg.hand_save_bonus * second_attempt_hand_save_event.to(torch.float32)
        event_bonus += cfg.second_save_bonus * second_save_event.to(torch.float32)
        event_bonus += cfg.second_hand_save_bonus * second_hand_save_event.to(torch.float32)
        event_bonus += cfg.recovery_bonus * just_recovered.to(torch.float32)
        event_bonus -= cfg.false_contact_penalty * false_contact.to(torch.float32)
        safety_penalty = cfg.unsafe_penalty * unsafe.to(torch.float32)
        safety_penalty += cfg.save_then_unsafe_penalty * (unsafe & self.first_save).to(
            torch.float32
        )
        timed_out = time >= cfg.episode_duration_sec
        self.phase[unsafe & active] = int(GoalkeeperEpisodePhase.FAILED)
        # Terminal phases are monotonic.  In particular, quarantining a
        # failed physics world restores a finite posture; timeout must not
        # then launder its recorded FAILED state into COMPLETE.
        self.phase[timed_out & ~unsafe & active] = int(GoalkeeperEpisodePhase.COMPLETE)
        terminated = unsafe | timed_out
        total = (
            reach
            + bimanual_reach
            + task_motion
            + upright
            + recovery_progress
            + event_bonus
            - smoothness_penalty
            - effort_penalty
            - safety_penalty
        )
        self._previous_contact.copy_(contact)
        self._previous_hand_contact.copy_(hand_contact)
        self._previous_save.copy_(save)
        self._previous_hard_height_potential.copy_(
            torch.where(shot > 0, hard_potential, torch.zeros_like(hard_potential))
        )
        self._previous_reach_potential.copy_(
            torch.where(shot > 0, reach_potential, torch.zeros_like(reach_potential))
        )
        self._previous_bimanual_potential.copy_(
            torch.where(shot > 0, bimanual_potential, torch.zeros_like(bimanual_potential))
        )
        self._previous_task_motion_potential.copy_(
            torch.where(shot > 0, task_motion_potential, torch.zeros_like(task_motion_potential))
        )
        self._previous_task_motion_arrival_window.copy_(
            torch.where(shot > 0, arrival_window, torch.zeros_like(arrival_window))
        )
        self._previous_recovery_potential.copy_(recovery_potential)
        self._previous_shot.copy_(shot)
        return {
            "total": total,
            "reach": reach,
            "bimanual_reach": bimanual_reach,
            "task_motion": task_motion,
            "upright": upright,
            "recovery_progress": recovery_progress,
            "smoothness_penalty": smoothness_penalty,
            "action_rate_penalty": action_rate_penalty,
            "joint_acceleration_penalty": joint_acceleration_penalty,
            "root_linear_speed_penalty": root_linear_speed_penalty,
            "root_angular_speed_penalty": root_angular_speed_penalty,
            "root_angular_excess_penalty": root_angular_excess_penalty,
            "action_magnitude_penalty": action_magnitude_penalty,
            "effort_penalty": effort_penalty,
            "event_bonus": event_bonus,
            "safety_penalty": safety_penalty,
            "phase": self.phase.clone(),
            "terminated": terminated,
            "first_save": self.first_save.clone(),
            "first_hand_save": self.first_hand_save.clone(),
            "recovered_after_first": self.recovered_after_first.clone(),
            "second_attempt_save": self.second_attempt_save.clone(),
            "second_attempt_hand_save": self.second_attempt_hand_save.clone(),
            "second_save": self.second_save.clone(),
            "second_hand_save": self.second_hand_save.clone(),
        }

    def summary(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "schema_version": "rosclaw_soccer.goalkeeper_multistep_summary.v6",
            "config_hash": self.config.config_hash,
            "environment_count": self.environment_count,
            "first_contact_rate": float(torch.mean(self.first_contact.to(torch.float32)).item()),
            "first_save_rate": float(torch.mean(self.first_save.to(torch.float32)).item()),
            "first_hand_save_rate": float(
                torch.mean(self.first_hand_save.to(torch.float32)).item()
            ),
            "recovery_rate": float(torch.mean(self.recovered_after_first.to(torch.float32)).item()),
            "second_attempt_save_rate": float(
                torch.mean(self.second_attempt_save.to(torch.float32)).item()
            ),
            "second_attempt_hand_save_rate": float(
                torch.mean(self.second_attempt_hand_save.to(torch.float32)).item()
            ),
            "second_contact_rate": float(torch.mean(self.second_contact.to(torch.float32)).item()),
            "second_save_rate": float(torch.mean(self.second_save.to(torch.float32)).item()),
            "second_hand_save_rate": float(
                torch.mean(self.second_hand_save.to(torch.float32)).item()
            ),
            "failed_rate": float(
                torch.mean(
                    (self.phase == int(GoalkeeperEpisodePhase.FAILED)).to(torch.float32)
                ).item()
            ),
            "bimanual_reach_fraction": float(
                (
                    torch.sum(self._bimanual_reach_steps).to(torch.float32)
                    / torch.clamp(torch.sum(self._active_flight_steps), min=1).to(torch.float32)
                ).item()
            ),
            "promotion_status": "TRAINING_METRICS_ONLY_NOT_PROMOTED",
            "activation_ceiling": "SIM_ONLY",
        }

    def _validate(self, sample: dict[str, Any]) -> None:
        torch = self.torch
        count = self.environment_count
        required = {
            "time_sec": (count,),
            "ball_velocity_mps": (count, 3),
            "ball_position_m": (count, 3),
            "intercept_target_m": (count, 3),
            "left_hand_position_m": (count, 3),
            "right_hand_position_m": (count, 3),
            "pelvis_height_m": (count,),
            "root_linear_velocity_mps": (count, 3),
            "root_angular_velocity_rad_s": (count, 3),
            "upright_projection": (count,),
            "ball_contact": (count,),
            "hand_contact": (count,),
            "true_save": (count,),
            "shot_index": (count,),
        }
        for name, shape in required.items():
            if name not in sample or tuple(sample[name].shape) != shape:
                raise ValueError(f"goalkeeper torch step {name} must have shape {shape}")
            if sample[name].device != self.phase.device:
                raise ValueError("goalkeeper torch step tensors must share one device")
        for name in ("action", "previous_action", "joint_acceleration_rad_s2", "applied_torque_nm"):
            if name not in sample or sample[name].ndim != 2 or sample[name].shape[0] != count:
                raise ValueError(f"goalkeeper torch step {name} must have shape (N, A)")
        if sample["action"].shape[1] != sample["previous_action"].shape[1]:
            raise ValueError("goalkeeper torch current and previous action widths must match")
        if sample["joint_acceleration_rad_s2"].shape[1] != sample["applied_torque_nm"].shape[1]:
            raise ValueError("goalkeeper torch joint acceleration and torque widths must match")
        if any(
            sample[name].dtype != torch.bool
            for name in ("ball_contact", "hand_contact", "true_save")
        ):
            raise ValueError("goalkeeper torch contact and save channels must be boolean")
        posture_exception = sample.get("posture_exception_granted")
        if posture_exception is not None and (
            tuple(posture_exception.shape) != (count,)
            or posture_exception.device != self.phase.device
            or posture_exception.dtype != torch.bool
        ):
            raise ValueError("goalkeeper torch posture exception must have shape (N,) and bool")
        pelvis_position = sample.get("pelvis_position_m")
        if pelvis_position is not None and (
            tuple(pelvis_position.shape) != (count, 3)
            or pelvis_position.device != self.phase.device
            or not pelvis_position.dtype.is_floating_point
        ):
            raise ValueError("goalkeeper torch pelvis position must have shape (N, 3)")
        if bool(torch.any(sample["hand_contact"] & ~sample["ball_contact"])):
            raise ValueError("goalkeeper hand contact must also be a robot-ball contact")
        if sample["shot_index"].dtype != torch.long:
            raise ValueError("goalkeeper torch shot index must use int64")
        for name, tensor in sample.items():
            if tensor.dtype.is_floating_point and not bool(torch.all(torch.isfinite(tensor))):
                raise ValueError(f"goalkeeper torch step {name} must be finite")


__all__ = ["TorchGoalkeeperMultiStepAccumulator"]
