"""Bounded whole-body mobility option for physics-trained G1 goalkeepers.

The option separates three responsibilities that the legacy combat adapter
coupled into one scalar: stable lateral locomotion, frozen whole-body teacher
activation, and upper-body residual reach.  It is simulation-only and all
targets remain behind joint, torque, action-rate, and CPU promotion guards.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

MOBILE_TEACHER_GROUP_SCALE = (0.25,) * 12 + (0.80,) * 3 + (1.00,) * 14
MOBILE_UPPER_BODY_KP = (110.0, 110.0, 80.0) + (
    100.0,
    100.0,
    100.0,
    100.0,
    20.0,
    20.0,
    20.0,
) * 2
MOBILE_UPPER_BODY_KD = (2.0, 2.0, 2.0) + (2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5) * 2


@dataclass(frozen=True)
class GoalkeeperMobilityOptionConfig:
    lateral_command_limit: float = 0.75
    recovery_command_limit: float = 0.55
    goal_boundary_m: float = 0.96
    capture_horizon_sec: float = 0.28
    waist_residual_limit: float = 0.22
    first_arm_residual_limit: float = 0.35
    second_arm_residual_limit: float = 0.42
    teacher_gate_step: float = 0.24
    teacher_gate_filter_fraction: float = 0.50
    predictive_teacher_gate_floor: float = 0.0
    residual_plasticity_scale: float = 0.0
    waist_residual_plasticity_scale: float | None = None
    arm_residual_plasticity_scale: float | None = None
    teacher_lower_body_scale: float = 0.25
    teacher_waist_scale: float = 0.80
    teacher_arm_scale: float = 1.00
    teacher_lower_body_target_step_rad: float = 0.08
    teacher_lower_body_target_filter_fraction: float = 0.35
    teacher_waist_target_step_rad: float = 0.05
    teacher_waist_target_filter_fraction: float = 0.25
    teacher_arm_target_step_rad: float = 0.045
    teacher_arm_target_filter_fraction: float = 0.15
    counter_rotation_enabled: bool = False
    anticipatory_arm_reach_enabled: bool = False
    predictive_teacher_warmstart_enabled: bool = False
    teacher_recovery_latch_enabled: bool = False
    teacher_recovery_hold_sec: float = 0.24
    teacher_recovery_decay_sec: float = 0.60
    lateral_velocity_guard_enabled: bool = False
    lateral_velocity_guard_onset_mps: float = 0.55
    lateral_velocity_guard_ceiling_mps: float = 0.85
    substep_upper_body_guard_enabled: bool = False
    substep_upper_body_guard_onset_rad_s: float = 1.80
    substep_upper_body_guard_ceiling_rad_s: float = 2.80
    substep_upper_body_minimum_position_scale: float = 0.05
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_mobility_option.v9"

    def __post_init__(self) -> None:
        values = (
            self.lateral_command_limit,
            self.recovery_command_limit,
            self.goal_boundary_m,
            self.capture_horizon_sec,
            self.waist_residual_limit,
            self.first_arm_residual_limit,
            self.second_arm_residual_limit,
            self.teacher_gate_step,
            self.teacher_gate_filter_fraction,
            self.teacher_recovery_hold_sec,
            self.teacher_recovery_decay_sec,
            self.lateral_velocity_guard_onset_mps,
            self.lateral_velocity_guard_ceiling_mps,
            self.substep_upper_body_guard_onset_rad_s,
            self.substep_upper_body_guard_ceiling_rad_s,
            self.substep_upper_body_minimum_position_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("goalkeeper mobility option values must be finite and positive")
        if not 0.35 <= self.lateral_command_limit <= 1.0:
            raise ValueError("goalkeeper mobility lateral command is invalid")
        if not 0.25 <= self.recovery_command_limit <= self.lateral_command_limit:
            raise ValueError("goalkeeper mobility recovery command is invalid")
        if not 0.75 <= self.goal_boundary_m <= 1.08:
            raise ValueError("goalkeeper mobility goal boundary is invalid")
        if not 0.10 <= self.capture_horizon_sec <= 0.50:
            raise ValueError("goalkeeper mobility capture horizon is invalid")
        if not 0.15 <= self.waist_residual_limit <= 0.40:
            raise ValueError("goalkeeper mobility waist residual is invalid")
        if not 0.30 <= self.first_arm_residual_limit <= 0.65:
            raise ValueError("goalkeeper mobility first-arm residual is invalid")
        if not self.first_arm_residual_limit <= self.second_arm_residual_limit <= 0.70:
            raise ValueError("goalkeeper mobility second-arm residual is invalid")
        if not 0.05 <= self.teacher_gate_step <= 0.40:
            raise ValueError("goalkeeper mobility teacher-gate step is invalid")
        if not 0.10 <= self.teacher_gate_filter_fraction <= 1.0:
            raise ValueError("goalkeeper mobility teacher-gate filter is invalid")
        if not math.isfinite(self.predictive_teacher_gate_floor) or not (
            0.0 <= self.predictive_teacher_gate_floor <= 0.80
        ):
            raise ValueError("goalkeeper mobility predictive teacher-gate floor is invalid")
        if (
            not math.isfinite(self.residual_plasticity_scale)
            or not 0.0 <= self.residual_plasticity_scale <= 1.0
        ):
            raise ValueError("goalkeeper mobility residual plasticity is invalid")
        group_plasticity = (
            self.waist_residual_plasticity_scale,
            self.arm_residual_plasticity_scale,
        )
        if any(
            value is not None
            and (not math.isfinite(value) or not 0.0 <= value <= 1.0)
            for value in group_plasticity
        ):
            raise ValueError("goalkeeper mobility group residual plasticity is invalid")
        teacher_scales = (
            self.teacher_lower_body_scale,
            self.teacher_waist_scale,
            self.teacher_arm_scale,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in teacher_scales):
            raise ValueError("goalkeeper mobility teacher group scale is invalid")
        if not (
            0.01 <= self.teacher_lower_body_target_step_rad <= 0.20
            and 0.10 <= self.teacher_lower_body_target_filter_fraction <= 1.0
            and 0.01 <= self.teacher_waist_target_step_rad <= 0.10
            and 0.10 <= self.teacher_waist_target_filter_fraction <= 1.0
            and 0.02 <= self.teacher_arm_target_step_rad <= 0.20
            and 0.10 <= self.teacher_arm_target_filter_fraction <= 1.0
        ):
            raise ValueError("goalkeeper mobility teacher target filter is invalid")
        if not isinstance(self.counter_rotation_enabled, bool):
            raise ValueError("goalkeeper mobility counter-rotation flag is invalid")
        if not isinstance(self.anticipatory_arm_reach_enabled, bool):
            raise ValueError("goalkeeper mobility anticipatory-arm flag is invalid")
        if not isinstance(self.predictive_teacher_warmstart_enabled, bool):
            raise ValueError("goalkeeper mobility predictive-teacher flag is invalid")
        if not isinstance(self.teacher_recovery_latch_enabled, bool):
            raise ValueError("goalkeeper mobility teacher-recovery latch flag is invalid")
        if not isinstance(self.lateral_velocity_guard_enabled, bool):
            raise ValueError("goalkeeper mobility lateral-velocity guard flag is invalid")
        if not isinstance(self.substep_upper_body_guard_enabled, bool):
            raise ValueError("goalkeeper mobility substep upper-body guard flag is invalid")
        if self.anticipatory_arm_reach_enabled and self.effective_arm_plasticity_scale <= 0.0:
            raise ValueError("goalkeeper anticipatory arms require residual plasticity")
        if not 0.10 <= self.teacher_recovery_hold_sec <= 0.60:
            raise ValueError("goalkeeper mobility teacher-recovery hold is invalid")
        if not 0.20 <= self.teacher_recovery_decay_sec <= 1.20:
            raise ValueError("goalkeeper mobility teacher-recovery decay is invalid")
        if not (
            0.30
            <= self.lateral_velocity_guard_onset_mps
            < self.lateral_velocity_guard_ceiling_mps
            <= 1.20
        ):
            raise ValueError("goalkeeper mobility lateral-velocity guard is invalid")
        if not (
            1.0
            <= self.substep_upper_body_guard_onset_rad_s
            < self.substep_upper_body_guard_ceiling_rad_s
            <= 3.5
            and 0.02 <= self.substep_upper_body_minimum_position_scale <= 0.40
        ):
            raise ValueError("goalkeeper mobility substep upper-body guard is invalid")
        if len(MOBILE_TEACHER_GROUP_SCALE) != 29:
            raise ValueError("goalkeeper mobility teacher group scale is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("goalkeeper mobility option is SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))

    @property
    def teacher_group_scale(self) -> tuple[float, ...]:
        return (
            (self.teacher_lower_body_scale,) * 12
            + (self.teacher_waist_scale,) * 3
            + (self.teacher_arm_scale,) * 14
        )

    @property
    def teacher_target_filter_contracts(
        self,
    ) -> tuple[tuple[int, int, float, float], ...]:
        """Return canonical joint slices and causal slew/filter settings."""

        return (
            (
                0,
                12,
                self.teacher_lower_body_target_step_rad,
                self.teacher_lower_body_target_filter_fraction,
            ),
            (
                12,
                15,
                self.teacher_waist_target_step_rad,
                self.teacher_waist_target_filter_fraction,
            ),
            (
                15,
                29,
                self.teacher_arm_target_step_rad,
                self.teacher_arm_target_filter_fraction,
            ),
        )

    @property
    def effective_waist_plasticity_scale(self) -> float:
        """Return the explicit waist scale, falling back for old checkpoints."""

        value = self.waist_residual_plasticity_scale
        return self.residual_plasticity_scale if value is None else value

    @property
    def effective_arm_plasticity_scale(self) -> float:
        """Return the explicit arm scale, falling back for old checkpoints."""

        value = self.arm_residual_plasticity_scale
        return self.residual_plasticity_scale if value is None else value


def project_recovery_command_numpy(
    *,
    requested: float,
    root_lateral_position_m: float,
    root_lateral_velocity_mps: float,
    config: GoalkeeperMobilityOptionConfig,
    predictive_threat: bool = False,
) -> float:
    """Keep a learned recovery command unless it would leave the goal mouth."""

    command_limit = (
        config.lateral_command_limit if predictive_threat else config.recovery_command_limit
    )
    command = float(np.clip(requested, -command_limit, command_limit))
    capture = root_lateral_position_m + config.capture_horizon_sec * root_lateral_velocity_mps
    # Keeper yaw is pi: positive local lateral command moves toward world -y.
    outward = abs(capture) >= config.goal_boundary_m and capture * command < 0.0
    if outward:
        return float(np.sign(capture) * abs(command))
    return command


def guard_lateral_velocity_numpy(
    *,
    requested: float,
    root_lateral_velocity_mps: float,
    config: GoalkeeperMobilityOptionConfig,
) -> float:
    """Fade acceleration into a bounded counter-step before lateral runaway."""

    command = float(np.clip(requested, -config.lateral_command_limit, config.lateral_command_limit))
    if not config.lateral_velocity_guard_enabled:
        return command
    speed = abs(root_lateral_velocity_mps)
    continuing = root_lateral_velocity_mps * command < 0.0
    if not continuing or speed <= config.lateral_velocity_guard_onset_mps:
        return command
    span = config.lateral_velocity_guard_ceiling_mps - config.lateral_velocity_guard_onset_mps
    authority = float(np.clip((config.lateral_velocity_guard_ceiling_mps - speed) / span, 0.0, 1.0))
    if authority > 0.0:
        return command * authority
    brake = float(np.clip(abs(command), 0.20, 0.35))
    return float(np.sign(root_lateral_velocity_mps) * brake)


def substep_upper_body_authority_numpy(
    *,
    root_angular_velocity_rad_s: np.ndarray,
    config: GoalkeeperMobilityOptionConfig,
) -> float:
    """Attenuate position drive inside a 2 ms physics step, preserving damping."""

    angular = np.asarray(root_angular_velocity_rad_s, dtype=np.float64)
    if angular.shape != (3,) or not np.all(np.isfinite(angular)):
        raise ValueError("goalkeeper substep angular velocity is invalid")
    if not config.substep_upper_body_guard_enabled:
        return 1.0
    speed = float(np.linalg.norm(angular))
    fraction = float(
        np.clip(
            (speed - config.substep_upper_body_guard_onset_rad_s)
            / (
                config.substep_upper_body_guard_ceiling_rad_s
                - config.substep_upper_body_guard_onset_rad_s
            ),
            0.0,
            1.0,
        )
    )
    return 1.0 - fraction * (1.0 - config.substep_upper_body_minimum_position_scale)


def substep_upper_body_authority_torch(
    *,
    root_angular_velocity_rad_s: Any,
    config: GoalkeeperMobilityOptionConfig,
) -> Any:
    """Torch equivalent of :func:`substep_upper_body_authority_numpy`."""

    import torch

    angular = root_angular_velocity_rad_s
    if angular.ndim != 2 or angular.shape[1] != 3 or not bool(torch.all(torch.isfinite(angular))):
        raise ValueError("goalkeeper substep angular velocity batch is invalid")
    if not config.substep_upper_body_guard_enabled:
        return torch.ones(angular.shape[0], dtype=angular.dtype, device=angular.device)
    speed = torch.linalg.vector_norm(angular, dim=1)
    fraction = torch.clamp(
        (speed - config.substep_upper_body_guard_onset_rad_s)
        / (
            config.substep_upper_body_guard_ceiling_rad_s
            - config.substep_upper_body_guard_onset_rad_s
        ),
        0.0,
        1.0,
    )
    return 1.0 - fraction * (1.0 - config.substep_upper_body_minimum_position_scale)


__all__ = [
    "GoalkeeperMobilityOptionConfig",
    "MOBILE_TEACHER_GROUP_SCALE",
    "MOBILE_UPPER_BODY_KD",
    "MOBILE_UPPER_BODY_KP",
    "project_recovery_command_numpy",
    "guard_lateral_velocity_numpy",
    "substep_upper_body_authority_numpy",
    "substep_upper_body_authority_torch",
]
