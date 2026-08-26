"""GPU-resident MJWarp environment for continuous G1 goalkeeper learning.

The environment reuses the qualified G1 locomotion policy as a frozen
cerebellar prior.  A learnable residual may command lateral shuffling and the
waist/arms, but cannot replace the lower-body stabilizer.  Every rollout uses
the same native MuJoCo G1, football, and goal geometry as the strict CPU exam.

All runtime dependencies are imported lazily so the normal Soccer package
remains usable without Torch, Warp, or MJWarp.  This is a SIM_ONLY candidate
generator; it has no promotion or hardware authority.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json
from rosclaw_soccer.training.goalkeeper_agility import (
    GoalkeeperAgilityConfig,
    shape_goalkeeper_action_torch,
)
from rosclaw_soccer.training.goalkeeper_multistep import GoalkeeperMultiStepConfig
from rosclaw_soccer.training.goalkeeper_multistep_torch import (
    TorchGoalkeeperMultiStepAccumulator,
)


@dataclass(frozen=True)
class GoalkeeperMJWarpConfig:
    """Content-addressed GPU physics and domain-randomization contract."""

    environment_count: int = 64
    control_dt_sec: float = 0.02
    physics_substeps: int = 10
    episode_duration_sec: float = 5.0
    first_shot_release_sec: float = 0.25
    first_shot_end_sec: float = 1.65
    second_shot_release_sec: float = 2.65
    second_shot_end_sec: float = 4.15
    keeper_x_m: float = 4.52
    ball_start_x_range_m: tuple[float, float] = (1.75, 2.30)
    ball_start_z_range_m: tuple[float, float] = (0.35, 0.65)
    target_y_range_m: tuple[float, float] = (-1.0, 1.0)
    target_z_range_m: tuple[float, float] = (0.25, 1.48)
    flight_time_range_sec: tuple[float, float] = (0.48, 0.72)
    second_shot_probability: float = 0.75
    shot_intent_cue_enabled: bool = False
    shot_intent_cue_lateral_noise_m: float = 0.16
    shot_intent_cue_height_noise_m: float = 0.12
    shot_intent_cue_dropout_probability: float = 0.12
    hard_shot_fraction: float = 0.0
    hard_shot_height_mode: Literal["low", "mid", "high", "balanced"] = "high"
    hard_shot_side_mode: Literal["negative", "positive", "balanced"] = "balanced"
    hard_shot_flight_time_range_sec: tuple[float, float] | None = None
    maximum_lateral_command_mps: float = 0.40
    residual_scale: float = 0.70
    arm_residual_scale_multiplier: float = 1.0
    action_filter_fraction: float = 0.30
    maximum_action_step: float = 0.18
    arm_action_filter_fraction: float = 0.30
    maximum_arm_action_step: float = 0.18
    second_shot_arm_authority_scale: float = 1.0
    agility: GoalkeeperAgilityConfig = GoalkeeperAgilityConfig()
    root_linear_speed_penalty_scale: float = 0.030
    root_angular_speed_penalty_scale: float = 0.080
    root_angular_speed_soft_limit_rad_s: float = 3.50
    root_angular_speed_excess_penalty_scale: float = 0.0
    flight_root_angular_penalty_scale: float = 1.0
    action_magnitude_penalty_scale: float = 0.008
    hand_contact_bonus: float = 4.0
    true_save_bonus: float = 25.0
    hand_save_bonus: float = 12.0
    second_hand_save_bonus: float = 28.0
    reach_reward_scale: float = 0.70
    reach_reward_semantics: str = "STATE_DENSITY"
    hard_height_reach_reward_scale: float = 0.0
    hard_height_reach_threshold_m: float = 1.10
    hard_height_reach_distance_decay: float = 1.25
    hard_height_reach_reward_semantics: str = "POTENTIAL_PROGRESS_ONLY"
    bimanual_reach_reward_scale: float = 1.00
    second_shot_reach_reward_multiplier: float = 1.0
    task_motion_reward_scale: float = 0.0
    recovery_progress_reward_scale: float = 0.0
    recovery_progress_linear_speed_decay: float = 2.0
    recovery_progress_angular_speed_decay: float = 0.50
    recovery_event_bonus: float = 15.0
    unsafe_penalty: float = 50.0
    save_then_unsafe_penalty: float = 0.0
    nconmax_per_world: int = 256
    njmax_per_world: int = 1024
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    difficulty_profile: Literal["standard", "match", "advanced", "elite"] = "standard"
    schema_version: str = "rosclaw_soccer.goalkeeper_mjwarp_config.v28"

    def __post_init__(self) -> None:
        if not 1 <= self.environment_count <= 4096:
            raise ValueError("MJWarp goalkeeper environment count must be in [1, 4096]")
        if not math.isclose(
            self.control_dt_sec / self.physics_substeps,
            0.002,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("MJWarp goalkeeper physics step must match qualified 0.002 s model")
        timeline = (
            self.first_shot_release_sec,
            self.first_shot_end_sec,
            self.second_shot_release_sec,
            self.second_shot_end_sec,
            self.episode_duration_sec,
        )
        if (
            any(not math.isfinite(value) for value in timeline)
            or tuple(sorted(timeline)) != timeline
        ):
            raise ValueError("MJWarp goalkeeper shot timeline must be finite and ordered")
        ranges = (
            self.ball_start_x_range_m,
            self.ball_start_z_range_m,
            self.target_y_range_m,
            self.target_z_range_m,
            self.flight_time_range_sec,
        )
        if any(
            not all(math.isfinite(value) for value in limits) or limits[0] >= limits[1]
            for limits in ranges
        ):
            raise ValueError("MJWarp goalkeeper randomization ranges must be finite and increasing")
        if not 0.0 <= self.second_shot_probability <= 1.0:
            raise ValueError("MJWarp goalkeeper second-shot probability must be in [0, 1]")
        if not math.isfinite(self.hard_shot_fraction) or not 0.0 <= self.hard_shot_fraction <= 1.0:
            raise ValueError("MJWarp goalkeeper hard-shot fraction must be in [0, 1]")
        if self.hard_shot_height_mode not in {"low", "mid", "high", "balanced"}:
            raise ValueError("MJWarp goalkeeper hard-shot height mode is invalid")
        if self.hard_shot_side_mode not in {"negative", "positive", "balanced"}:
            raise ValueError("MJWarp goalkeeper hard-shot side mode is invalid")
        if self.hard_shot_flight_time_range_sec is not None:
            hard_flight = self.hard_shot_flight_time_range_sec
            if (
                not all(math.isfinite(value) for value in hard_flight)
                or hard_flight[0] >= hard_flight[1]
                or not 0.30 <= hard_flight[0] < hard_flight[1] <= 1.20
                or self.hard_shot_fraction <= 0.0
            ):
                raise ValueError("MJWarp goalkeeper hard-shot flight curriculum is invalid")
        if not isinstance(self.shot_intent_cue_enabled, bool):
            raise ValueError("MJWarp goalkeeper shot-intent cue flag must be boolean")
        if (
            not math.isfinite(self.shot_intent_cue_lateral_noise_m)
            or not 0.05 <= self.shot_intent_cue_lateral_noise_m <= 0.40
            or not math.isfinite(self.shot_intent_cue_height_noise_m)
            or not 0.05 <= self.shot_intent_cue_height_noise_m <= 0.35
            or not math.isfinite(self.shot_intent_cue_dropout_probability)
            or not 0.0 <= self.shot_intent_cue_dropout_probability <= 0.50
        ):
            raise ValueError("MJWarp goalkeeper shot-intent cue uncertainty is invalid")
        if self.difficulty_profile not in {"standard", "match", "advanced", "elite"}:
            raise ValueError("MJWarp goalkeeper difficulty profile is invalid")
        if self.difficulty_profile == "match" and not self.shot_intent_cue_enabled:
            raise ValueError("match goalkeeper profile requires causal shot-intent cue")
        if not 0.0 < self.maximum_lateral_command_mps <= 0.40:
            raise ValueError("MJWarp goalkeeper lateral command exceeds qualified locomotion range")
        if not all(
            0.0 < value <= 1.0
            for value in (
                self.residual_scale,
                self.action_filter_fraction,
                self.maximum_action_step,
                self.arm_action_filter_fraction,
                self.maximum_arm_action_step,
                self.second_shot_arm_authority_scale,
                self.root_linear_speed_penalty_scale,
                self.root_angular_speed_penalty_scale,
                self.flight_root_angular_penalty_scale,
                self.action_magnitude_penalty_scale,
            )
        ):
            raise ValueError("MJWarp goalkeeper residual/filter settings must be in (0, 1]")
        reward_values = (
            self.hand_contact_bonus,
            self.second_hand_save_bonus,
            self.reach_reward_scale,
            self.bimanual_reach_reward_scale,
            self.second_shot_reach_reward_multiplier,
        )
        if any(not math.isfinite(value) or not 0.0 < value <= 100.0 for value in reward_values):
            raise ValueError("MJWarp goalkeeper reward settings must be in (0, 100]")
        save_event_rewards = (
            self.true_save_bonus,
            self.hand_save_bonus,
            self.recovery_event_bonus,
        )
        if any(
            not math.isfinite(value) or not 0.0 < value <= 500.0 for value in save_event_rewards
        ):
            raise ValueError("MJWarp goalkeeper success-event rewards must be in (0, 500]")
        if not math.isfinite(self.unsafe_penalty) or not 10.0 <= self.unsafe_penalty <= 1000.0:
            raise ValueError("MJWarp goalkeeper unsafe penalty must be in [10, 1000]")
        if not (
            math.isfinite(self.save_then_unsafe_penalty)
            and 0.0 <= self.save_then_unsafe_penalty <= 2_000.0
        ):
            raise ValueError("MJWarp goalkeeper save-then-unsafe penalty is invalid")
        if not (
            math.isfinite(self.root_angular_speed_soft_limit_rad_s)
            and 0.50 <= self.root_angular_speed_soft_limit_rad_s <= 8.0
            and math.isfinite(self.root_angular_speed_excess_penalty_scale)
            and 0.0 <= self.root_angular_speed_excess_penalty_scale <= 100.0
            and self.flight_root_angular_penalty_scale <= 1.0
        ):
            raise ValueError("MJWarp goalkeeper root-angular tail penalty is invalid")
        if (
            not math.isfinite(self.hard_height_reach_reward_scale)
            or not 0.0 <= self.hard_height_reach_reward_scale <= 10.0
            or not math.isfinite(self.hard_height_reach_threshold_m)
            or not 0.80 <= self.hard_height_reach_threshold_m <= 1.40
            or not math.isfinite(self.hard_height_reach_distance_decay)
            or not 0.50 <= self.hard_height_reach_distance_decay <= 4.0
        ):
            raise ValueError("MJWarp goalkeeper hard-height reach reward is invalid")
        if self.hard_height_reach_reward_semantics != "POTENTIAL_PROGRESS_ONLY":
            raise ValueError("MJWarp goalkeeper hard-height reach semantics are invalid")
        if self.reach_reward_semantics not in {"STATE_DENSITY", "POTENTIAL_PROGRESS_ONLY"}:
            raise ValueError("MJWarp goalkeeper reach reward semantics are invalid")
        if (
            not math.isfinite(self.task_motion_reward_scale)
            or not 0.0 <= self.task_motion_reward_scale <= 20.0
        ):
            raise ValueError("MJWarp goalkeeper task-motion reward is invalid")
        if (
            not math.isfinite(self.recovery_progress_reward_scale)
            or not 0.0 <= self.recovery_progress_reward_scale <= 100.0
            or not math.isfinite(self.recovery_progress_linear_speed_decay)
            or not 0.10 <= self.recovery_progress_linear_speed_decay <= 20.0
            or not math.isfinite(self.recovery_progress_angular_speed_decay)
            or not 0.05 <= self.recovery_progress_angular_speed_decay <= 10.0
        ):
            raise ValueError("MJWarp goalkeeper recovery-progress reward is invalid")
        if not 1.0 <= self.second_shot_reach_reward_multiplier <= 3.0:
            raise ValueError("MJWarp goalkeeper second-shot reach multiplier is invalid")
        if not 1.0 <= self.arm_residual_scale_multiplier <= 1.50:
            raise ValueError("MJWarp goalkeeper arm residual multiplier must be in [1, 1.5]")
        if self.nconmax_per_world < 128 or self.njmax_per_world < 512:
            raise ValueError("MJWarp goalkeeper contact/Jacobian capacities are too small")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("MJWarp goalkeeper learning is SIM_ONLY")

    @property
    def episode_steps(self) -> int:
        return int(round(self.episode_duration_sec / self.control_dt_sec))

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))

    @property
    def maximum_applied_actor_action_step(self) -> float:
        """Largest declared post-filter step across core and arm channels."""

        return max(
            self.maximum_action_step * self.action_filter_fraction,
            self.maximum_arm_action_step * self.arm_action_filter_fraction,
        )


def _root_angular_speed_tail_summary(
    *, torch: Any, maximum_speeds: Any, soft_limit_rad_s: float
) -> dict[str, float]:
    """Expose population tail risk without weakening the hard maximum gate."""

    strict_stability_ceiling = 3.50
    return {
        "maximum_root_angular_speed_rad_s": float(maximum_speeds.max()),
        "p95_maximum_root_angular_speed_rad_s": float(torch.quantile(maximum_speeds, 0.95)),
        "p99_maximum_root_angular_speed_rad_s": float(torch.quantile(maximum_speeds, 0.99)),
        "root_angular_speed_soft_limit_rad_s": soft_limit_rad_s,
        "root_angular_speed_soft_limit_exceedance_rate": float(
            (maximum_speeds > soft_limit_rad_s).to(torch.float32).mean()
        ),
        "strict_stability_ceiling_rad_s": strict_stability_ceiling,
        "strict_stability_ceiling_exceedance_rate": float(
            (maximum_speeds > strict_stability_ceiling).to(torch.float32).mean()
        ),
    }


def goalkeeper_world_config(
    *,
    difficulty_profile: Literal["standard", "match", "advanced", "elite"],
    environment_count: int,
    second_shot_probability: float = 0.75,
    shot_intent_cue_enabled: bool = False,
    hard_shot_fraction: float = 0.0,
    hard_shot_height_mode: Literal["low", "mid", "high", "balanced"] = "high",
    hard_shot_side_mode: Literal["negative", "positive", "balanced"] = "balanced",
    hard_shot_flight_time_range_sec: tuple[float, float] | None = None,
    hard_height_reach_reward_scale: float = 0.0,
    reach_reward_semantics: str = "STATE_DENSITY",
    hard_height_reach_threshold_m: float = 1.10,
    hard_height_reach_distance_decay: float = 1.25,
    task_motion_reward_scale: float = 0.0,
    recovery_event_bonus: float = 15.0,
    unsafe_penalty: float = 50.0,
    save_then_unsafe_penalty: float = 0.0,
) -> GoalkeeperMJWarpConfig:
    """Return one declared shot curriculum instead of ad-hoc range edits."""

    if difficulty_profile == "standard":
        return GoalkeeperMJWarpConfig(
            environment_count=environment_count,
            second_shot_probability=second_shot_probability,
            shot_intent_cue_enabled=shot_intent_cue_enabled,
            hard_shot_fraction=hard_shot_fraction,
            hard_shot_height_mode=hard_shot_height_mode,
            hard_shot_side_mode=hard_shot_side_mode,
            hard_shot_flight_time_range_sec=hard_shot_flight_time_range_sec,
            hard_height_reach_reward_scale=hard_height_reach_reward_scale,
            reach_reward_semantics=reach_reward_semantics,
            hard_height_reach_threshold_m=hard_height_reach_threshold_m,
            hard_height_reach_distance_decay=hard_height_reach_distance_decay,
            task_motion_reward_scale=task_motion_reward_scale,
            recovery_event_bonus=recovery_event_bonus,
            unsafe_penalty=unsafe_penalty,
            save_then_unsafe_penalty=save_then_unsafe_penalty,
            difficulty_profile="standard",
        )
    if difficulty_profile == "match":
        # A shooter exposes a realistic wind-up before launch, but the actual
        # shot is wider, higher and faster than the standard curriculum.  This
        # separates perceptual anticipation from ball-flight difficulty.
        return GoalkeeperMJWarpConfig(
            environment_count=environment_count,
            first_shot_release_sec=0.70,
            first_shot_end_sec=1.70,
            second_shot_release_sec=3.05,
            second_shot_end_sec=4.35,
            ball_start_x_range_m=(0.45, 1.45),
            ball_start_z_range_m=(0.22, 0.78),
            target_y_range_m=(-1.08, 1.08),
            target_z_range_m=(0.10, 1.62),
            flight_time_range_sec=(0.34, 0.54),
            second_shot_probability=second_shot_probability,
            shot_intent_cue_enabled=shot_intent_cue_enabled,
            hard_shot_fraction=hard_shot_fraction,
            hard_shot_height_mode=hard_shot_height_mode,
            hard_shot_side_mode=hard_shot_side_mode,
            hard_shot_flight_time_range_sec=hard_shot_flight_time_range_sec,
            hard_height_reach_reward_scale=hard_height_reach_reward_scale,
            reach_reward_semantics=reach_reward_semantics,
            hard_height_reach_threshold_m=hard_height_reach_threshold_m,
            hard_height_reach_distance_decay=hard_height_reach_distance_decay,
            task_motion_reward_scale=task_motion_reward_scale,
            recovery_event_bonus=recovery_event_bonus,
            unsafe_penalty=unsafe_penalty,
            save_then_unsafe_penalty=save_then_unsafe_penalty,
            second_shot_reach_reward_multiplier=1.80,
            agility=GoalkeeperAgilityConfig(
                angular_guard_onset_rad_s=1.35,
                angular_guard_ceiling_rad_s=2.65,
                minimum_upper_body_scale=0.16,
                waist_authority_scale=0.52,
                arm_authority_scale=0.86,
                counter_rotation_gain=0.20,
            ),
            difficulty_profile="match",
        )
    if difficulty_profile == "advanced":
        return GoalkeeperMJWarpConfig(
            environment_count=environment_count,
            first_shot_release_sec=0.18,
            first_shot_end_sec=1.40,
            second_shot_release_sec=2.18,
            second_shot_end_sec=3.75,
            ball_start_x_range_m=(0.65, 1.65),
            ball_start_z_range_m=(0.25, 0.72),
            target_y_range_m=(-1.06, 1.06),
            target_z_range_m=(0.16, 1.48),
            flight_time_range_sec=(0.38, 0.60),
            second_shot_probability=second_shot_probability,
            shot_intent_cue_enabled=shot_intent_cue_enabled,
            hard_shot_fraction=hard_shot_fraction,
            hard_shot_height_mode=hard_shot_height_mode,
            hard_shot_side_mode=hard_shot_side_mode,
            hard_shot_flight_time_range_sec=hard_shot_flight_time_range_sec,
            hard_height_reach_reward_scale=hard_height_reach_reward_scale,
            reach_reward_semantics=reach_reward_semantics,
            hard_height_reach_threshold_m=hard_height_reach_threshold_m,
            hard_height_reach_distance_decay=hard_height_reach_distance_decay,
            task_motion_reward_scale=task_motion_reward_scale,
            recovery_event_bonus=recovery_event_bonus,
            unsafe_penalty=unsafe_penalty,
            save_then_unsafe_penalty=save_then_unsafe_penalty,
            second_shot_reach_reward_multiplier=1.60,
            agility=GoalkeeperAgilityConfig(
                angular_guard_onset_rad_s=1.40,
                angular_guard_ceiling_rad_s=2.75,
                minimum_upper_body_scale=0.18,
                waist_authority_scale=0.55,
                arm_authority_scale=0.82,
                counter_rotation_gain=0.18,
            ),
            difficulty_profile="advanced",
        )
    if difficulty_profile == "elite":
        return GoalkeeperMJWarpConfig(
            environment_count=environment_count,
            first_shot_release_sec=0.12,
            first_shot_end_sec=1.24,
            second_shot_release_sec=1.92,
            second_shot_end_sec=3.30,
            ball_start_x_range_m=(0.15, 1.10),
            ball_start_z_range_m=(0.18, 0.86),
            target_y_range_m=(-1.10, 1.10),
            target_z_range_m=(0.10, 1.65),
            flight_time_range_sec=(0.30, 0.50),
            second_shot_probability=second_shot_probability,
            shot_intent_cue_enabled=shot_intent_cue_enabled,
            hard_shot_fraction=hard_shot_fraction,
            hard_shot_height_mode=hard_shot_height_mode,
            hard_shot_side_mode=hard_shot_side_mode,
            hard_shot_flight_time_range_sec=hard_shot_flight_time_range_sec,
            hard_height_reach_reward_scale=hard_height_reach_reward_scale,
            reach_reward_semantics=reach_reward_semantics,
            hard_height_reach_threshold_m=hard_height_reach_threshold_m,
            hard_height_reach_distance_decay=hard_height_reach_distance_decay,
            task_motion_reward_scale=task_motion_reward_scale,
            recovery_event_bonus=recovery_event_bonus,
            unsafe_penalty=unsafe_penalty,
            save_then_unsafe_penalty=save_then_unsafe_penalty,
            maximum_lateral_command_mps=0.36,
            arm_residual_scale_multiplier=1.18,
            arm_action_filter_fraction=0.36,
            maximum_arm_action_step=0.20,
            second_shot_arm_authority_scale=0.65,
            root_angular_speed_penalty_scale=0.12,
            action_magnitude_penalty_scale=0.010,
            hand_contact_bonus=8.0,
            hand_save_bonus=20.0,
            second_hand_save_bonus=40.0,
            reach_reward_scale=1.40,
            bimanual_reach_reward_scale=1.60,
            second_shot_reach_reward_multiplier=2.00,
            agility=GoalkeeperAgilityConfig(
                lateral_response_gain=1.00,
                angular_guard_onset_rad_s=1.10,
                angular_guard_ceiling_rad_s=2.30,
                minimum_upper_body_scale=0.10,
                waist_authority_scale=0.55,
                arm_authority_scale=0.88,
                counter_rotation_gain=0.25,
            ),
            difficulty_profile="elite",
        )
    raise ValueError("unknown goalkeeper shot difficulty profile")


# The frozen locomotion expert uses Isaac joint order internally.  MuJoCo and
# ROSClaw use DDS motor order externally.
_LOCO_TO_MOTOR = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)
_LOCO_DEFAULT = (
    -0.2,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.42,
    0.42,
    0.35,
    0.35,
    -0.23,
    -0.23,
    0.18,
    -0.18,
    0.0,
    0.0,
    0.0,
    0.0,
    0.87,
    0.87,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
_LOCO_KP = (
    200,
    200,
    200,
    150,
    150,
    200,
    150,
    150,
    200,
    200,
    200,
    100,
    100,
    20,
    20,
    100,
    100,
    20,
    20,
    50,
    50,
    50,
    50,
    40,
    40,
    40,
    40,
    40,
    40,
)
_LOCO_KD = (
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
)
# One lateral locomotion command plus waist and both arms.  The lower-body
# residual is intentionally unavailable: stability stays under the frozen,
# qualified cerebellar prior.
_RESIDUAL_MOTOR_INDICES = tuple(range(12, 29))
_RESIDUAL_LIMITS_RAD = (
    0.16,
    0.14,
    0.12,
    0.70,
    0.75,
    0.55,
    0.55,
    0.18,
    0.16,
    0.16,
    0.70,
    0.75,
    0.55,
    0.55,
    0.18,
    0.16,
    0.16,
)


class GoalkeeperMJWarpBatch:
    """One-device vector world with Torch/Warp zero-copy state exchange."""

    observation_size = 74
    action_size = 18
    lower_body_authority = "FROZEN_QUALIFIED_LOCOMOTION_PRIOR"
    learned_residual_authority = "LATERAL_COMMAND_AND_UPPER_BODY_ONLY"

    def __init__(
        self,
        *,
        asset_root: Path,
        locomotion_policy_path: Path,
        device: Any,
        config: GoalkeeperMJWarpConfig | None = None,
    ) -> None:
        import mujoco
        import mujoco_warp as mjw
        import torch
        import warp as wp

        from rosclaw_soccer.world.field import build_g1_stadium_model

        self.torch = torch
        self.wp = wp
        self.mjw = mjw
        self.device = torch.device(device)
        self.config = config or GoalkeeperMJWarpConfig()
        self.observation_size = 77 if self.config.shot_intent_cue_enabled else 74
        self.count = self.config.environment_count
        self._asset_root = asset_root.expanduser().resolve()
        self._policy_path = locomotion_policy_path.expanduser().resolve()
        if not self._policy_path.is_file():
            raise FileNotFoundError(f"qualified locomotion policy not found: {self._policy_path}")
        wp.config.log_level = wp.LOG_WARNING
        wp.init()
        # MuJoCo-Warp launches some collision kernels through Warp's current
        # device rather than inferring it from Torch views.  ScopedDevice only
        # covered model construction here, so a process assigned to cuda:1+
        # later launched broadphase work on the default cuda:0 and could hit an
        # illegal cross-device memory access.  One trainer process owns one
        # device; pin its Warp default for every subsequent forward/step.
        wp.set_device(str(self.device))
        if str(wp.get_device()) != str(self.device):
            raise RuntimeError("MJWarp process did not bind the requested physics device")
        with wp.ScopedDevice(str(self.device)):
            self.cpu_model = build_g1_stadium_model(self._asset_root)
            cpu_data = mujoco.MjData(self.cpu_model)
            mujoco.mj_forward(self.cpu_model, cpu_data)
            self.model = mjw.put_model(self.cpu_model)
            self.data = mjw.put_data(
                self.cpu_model,
                cpu_data,
                nworld=self.count,
                nconmax=self.config.nconmax_per_world,
                njmax=self.config.njmax_per_world,
            )

        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.xpos = wp.to_torch(self.data.xpos)
        self.xquat = wp.to_torch(self.data.xquat)
        self.geom_xpos = wp.to_torch(self.data.geom_xpos)
        self.contact_geom = wp.to_torch(self.data.contact.geom)
        self.contact_world = wp.to_torch(self.data.contact.worldid)
        self.contact_distance = wp.to_torch(self.data.contact.dist)
        # The safety quarantine restores failed vector worlds immediately so
        # the remaining batch can continue.  Preserve the last finite physics
        # state before that restore; otherwise a failure-driven curriculum
        # would accidentally learn from the default standing reset pose.
        self._pre_quarantine_qpos = self.qpos.clone()
        self._pre_quarantine_qvel = self.qvel.clone()
        self._pre_quarantine_left_foot = torch.zeros(
            self.count, dtype=torch.bool, device=self.device
        )
        self._pre_quarantine_right_foot = torch.zeros_like(
            self._pre_quarantine_left_foot
        )
        self._loco_to_motor = torch.tensor(_LOCO_TO_MOTOR, dtype=torch.long, device=self.device)
        self._loco_default = torch.tensor(_LOCO_DEFAULT, dtype=torch.float32, device=self.device)
        self._kp = torch.zeros(29, dtype=torch.float32, device=self.device)
        self._kd = torch.zeros(29, dtype=torch.float32, device=self.device)
        self._kp[self._loco_to_motor] = torch.tensor(
            _LOCO_KP, dtype=torch.float32, device=self.device
        )
        self._kd[self._loco_to_motor] = torch.tensor(
            _LOCO_KD, dtype=torch.float32, device=self.device
        )
        self._torque_limits = torch.tensor(
            G1_HARD_TORQUE_LIMITS, dtype=torch.float32, device=self.device
        )
        self._residual_indices = torch.tensor(
            _RESIDUAL_MOTOR_INDICES, dtype=torch.long, device=self.device
        )
        self._residual_limits = torch.tensor(
            _RESIDUAL_LIMITS_RAD, dtype=torch.float32, device=self.device
        )
        self._residual_joint_lower = torch.tensor(
            self.cpu_model.jnt_range[13:30, 0], dtype=torch.float32, device=self.device
        )
        self._residual_joint_upper = torch.tensor(
            self.cpu_model.jnt_range[13:30, 1], dtype=torch.float32, device=self.device
        )
        self._joint_ranges = torch.tensor(
            self.cpu_model.jnt_range[1:30], dtype=torch.float32, device=self.device
        )
        self._joint_limited = torch.tensor(
            self.cpu_model.jnt_limited[1:30].astype(bool),
            dtype=torch.bool,
            device=self.device,
        )
        self._geom_body = torch.tensor(
            self.cpu_model.geom_bodyid, dtype=torch.long, device=self.device
        )
        self._ball_body = int(self.cpu_model.body("ball").id)
        self._hand_geoms = torch.tensor(
            (
                int(self.cpu_model.geom("left_hand_collision").id),
                int(self.cpu_model.geom("right_hand_collision").id),
                int(self.cpu_model.geom("left_goalkeeper_glove").id),
                int(self.cpu_model.geom("right_goalkeeper_glove").id),
            ),
            dtype=torch.long,
            device=self.device,
        )
        self._pelvis_body = int(self.cpu_model.body("pelvis").id)
        self._left_foot_body = int(self.cpu_model.body("left_ankle_roll_link").id)
        self._right_foot_body = int(self.cpu_model.body("right_ankle_roll_link").id)
        self._left_hand_geom = int(self.cpu_model.geom("left_hand_collision").id)
        self._right_hand_geom = int(self.cpu_model.geom("right_hand_collision").id)
        self._loco_policy = torch.jit.load(str(self._policy_path), map_location=self.device)
        self._loco_policy.eval()
        # Rehydrate the scripted LSTM as a native module so batched cuDNN
        # weights remain contiguous.  The exported wrapper owns hidden state
        # for one deployment robot and otherwise repacks weights every call.
        self._loco_rnn = torch.nn.LSTM(96, 256, num_layers=1).to(self.device)
        self._loco_rnn.load_state_dict(self._loco_policy.rnn.state_dict())
        self._loco_rnn.eval()
        self._loco_rnn.flatten_parameters()
        self._loco_hidden = torch.zeros((1, self.count, 256), device=self.device)
        self._loco_cell = torch.zeros_like(self._loco_hidden)
        self._loco_action = torch.zeros((self.count, 29), device=self.device)
        self._target = torch.zeros((self.count, 29), device=self.device)
        self._previous_actor_action = torch.zeros(
            (self.count, self.action_size), device=self.device
        )
        self._previous_joint_velocity = torch.zeros((self.count, 29), device=self.device)
        self._target_one = torch.zeros((self.count, 3), device=self.device)
        self._target_two = torch.zeros_like(self._target_one)
        self._flight_one = torch.zeros(self.count, device=self.device)
        self._flight_two = torch.zeros_like(self._flight_one)
        self._start_one = torch.zeros((self.count, 2), device=self.device)
        self._start_two = torch.zeros_like(self._start_one)
        self._intent_cue_one = torch.zeros((self.count, 3), device=self.device)
        self._intent_cue_two = torch.zeros_like(self._intent_cue_one)
        self._second_enabled = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self._shot_index = torch.zeros(self.count, dtype=torch.long, device=self.device)
        self._quarantined = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self._nonfinite_quarantine_latched = torch.zeros_like(self._quarantined)
        self._failure_pelvis_height = torch.full((self.count,), float("nan"), device=self.device)
        self._failure_upright_projection = torch.full_like(
            self._failure_pelvis_height, float("nan")
        )
        self._failure_root_linear_speed = torch.full_like(self._failure_pelvis_height, float("nan"))
        self._failure_root_angular_speed = torch.full_like(
            self._failure_pelvis_height, float("nan")
        )
        self._failure_step_index = torch.full(
            (self.count,), -1, dtype=torch.long, device=self.device
        )
        self._failure_posture_exception_granted = torch.zeros(
            self.count, dtype=torch.bool, device=self.device
        )
        self._failure_option_age_steps = torch.full_like(self._failure_step_index, -1)
        self._failure_maximum_applied_option_gate = torch.full_like(
            self._failure_pelvis_height, float("nan")
        )
        self._contact_latched = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self._maximum_lateral_displacement = torch.zeros(self.count, device=self.device)
        self._maximum_lateral_speed = torch.zeros(self.count, device=self.device)
        self._ready_left_hand_relative = torch.zeros((self.count, 3), device=self.device)
        self._ready_right_hand_relative = torch.zeros_like(self._ready_left_hand_relative)
        self._previous_left_hand_relative = torch.zeros_like(self._ready_left_hand_relative)
        self._previous_right_hand_relative = torch.zeros_like(self._ready_left_hand_relative)
        self._maximum_hand_displacement = torch.zeros(self.count, device=self.device)
        self._maximum_hand_speed = torch.zeros(self.count, device=self.device)
        self._minimum_hand_target_distance = torch.full((self.count,), 3.0, device=self.device)
        self._first_decisive_state_recorded = torch.zeros(
            self.count, dtype=torch.bool, device=self.device
        )
        self._first_decisive_pelvis_lateral_error = torch.full(
            (self.count,), 3.0, device=self.device
        )
        self._first_decisive_hand_intercept_distance = torch.full(
            (self.count,), 3.0, device=self.device
        )
        self._maximum_root_angular_speed = torch.zeros(self.count, device=self.device)
        self._minimum_upper_body_authority = torch.ones(self.count, device=self.device)
        self._second_release_lateral_error = torch.zeros(self.count, device=self.device)
        self._episode_seed = 0
        self._step_index = 0
        reward_config = GoalkeeperMultiStepConfig(
            control_dt_sec=self.config.control_dt_sec,
            episode_duration_sec=self.config.episode_duration_sec,
            first_shot_release_sec=self.config.first_shot_release_sec,
            second_shot_release_sec=self.config.second_shot_release_sec,
            # Physical joint acceleration is measured in rad/s^2.  Scale the
            # squared term accordingly instead of treating it like an action.
            joint_acceleration_penalty_scale=3.0e-7,
            contact_bonus=5.0,
            hand_contact_bonus=self.config.hand_contact_bonus,
            true_save_bonus=self.config.true_save_bonus,
            hand_save_bonus=self.config.hand_save_bonus,
            second_hand_save_bonus=self.config.second_hand_save_bonus,
            second_save_bonus=40.0,
            recovery_bonus=self.config.recovery_event_bonus,
            reach_reward_scale=self.config.reach_reward_scale,
            reach_reward_semantics=self.config.reach_reward_semantics,
            hard_height_reach_reward_scale=self.config.hard_height_reach_reward_scale,
            hard_height_reach_threshold_m=self.config.hard_height_reach_threshold_m,
            hard_height_reach_distance_decay=self.config.hard_height_reach_distance_decay,
            hard_height_reach_reward_semantics=(self.config.hard_height_reach_reward_semantics),
            bimanual_reach_reward_scale=self.config.bimanual_reach_reward_scale,
            second_shot_reach_reward_multiplier=(self.config.second_shot_reach_reward_multiplier),
            task_motion_reward_scale=self.config.task_motion_reward_scale,
            recovery_progress_reward_scale=self.config.recovery_progress_reward_scale,
            recovery_progress_linear_speed_decay=(self.config.recovery_progress_linear_speed_decay),
            recovery_progress_angular_speed_decay=(
                self.config.recovery_progress_angular_speed_decay
            ),
            root_linear_speed_penalty_scale=self.config.root_linear_speed_penalty_scale,
            root_angular_speed_penalty_scale=self.config.root_angular_speed_penalty_scale,
            root_angular_speed_soft_limit_rad_s=(self.config.root_angular_speed_soft_limit_rad_s),
            root_angular_speed_excess_penalty_scale=(
                self.config.root_angular_speed_excess_penalty_scale
            ),
            flight_root_angular_penalty_scale=(self.config.flight_root_angular_penalty_scale),
            action_magnitude_penalty_scale=self.config.action_magnitude_penalty_scale,
            unsafe_penalty=self.config.unsafe_penalty,
            save_then_unsafe_penalty=self.config.save_then_unsafe_penalty,
        )
        self.task = TorchGoalkeeperMultiStepAccumulator(
            self.count,
            device=self.device,
            config=reward_config,
            validate_each_step=False,
        )

    @property
    def world_steps_per_control_step(self) -> int:
        return self.count * self.config.physics_substeps

    def reset(self, *, seed: int) -> Any:
        """Reset all worlds and sample two independent, causal shots."""

        torch = self.torch
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        self._episode_seed = seed
        self._step_index = 0
        self._shot_index.zero_()
        self._quarantined.zero_()
        self._nonfinite_quarantine_latched.zero_()
        self._failure_pelvis_height.fill_(float("nan"))
        self._failure_upright_projection.fill_(float("nan"))
        self._failure_root_linear_speed.fill_(float("nan"))
        self._failure_root_angular_speed.fill_(float("nan"))
        self._failure_step_index.fill_(-1)
        self._failure_posture_exception_granted.zero_()
        self._failure_option_age_steps.fill_(-1)
        self._failure_maximum_applied_option_gate.fill_(float("nan"))
        self._contact_latched.zero_()
        self.task.reset()
        self._loco_hidden.zero_()
        self._loco_cell.zero_()
        self._loco_action.zero_()
        self._previous_actor_action.zero_()
        self._previous_joint_velocity.zero_()
        self._maximum_lateral_displacement.zero_()
        self._maximum_lateral_speed.zero_()
        self._maximum_hand_displacement.zero_()
        self._maximum_hand_speed.zero_()
        self._minimum_hand_target_distance.fill_(3.0)
        self._first_decisive_state_recorded.zero_()
        self._first_decisive_pelvis_lateral_error.fill_(3.0)
        self._first_decisive_hand_intercept_distance.fill_(3.0)
        self._maximum_root_angular_speed.zero_()
        self._minimum_upper_body_authority.fill_(1.0)
        self._second_release_lateral_error.zero_()
        self._sample_shot(self._target_one, self._flight_one, self._start_one, generator)
        self._sample_shot(self._target_two, self._flight_two, self._start_two, generator)
        self._intent_cue_one.zero_()
        self._intent_cue_two.zero_()
        if self.config.shot_intent_cue_enabled:
            self._sample_intent_cue(self._intent_cue_one, self._target_one, generator)
            self._sample_intent_cue(self._intent_cue_two, self._target_two, generator)
        self._second_enabled.copy_(
            torch.rand(self.count, generator=generator, device=self.device)
            < self.config.second_shot_probability
        )

        self.qpos.zero_()
        self.qvel.zero_()
        self.ctrl.zero_()
        self.qpos[:, 0] = self.config.keeper_x_m
        self.qpos[:, 2] = 0.793
        # Facing the incoming -x direction: 180-degree yaw in wxyz order.
        self.qpos[:, 6] = 1.0
        self._target.zero_()
        self._target[:, self._loco_to_motor] = self._loco_default
        self.qpos[:, 7:36] = self._target
        self._park_ball()
        self.mjw.forward(self.model, self.data)
        left_relative = self.geom_xpos[:, self._left_hand_geom] - self.qpos[:, :3]
        right_relative = self.geom_xpos[:, self._right_hand_geom] - self.qpos[:, :3]
        self._ready_left_hand_relative.copy_(left_relative)
        self._ready_right_hand_relative.copy_(right_relative)
        self._previous_left_hand_relative.copy_(left_relative)
        self._previous_right_hand_relative.copy_(right_relative)
        self._previous_joint_velocity.copy_(self.qvel[:, 6:35])
        self._pre_quarantine_qpos.copy_(self.qpos)
        self._pre_quarantine_qvel.copy_(self.qvel)
        left_foot, right_foot = self._foot_contact_state()
        self._pre_quarantine_left_foot.copy_(left_foot)
        self._pre_quarantine_right_foot.copy_(right_foot)
        return self.observation()

    def _restore_quarantined_worlds(self) -> None:
        """Keep failed batched worlds finite without erasing their failure."""

        mask = self._quarantined
        if not bool(self.torch.any(mask)):
            return
        self.qpos[mask] = 0.0
        self.qpos[mask, 0] = self.config.keeper_x_m
        self.qpos[mask, 2] = 0.793
        self.qpos[mask, 6] = 1.0
        ready = self.torch.zeros(29, dtype=self.qpos.dtype, device=self.device)
        ready[self._loco_to_motor] = self._loco_default
        self.qpos[mask, 7:36] = ready
        self.qpos[mask, 36] = -20.0
        self.qpos[mask, 38] = 0.115
        self.qpos[mask, 39] = 1.0
        self.qvel[mask] = 0.0
        self.ctrl[mask] = 0.0
        self._target[mask] = ready
        self._previous_actor_action[mask] = 0.0
        self._previous_joint_velocity[mask] = 0.0
        self._loco_hidden[:, mask] = 0.0
        self._loco_cell[:, mask] = 0.0
        self._loco_action[mask] = 0.0
        self.mjw.forward(self.model, self.data)

    def observation(self) -> Any:
        """Return the bounded, privileged-free actor observation."""

        torch = self.torch
        root = self.qpos[:, :3]
        ball = self.qpos[:, 36:39]
        estimated_intercept = self._causal_intercept()
        qw, qx, qy, qz = (self.qpos[:, index] for index in range(3, 7))
        gravity = torch.stack(
            (
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ),
            dim=1,
        )
        # Ball velocity and remaining time are causal measurements.  The
        # intercept is estimated from current ball state, not a future trace.
        current_time = self._step_index * self.config.control_dt_sec
        cue_width = 3 if self.config.shot_intent_cue_enabled else 0
        cue = torch.zeros((self.count, cue_width), device=self.device)
        if self.config.shot_intent_cue_enabled:
            first_release = int(
                round(self.config.first_shot_release_sec / self.config.control_dt_sec)
            )
            first_end = int(round(self.config.first_shot_end_sec / self.config.control_dt_sec))
            second_release = int(
                round(self.config.second_shot_release_sec / self.config.control_dt_sec)
            )
            selected_cue = None
            if self._step_index < first_release:
                selected_cue = self._intent_cue_one
            elif first_end <= self._step_index < second_release:
                selected_cue = self._intent_cue_two
            if selected_cue is not None:
                cue[:, 0] = (selected_cue[:, 0] - root[:, 1]) * 0.5
                cue[:, 1] = (selected_cue[:, 1] - root[:, 2]) * 0.5
                cue[:, 2] = selected_cue[:, 2]
        auxiliary_proprioception = self._actor_auxiliary_proprioception()
        phase = torch.zeros((self.count, 3), device=self.device)
        phase[:, 0] = (self._shot_index == 0).to(torch.float32)
        phase[:, 1] = (self._shot_index == 1).to(torch.float32)
        phase[:, 2] = (self._shot_index == 2).to(torch.float32)
        obs = torch.cat(
            (
                (ball - root) * 0.4,
                self.qvel[:, 35:38] * 0.2,
                (estimated_intercept - root) * 0.5,
                gravity,
                self.qvel[:, :3] * 0.5,
                self.qvel[:, 3:6] * 0.25,
                (self.qpos[:, 19:36] - self._target[:, 12:29]),
                self.qvel[:, 18:35] * 0.05,
                self._previous_actor_action,
                auxiliary_proprioception,
                cue,
                phase,
                torch.full(
                    (self.count, 1),
                    current_time / self.config.episode_duration_sec,
                    device=self.device,
                ),
            ),
            dim=1,
        )
        if obs.shape[1] != self.observation_size:
            raise RuntimeError("MJWarp goalkeeper observation contract changed")
        return torch.clamp(obs, -10.0, 10.0)

    def _actor_auxiliary_proprioception(self) -> Any:
        """Return optional causal features inserted before cue/phase clocks."""

        return self.torch.empty((self.count, 0), device=self.device)

    def step(self, actor_action: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Advance one control step and return observation/reward/done/info."""

        torch = self.torch
        if tuple(actor_action.shape) != (self.count, self.action_size):
            raise ValueError("MJWarp goalkeeper actor action has the wrong shape")
        self._apply_timeline_releases()
        requested_action = torch.clamp(actor_action, -1.0, 1.0)
        requested_action, upper_body_authority = self._shape_actor_action(requested_action)
        requested_action = requested_action.clone()
        requested_action[self._quarantined] = 0.0
        arm_action_start_index = int(getattr(self, "arm_action_start_index", 4))
        residual_arm_start_index = int(getattr(self, "residual_arm_start_index", 3))
        requested_action[self._shot_index == 2, arm_action_start_index:] *= (
            self.config.second_shot_arm_authority_scale
        )
        action_step_limit = torch.full_like(requested_action, self.config.maximum_action_step)
        action_step_limit[:, arm_action_start_index:] = self.config.maximum_arm_action_step
        action_filter = torch.full_like(requested_action, self.config.action_filter_fraction)
        action_filter[:, arm_action_start_index:] = self.config.arm_action_filter_fraction
        action_delta = torch.clamp(
            requested_action - self._previous_actor_action,
            -action_step_limit,
            action_step_limit,
        )
        action = self._previous_actor_action + action_filter * action_delta
        pre_ball_velocity = self.qvel[:, 35:38].clone()
        target = self._locomotion_target(action[:, 0])
        residual = action[:, 1:] * self._residual_limits * self.config.residual_scale
        residual[:, residual_arm_start_index:] *= self.config.arm_residual_scale_multiplier
        target[:, self._residual_indices] += residual
        target[:, self._residual_indices] = torch.clamp(
            target[:, self._residual_indices],
            self._residual_joint_lower,
            self._residual_joint_upper,
        )
        contact = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        hand_contact = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        joint_guard_active = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        for _ in range(self.config.physics_substeps):
            control_kp = getattr(self, "_step_kp", self._kp)
            control_kd = getattr(self, "_step_kd", self._kd)
            substep_target = self._substep_position_target(target)
            position_torque = control_kp * (substep_target - self.qpos[:, 7:36])
            substep_authority = self._substep_upper_body_position_authority()
            if substep_authority.ndim == 1:
                substep_authority = substep_authority.unsqueeze(1)
            position_torque[:, 12:] *= substep_authority
            # Preserve velocity damping while shedding the destabilizing
            # position impulse. Scaling their sum would also remove braking.
            torque = position_torque - control_kd * self.qvel[:, 6:35]
            from rosclaw_soccer.training.joint_guard import project_joint_safe_torque_torch

            torque, guard = project_joint_safe_torque_torch(
                joint_position=self.qpos[:, 7:36],
                joint_velocity=self.qvel[:, 6:35],
                commanded_torque=torque,
                joint_ranges=self._joint_ranges,
                limited=self._joint_limited,
            )
            joint_guard_active |= torch.any(guard, dim=1)
            self.ctrl.copy_(torch.clamp(torque, -self._torque_limits, self._torque_limits))
            self.mjw.step(self.model, self.data)
            substep_contact, substep_hand_contact = self._robot_ball_contacts()
            contact |= substep_contact
            hand_contact |= substep_hand_contact
        finite_rows = torch.all(torch.isfinite(self.qpos), dim=1)
        finite_rows &= torch.all(torch.isfinite(self.qvel), dim=1)
        finite_rows &= torch.all(torch.isfinite(self.ctrl), dim=1)
        new_nonfinite = ~finite_rows
        self._nonfinite_quarantine_latched |= new_nonfinite
        self._quarantined |= new_nonfinite
        if bool(torch.any(new_nonfinite)):
            from rosclaw_soccer.training.goalkeeper_multistep import GoalkeeperEpisodePhase

            self.task.phase[new_nonfinite] = int(GoalkeeperEpisodePhase.FAILED)
            contact[new_nonfinite] = False
            hand_contact[new_nonfinite] = False
            self._restore_quarantined_worlds()
        self._target.copy_(target)
        post_ball_velocity = self.qvel[:, 35:38]
        true_save = (
            contact
            & (pre_ball_velocity[:, 0] > 0.25)
            & (post_ball_velocity[:, 0] < 0.65 * pre_ball_velocity[:, 0])
        )
        self._contact_latched |= contact
        qw, qx, qy, qz = (self.qpos[:, index] for index in range(3, 7))
        upright = 2.0 * (qw * qw + qz * qz) - 1.0
        joint_velocity = self.qvel[:, 6:35]
        joint_acceleration = (
            joint_velocity - self._previous_joint_velocity
        ) / self.config.control_dt_sec
        selected_target = torch.where(
            (self._shot_index == 2).unsqueeze(1), self._target_two, self._target_one
        )
        # During recovery, return the reach target to a goalkeeper-ready hand
        # pocket instead of rewarding a frozen post-impact pose.
        mobile_ready = bool(getattr(self, "mobility_option_enabled", False))
        ready_y = (
            torch.clamp(
                self.qpos[:, 1] + 0.28 * self.qvel[:, 1],
                self.config.target_y_range_m[0],
                self.config.target_y_range_m[1],
            )
            if mobile_ready
            else torch.zeros(self.count, device=self.device)
        )
        ready_target = torch.stack(
            (
                self.qpos[:, 0] - 0.03,
                ready_y,
                torch.full((self.count,), 0.82, device=self.device),
            ),
            dim=1,
        )
        selected_target = torch.where(
            (self._shot_index == 0).unsqueeze(1), ready_target, selected_target
        )
        time = torch.full(
            (self.count,),
            (self._step_index + 1) * self.config.control_dt_sec,
            device=self.device,
        )
        event_shot_index = self._shot_index.clone()
        posture_exception_granted = self._posture_exception_granted(upright)
        sample = {
            "time_sec": time,
            "ball_position_m": self.qpos[:, 36:39],
            "ball_velocity_mps": post_ball_velocity,
            "intercept_target_m": selected_target,
            "left_hand_position_m": self.geom_xpos[:, self._left_hand_geom],
            "right_hand_position_m": self.geom_xpos[:, self._right_hand_geom],
            "pelvis_height_m": self.qpos[:, 2],
            "pelvis_position_m": self.qpos[:, :3],
            "root_linear_velocity_mps": self.qvel[:, :3],
            "root_angular_velocity_rad_s": self.qvel[:, 3:6],
            "upright_projection": upright,
            "posture_exception_granted": posture_exception_granted,
            "action": action,
            "previous_action": self._previous_actor_action,
            "joint_acceleration_rad_s2": joint_acceleration,
            "applied_torque_nm": self.ctrl,
            "ball_contact": contact,
            "hand_contact": hand_contact,
            "true_save": true_save,
            "shot_index": event_shot_index,
        }
        result = self.task.step(sample)
        result["terminated"] |= new_nonfinite
        result["nonfinite_override"] = torch.where(
            new_nonfinite,
            -self.task.config.unsafe_penalty - result["total"],
            torch.zeros_like(result["total"]),
        )
        result["total"] = torch.where(
            new_nonfinite,
            torch.full_like(result["total"], -self.task.config.unsafe_penalty),
            result["total"],
        )
        first_arrival_time = self.config.first_shot_release_sec + self._flight_one
        decisive_now = (
            (event_shot_index == 1)
            & ~self._first_decisive_state_recorded
            & (true_save | (time >= first_arrival_time))
        )
        decisive_intercept = torch.where(
            true_save.unsqueeze(1),
            self.qpos[:, 36:39],
            self._target_one,
        )
        decisive_hand_distance = torch.minimum(
            torch.linalg.vector_norm(
                self.geom_xpos[:, self._left_hand_geom] - decisive_intercept,
                dim=1,
            ),
            torch.linalg.vector_norm(
                self.geom_xpos[:, self._right_hand_geom] - decisive_intercept,
                dim=1,
            ),
        )
        self._first_decisive_pelvis_lateral_error.copy_(
            torch.where(
                decisive_now,
                torch.abs(decisive_intercept[:, 1] - self.qpos[:, 1]),
                self._first_decisive_pelvis_lateral_error,
            )
        )
        self._first_decisive_hand_intercept_distance.copy_(
            torch.where(
                decisive_now,
                decisive_hand_distance,
                self._first_decisive_hand_intercept_distance,
            )
        )
        self._first_decisive_state_recorded |= decisive_now
        # A real save ends the reach phase immediately.  Leaving the actor in
        # flight mode until a fixed timeout teaches it to keep extending after
        # impact, which creates the visible twist and delays recovery.
        self._shot_index[true_save] = 0
        self._previous_actor_action.copy_(action)
        self._previous_joint_velocity.copy_(joint_velocity)
        self._maximum_lateral_displacement.copy_(
            torch.maximum(self._maximum_lateral_displacement, torch.abs(self.qpos[:, 1]))
        )
        self._maximum_lateral_speed.copy_(
            torch.maximum(self._maximum_lateral_speed, torch.abs(self.qvel[:, 1]))
        )
        left_hand_relative = self.geom_xpos[:, self._left_hand_geom] - self.qpos[:, :3]
        right_hand_relative = self.geom_xpos[:, self._right_hand_geom] - self.qpos[:, :3]
        hand_displacement = torch.maximum(
            torch.linalg.vector_norm(
                left_hand_relative - self._ready_left_hand_relative,
                dim=1,
            ),
            torch.linalg.vector_norm(
                right_hand_relative - self._ready_right_hand_relative,
                dim=1,
            ),
        )
        hand_speed = (
            torch.maximum(
                torch.linalg.vector_norm(
                    left_hand_relative - self._previous_left_hand_relative,
                    dim=1,
                ),
                torch.linalg.vector_norm(
                    right_hand_relative - self._previous_right_hand_relative,
                    dim=1,
                ),
            )
            / self.config.control_dt_sec
        )
        hand_target_distance = torch.minimum(
            torch.linalg.vector_norm(
                self.geom_xpos[:, self._left_hand_geom] - selected_target,
                dim=1,
            ),
            torch.linalg.vector_norm(
                self.geom_xpos[:, self._right_hand_geom] - selected_target,
                dim=1,
            ),
        )
        active_shot = (event_shot_index > 0) & ~self._quarantined
        self._minimum_hand_target_distance.copy_(
            torch.where(
                active_shot,
                torch.minimum(self._minimum_hand_target_distance, hand_target_distance),
                self._minimum_hand_target_distance,
            )
        )
        self._maximum_hand_displacement.copy_(
            torch.maximum(self._maximum_hand_displacement, hand_displacement)
        )
        self._maximum_hand_speed.copy_(torch.maximum(self._maximum_hand_speed, hand_speed))
        self._previous_left_hand_relative.copy_(left_hand_relative)
        self._previous_right_hand_relative.copy_(right_hand_relative)
        self._maximum_root_angular_speed.copy_(
            torch.maximum(
                self._maximum_root_angular_speed,
                torch.linalg.vector_norm(self.qvel[:, 3:6], dim=1),
            )
        )
        self._minimum_upper_body_authority.copy_(
            torch.minimum(self._minimum_upper_body_authority, upper_body_authority)
        )
        failed_now = (self.task.phase == 7) & ~self._quarantined
        self._failure_pelvis_height.copy_(
            torch.where(failed_now, self.qpos[:, 2], self._failure_pelvis_height)
        )
        self._failure_upright_projection.copy_(
            torch.where(failed_now, upright, self._failure_upright_projection)
        )
        self._failure_root_linear_speed.copy_(
            torch.where(
                failed_now,
                torch.linalg.vector_norm(self.qvel[:, :3], dim=1),
                self._failure_root_linear_speed,
            )
        )
        self._failure_root_angular_speed.copy_(
            torch.where(
                failed_now,
                torch.linalg.vector_norm(self.qvel[:, 3:6], dim=1),
                self._failure_root_angular_speed,
            )
        )
        self._failure_step_index.copy_(
            torch.where(
                failed_now,
                torch.full_like(self._failure_step_index, self._step_index + 1),
                self._failure_step_index,
            )
        )
        self._failure_posture_exception_granted.copy_(
            torch.where(
                failed_now,
                posture_exception_granted,
                self._failure_posture_exception_granted,
            )
        )
        option_age_steps = getattr(self, "_option_age_steps", self._failure_option_age_steps)
        maximum_option_gate = getattr(
            self,
            "_maximum_applied_option_gate",
            self._failure_maximum_applied_option_gate,
        )
        self._failure_option_age_steps.copy_(
            torch.where(failed_now, option_age_steps, self._failure_option_age_steps)
        )
        self._failure_maximum_applied_option_gate.copy_(
            torch.where(
                failed_now,
                maximum_option_gate,
                self._failure_maximum_applied_option_gate,
            )
        )
        self._pre_quarantine_qpos.copy_(self.qpos)
        self._pre_quarantine_qvel.copy_(self.qvel)
        left_foot, right_foot = self._foot_contact_state()
        self._pre_quarantine_left_foot.copy_(left_foot)
        self._pre_quarantine_right_foot.copy_(right_foot)
        self._quarantined |= self.task.phase == 7
        self._restore_quarantined_worlds()
        self._step_index += 1
        info = {
            **result,
            "requested_action": requested_action.clone(),
            "applied_action": action.clone(),
            "joint_guard_active": joint_guard_active,
            "ball_contact": contact,
            "hand_contact": hand_contact,
            "true_save": true_save,
            "event_shot_index": event_shot_index,
            "pelvis_height_m": self.qpos[:, 2].clone(),
            "upright_projection": upright.clone(),
            "target_m": selected_target.clone(),
            "upper_body_authority": upper_body_authority.clone(),
            "maximum_lateral_displacement_m": self._maximum_lateral_displacement.clone(),
            "maximum_lateral_speed_mps": self._maximum_lateral_speed.clone(),
            "maximum_hand_displacement_m": self._maximum_hand_displacement.clone(),
            "maximum_hand_speed_mps": self._maximum_hand_speed.clone(),
            "minimum_hand_target_distance_m": self._minimum_hand_target_distance.clone(),
            "first_decisive_pelvis_lateral_error_m": (
                self._first_decisive_pelvis_lateral_error.clone()
            ),
            "first_decisive_hand_intercept_distance_m": (
                self._first_decisive_hand_intercept_distance.clone()
            ),
            "maximum_root_angular_speed_rad_s": self._maximum_root_angular_speed.clone(),
            "second_release_lateral_error_m": self._second_release_lateral_error.clone(),
            "quarantined": self._quarantined.clone(),
            "nonfinite_quarantined": self._nonfinite_quarantine_latched.clone(),
            "failure_pelvis_height_m": self._failure_pelvis_height.clone(),
            "failure_upright_projection": self._failure_upright_projection.clone(),
            "failure_root_linear_speed_mps": self._failure_root_linear_speed.clone(),
            "failure_root_angular_speed_rad_s": self._failure_root_angular_speed.clone(),
            "failure_step_index": self._failure_step_index.clone(),
            "failure_posture_exception_granted": (self._failure_posture_exception_granted.clone()),
            "failure_option_age_steps": self._failure_option_age_steps.clone(),
            "failure_maximum_applied_option_gate": (
                self._failure_maximum_applied_option_gate.clone()
            ),
        }
        return self.observation(), result["total"], result["terminated"], info

    def _posture_exception_granted(self, upright_projection: Any) -> Any:
        """Environment-owned hook for explicitly bounded dynamic skills."""

        del upright_projection
        return self.torch.zeros(self.count, dtype=self.torch.bool, device=self.device)

    def _shape_actor_action(self, requested_action: Any) -> tuple[Any, Any]:
        """Apply the default lateral/upper-body agility guard."""

        return shape_goalkeeper_action_torch(
            requested_action=requested_action,
            root_lateral_position_m=self.qpos[:, 1],
            root_lateral_velocity_mps=self.qvel[:, 1],
            root_angular_velocity_rad_s=self.qvel[:, 3:6],
            shot_active=self._shot_index > 0,
            config=self.config.agility,
        )

    def _substep_upper_body_position_authority(self) -> Any:
        """Return a 2 ms position-gain guard while preserving velocity damping."""

        if not bool(getattr(self, "mobility_option_enabled", False)):
            return self.torch.ones(self.count, device=self.device)
        from rosclaw_soccer.training.goalkeeper_mobility_option import (
            GoalkeeperMobilityOptionConfig,
            substep_upper_body_authority_torch,
        )

        return substep_upper_body_authority_torch(
            root_angular_velocity_rad_s=self.qvel[:, 3:6],
            config=getattr(
                self,
                "mobility_option_config",
                GoalkeeperMobilityOptionConfig(),
            ),
        )

    def _substep_position_target(self, target: Any) -> Any:
        """Return the position target used by this physics substep."""

        return target

    def finite_state(self) -> bool:
        torch = self.torch
        return bool(
            torch.all(torch.isfinite(self.qpos))
            & torch.all(torch.isfinite(self.qvel))
            & torch.all(torch.isfinite(self.ctrl))
        )

    def recovery_snapshot_state(self) -> tuple[Any, Any, Any, Any]:
        """Return the last finite pre-quarantine state for offline evidence.

        Returned tensors are clones so an asynchronous corpus writer cannot
        observe the next simulator step through the zero-copy Warp views.
        This is read-only evidence access and grants no control authority.
        """

        return (
            self._pre_quarantine_qpos.clone(),
            self._pre_quarantine_qvel.clone(),
            self._pre_quarantine_left_foot.clone(),
            self._pre_quarantine_right_foot.clone(),
        )

    def summary(self) -> dict[str, Any]:
        maximum_root_angular_speed = self._maximum_root_angular_speed
        root_angular_soft_limit = self.config.root_angular_speed_soft_limit_rad_s
        return {
            **self.task.summary(),
            "schema_version": "rosclaw_soccer.goalkeeper_mjwarp_summary.v6",
            "physics_backend": "mujoco_warp",
            "physics_model": "qualified_g1_native_mujoco",
            "config_hash": self.config.config_hash,
            "world_steps": self._step_index * self.world_steps_per_control_step,
            "finite_state": self.finite_state(),
            "lower_body_authority": self.lower_body_authority,
            "learned_residual_authority": self.learned_residual_authority,
            "mean_maximum_lateral_displacement_m": float(self._maximum_lateral_displacement.mean()),
            "mean_maximum_lateral_speed_mps": float(self._maximum_lateral_speed.mean()),
            "mean_maximum_hand_displacement_m": float(self._maximum_hand_displacement.mean()),
            "mean_maximum_hand_speed_mps": float(self._maximum_hand_speed.mean()),
            "mean_minimum_hand_target_distance_m": float(self._minimum_hand_target_distance.mean()),
            "mean_first_decisive_pelvis_lateral_error_m": float(
                self._first_decisive_pelvis_lateral_error.mean()
            ),
            "mean_first_decisive_hand_intercept_distance_m": float(
                self._first_decisive_hand_intercept_distance.mean()
            ),
            **_root_angular_speed_tail_summary(
                torch=self.torch,
                maximum_speeds=maximum_root_angular_speed,
                soft_limit_rad_s=root_angular_soft_limit,
            ),
            "mean_minimum_upper_body_authority": float(self._minimum_upper_body_authority.mean()),
            "mean_second_release_lateral_error_m": float(self._second_release_lateral_error.mean()),
            "quarantined_rate": float(self._quarantined.to(self.torch.float32).mean()),
            "nonfinite_quarantine_rate": float(
                self._nonfinite_quarantine_latched.to(self.torch.float32).mean()
            ),
            "promotion_status": "CANDIDATE_GENERATOR_NOT_PROMOTION_AUTHORITY",
            "activation_ceiling": "SIM_ONLY",
        }

    def _sample_shot(self, target: Any, flight: Any, start: Any, generator: Any) -> None:
        torch = self.torch
        target[:, 0] = self.config.keeper_x_m - 0.08
        target[:, 1] = self._uniform(self.config.target_y_range_m, generator)
        target[:, 2] = self._uniform(self.config.target_z_range_m, generator)
        flight.copy_(self._uniform(self.config.flight_time_range_sec, generator))
        hard = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        if self.config.hard_shot_fraction > 0.0:
            hard = (
                torch.rand(self.count, generator=generator, device=self.device)
                < self.config.hard_shot_fraction
            )
            far_min = min(0.72, self.config.target_y_range_m[1] - 0.02)
            far_magnitude = self._uniform((far_min, self.config.target_y_range_m[1]), generator)
            far_sign = torch.where(
                torch.rand(self.count, generator=generator, device=self.device) < 0.5,
                -torch.ones(self.count, device=self.device),
                torch.ones(self.count, device=self.device),
            )
            if self.config.hard_shot_side_mode == "negative":
                far_sign.fill_(-1.0)
            elif self.config.hard_shot_side_mode == "positive":
                far_sign.fill_(1.0)
            high_min = min(1.10, self.config.target_z_range_m[1] - 0.02)
            hard_height = self._uniform((high_min, self.config.target_z_range_m[1]), generator)
            low_high = min(0.60, self.config.target_z_range_m[1] - 0.04)
            mid_low = max(0.60, self.config.target_z_range_m[0] + 0.02)
            mid_high = min(1.10, self.config.target_z_range_m[1] - 0.02)
            low = self._uniform((self.config.target_z_range_m[0], low_high), generator)
            mid = self._uniform((mid_low, mid_high), generator)
            if self.config.hard_shot_height_mode == "low":
                hard_height = low
            elif self.config.hard_shot_height_mode == "mid":
                hard_height = mid
            elif self.config.hard_shot_height_mode == "balanced":
                height_band = torch.randint(
                    0,
                    3,
                    (self.count,),
                    generator=generator,
                    device=self.device,
                )
                hard_height = torch.where(height_band == 0, low, hard_height)
                hard_height = torch.where(height_band == 1, mid, hard_height)
            target[:, 1] = torch.where(hard, far_sign * far_magnitude, target[:, 1])
            target[:, 2] = torch.where(hard, hard_height, target[:, 2])
        if self.config.hard_shot_flight_time_range_sec is not None:
            hard_flight = self._uniform(self.config.hard_shot_flight_time_range_sec, generator)
            flight.copy_(torch.where(hard, hard_flight, flight))
        start[:, 0] = self._uniform(self.config.ball_start_x_range_m, generator)
        start[:, 1] = self._uniform(self.config.ball_start_z_range_m, generator)

    def _sample_intent_cue(self, cue: Any, target: Any, generator: Any) -> None:
        """Sample a noisy, droppable shooter-telegraph belief, never an exact target."""

        torch = self.torch
        cue[:, 0] = torch.clamp(
            target[:, 1]
            + self.config.shot_intent_cue_lateral_noise_m
            * torch.randn(self.count, generator=generator, device=self.device),
            self.config.target_y_range_m[0],
            self.config.target_y_range_m[1],
        )
        cue[:, 1] = torch.clamp(
            target[:, 2]
            + self.config.shot_intent_cue_height_noise_m
            * torch.randn(self.count, generator=generator, device=self.device),
            self.config.target_z_range_m[0],
            self.config.target_z_range_m[1],
        )
        cue[:, 2] = (
            torch.rand(self.count, generator=generator, device=self.device)
            >= self.config.shot_intent_cue_dropout_probability
        ).to(torch.float32)
        cue[:, :2] *= cue[:, 2:3]

    def _uniform(self, limits: tuple[float, float], generator: Any) -> Any:
        torch = self.torch
        return limits[0] + (limits[1] - limits[0]) * torch.rand(
            self.count, generator=generator, device=self.device
        )

    def _park_ball(self) -> None:
        self.qpos[:, 36] = -20.0
        self.qpos[:, 37] = 0.0
        self.qpos[:, 38] = 0.115
        self.qpos[:, 39] = 1.0
        self.qpos[:, 40:43] = 0.0
        self.qvel[:, 35:41] = 0.0

    def _launch(self, target: Any, flight: Any, start: Any, enabled: Any) -> None:
        start_x = start[:, 0]
        start_z = start[:, 1]
        self.qpos[enabled, 36] = start_x[enabled]
        # Offset launch y slightly from target to avoid a one-dimensional task.
        self.qpos[enabled, 37] = (0.18 * target[:, 1])[enabled]
        self.qpos[enabled, 38] = start_z[enabled]
        self.qpos[enabled, 39] = 1.0
        self.qpos[enabled, 40:43] = 0.0
        self.qvel[enabled, 35] = ((target[:, 0] - start_x) / flight)[enabled]
        self.qvel[enabled, 36] = ((target[:, 1] - self.qpos[:, 37]) / flight)[enabled]
        self.qvel[enabled, 37] = ((target[:, 2] - start_z + 0.5 * 9.81 * flight * flight) / flight)[
            enabled
        ]
        self.qvel[enabled, 38:41] = 0.0
        self.mjw.forward(self.model, self.data)

    def _apply_timeline_releases(self) -> None:
        first_release = int(round(self.config.first_shot_release_sec / self.config.control_dt_sec))
        first_end = int(round(self.config.first_shot_end_sec / self.config.control_dt_sec))
        second_release = int(
            round(self.config.second_shot_release_sec / self.config.control_dt_sec)
        )
        second_end = int(round(self.config.second_shot_end_sec / self.config.control_dt_sec))
        if self._step_index < first_release:
            self._park_ball()
            self.mjw.forward(self.model, self.data)
        elif self._step_index == first_release:
            self._shot_index.fill_(1)
            enabled = self.torch.ones(self.count, dtype=self.torch.bool, device=self.device)
            self._launch(self._target_one, self._flight_one, self._start_one, enabled)
        elif self._step_index == first_end:
            self._shot_index.zero_()
            self._park_ball()
            self.mjw.forward(self.model, self.data)
        elif self._step_index == second_release:
            self._second_release_lateral_error.copy_(self.torch.abs(self.qpos[:, 1]))
            self._shot_index.copy_(self._second_enabled.to(self.torch.long) * 2)
            self._park_ball()
            self._launch(
                self._target_two,
                self._flight_two,
                self._start_two,
                self._second_enabled,
            )
        elif self._step_index == second_end:
            self._shot_index.zero_()
            self._park_ball()
            self.mjw.forward(self.model, self.data)

    def _locomotion_target(self, lateral_action: Any) -> Any:
        torch = self.torch
        qw, qx, qy, qz = (self.qpos[:, index] for index in range(3, 7))
        gravity = torch.stack(
            (
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ),
            dim=1,
        )
        obs = torch.zeros((self.count, 96), device=self.device)
        obs[:, :3] = self.qvel[:, 3:6]
        obs[:, 3:6] = gravity
        # Keeper yaw is pi, so local lateral +y maps to world -y.
        obs[:, 7] = lateral_action * self.config.maximum_lateral_command_mps
        obs[:, 9:38] = self.qpos[:, 7:36][:, self._loco_to_motor] - self._loco_default
        obs[:, 38:67] = self.qvel[:, 6:35][:, self._loco_to_motor]
        obs[:, 67:96] = self._loco_action
        with torch.inference_mode():
            encoded = self._loco_policy.normalizer.forward(torch.clamp(obs, -100.0, 100.0))
            sequence, (hidden, cell) = self._loco_rnn(
                encoded.unsqueeze(0), (self._loco_hidden, self._loco_cell)
            )
            self._loco_hidden.copy_(hidden)
            self._loco_cell.copy_(cell)
            self._loco_action.copy_(
                torch.clamp(self._loco_policy.actor.forward(sequence.squeeze(0)), -100.0, 100.0)
            )
        target = torch.zeros((self.count, 29), device=self.device)
        target[:, self._loco_to_motor] = 0.25 * self._loco_action + self._loco_default
        return target

    def _causal_intercept(self) -> Any:
        """Estimate a goal-line intercept from current ball pose and velocity."""

        torch = self.torch
        ball = self.qpos[:, 36:39]
        velocity = self.qvel[:, 35:38]
        active = (self._shot_index > 0) & (velocity[:, 0] > 0.10)
        time_to_line = torch.clamp(
            (self.config.keeper_x_m - 0.08 - ball[:, 0]) / torch.clamp(velocity[:, 0], min=0.10),
            0.0,
            1.2,
        )
        intercept = torch.stack(
            (
                torch.full((self.count,), self.config.keeper_x_m - 0.08, device=self.device),
                ball[:, 1] + velocity[:, 1] * time_to_line,
                ball[:, 2]
                + velocity[:, 2] * time_to_line
                - 0.5 * 9.81 * time_to_line * time_to_line,
            ),
            dim=1,
        )
        intercept[:, 1] = torch.clamp(
            intercept[:, 1], self.config.target_y_range_m[0], self.config.target_y_range_m[1]
        )
        intercept[:, 2] = torch.clamp(
            intercept[:, 2], self.config.target_z_range_m[0], self.config.target_z_range_m[1]
        )
        # During recovery expose displacement from the goal centre.  The old
        # ready target copied the current root y coordinate, making the error
        # identically zero and preventing a causal actor from learning to
        # recenter between consecutive shots.
        mobile_ready = bool(getattr(self, "mobility_option_enabled", False))
        ready_y = (
            torch.clamp(
                self.qpos[:, 1] + 0.28 * self.qvel[:, 1],
                self.config.target_y_range_m[0],
                self.config.target_y_range_m[1],
            )
            if mobile_ready
            else torch.zeros(self.count, device=self.device)
        )
        ready_z = torch.full((self.count,), 0.82, device=self.device)
        if self.config.shot_intent_cue_enabled:
            first_release = int(
                round(self.config.first_shot_release_sec / self.config.control_dt_sec)
            )
            first_end = int(round(self.config.first_shot_end_sec / self.config.control_dt_sec))
            second_release = int(
                round(self.config.second_shot_release_sec / self.config.control_dt_sec)
            )
            selected_cue = None
            if self._step_index < first_release:
                selected_cue = self._intent_cue_one
            elif first_end <= self._step_index < second_release:
                selected_cue = self._intent_cue_two
            if selected_cue is not None:
                visible = selected_cue[:, 2] > 0.5
                ready_y = torch.where(visible, selected_cue[:, 0], ready_y)
                ready_z = torch.where(visible, selected_cue[:, 1], ready_z)
        ready = torch.stack(
            (
                torch.full((self.count,), self.config.keeper_x_m - 0.08, device=self.device),
                ready_y,
                ready_z,
            ),
            dim=1,
        )
        return torch.where(active.unsqueeze(1), intercept, ready)

    def _robot_ball_contacts(self) -> tuple[Any, Any]:
        """Classify ball contacts as any-body and hand/glove-only per world."""

        torch = self.torch
        geom = self.contact_geom
        g0 = geom[:, 0].to(torch.long)
        g1 = geom[:, 1].to(torch.long)
        valid_index = (
            (g0 >= 0) & (g0 < self.cpu_model.ngeom) & (g1 >= 0) & (g1 < self.cpu_model.ngeom)
        )
        safe_g0 = torch.clamp(g0, 0, self.cpu_model.ngeom - 1)
        safe_g1 = torch.clamp(g1, 0, self.cpu_model.ngeom - 1)
        body0 = self._geom_body[safe_g0]
        body1 = self._geom_body[safe_g1]
        ball0 = body0 == self._ball_body
        ball1 = body1 == self._ball_body
        robot0 = (body0 >= 1) & (body0 < self._ball_body)
        robot1 = (body1 >= 1) & (body1 < self._ball_body)
        active = (
            valid_index & (self.contact_distance <= 0.002) & ((ball0 & robot1) | (ball1 & robot0))
        )
        hand0 = torch.any(safe_g0.unsqueeze(1) == self._hand_geoms.unsqueeze(0), dim=1)
        hand1 = torch.any(safe_g1.unsqueeze(1) == self._hand_geoms.unsqueeze(0), dim=1)
        hand_active = active & ((ball0 & hand1) | (ball1 & hand0))
        world = torch.clamp(self.contact_world.to(torch.long), 0, self.count - 1)
        counts = torch.zeros(self.count, dtype=torch.float32, device=self.device)
        counts.scatter_add_(0, world, active.to(torch.float32))
        hand_counts = torch.zeros_like(counts)
        hand_counts.scatter_add_(0, world, hand_active.to(torch.float32))
        return counts > 0.0, hand_counts > 0.0

    def _foot_contact_state(self) -> tuple[Any, Any]:
        """Return causal left/right ground-contact flags.

        The targeted-dive decoder was trained with ``right - left`` support
        semantics, but the batched runtime previously hard-coded zero.  That
        erased the distinction between the stance and recovery leg exactly
        when the controller had to absorb a lateral impulse.  Ground contacts
        are public proprioception; no future ball or privileged target state
        enters this signal.
        """

        torch = self.torch
        geom = self.contact_geom
        g0 = geom[:, 0].to(torch.long)
        g1 = geom[:, 1].to(torch.long)
        valid = (
            (g0 >= 0)
            & (g0 < self.cpu_model.ngeom)
            & (g1 >= 0)
            & (g1 < self.cpu_model.ngeom)
            & (self.contact_distance <= 0.002)
        )
        safe_g0 = torch.clamp(g0, 0, self.cpu_model.ngeom - 1)
        safe_g1 = torch.clamp(g1, 0, self.cpu_model.ngeom - 1)
        body0 = self._geom_body[safe_g0]
        body1 = self._geom_body[safe_g1]
        ground0 = body0 == 0
        ground1 = body1 == 0
        left = valid & (
            (ground0 & (body1 == self._left_foot_body))
            | (ground1 & (body0 == self._left_foot_body))
        )
        right = valid & (
            (ground0 & (body1 == self._right_foot_body))
            | (ground1 & (body0 == self._right_foot_body))
        )
        world = torch.clamp(self.contact_world.to(torch.long), 0, self.count - 1)
        left_count = torch.zeros(self.count, dtype=torch.float32, device=self.device)
        right_count = torch.zeros_like(left_count)
        left_count.scatter_add_(0, world, left.to(torch.float32))
        right_count.scatter_add_(0, world, right.to(torch.float32))
        return left_count > 0.0, right_count > 0.0

    def _foot_support_side(self) -> Any:
        """Return decoder-compatible ``right - left`` support semantics."""

        left_contact, right_contact = self._foot_contact_state()
        return right_contact.to(self.torch.float32) - left_contact.to(self.torch.float32)


__all__ = ["GoalkeeperMJWarpBatch", "GoalkeeperMJWarpConfig"]
