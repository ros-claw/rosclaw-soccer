"""Target-conditioned whole-body dive option for the G1 goalkeeper.

The phase-only decoder is useful as a motion-compression experiment, but it
cannot decide *where* or *when* to intercept a shot.  This module keeps the
qualified bilateral motion as an imitation anchor while exposing the missing
task variables to a compact neural option.  The resulting checkpoint remains
train-only, non-commercial and ``SIM_ONLY``; an environment-owned monitor,
not the network, owns activation, phase progression and safety failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_dive_option import (
    GoalkeeperDiveOptionConfig,
    GoalkeeperDivePhase,
    build_balanced_dive_imitation_seed,
    load_official_goalkeeper_dive_atlas,
)
from rosclaw_soccer.training.goalkeeper_mjwarp import _LOCO_DEFAULT, _LOCO_TO_MOTOR
from rosclaw_soccer.training.goalkeeper_reach import (
    GoalkeeperReachAtlasConfig,
    GoalkeeperReachConfig,
    build_g1_task_space_reach_atlas,
    task_space_reach_from_target_numpy,
)

TARGETED_DIVE_INPUT_SIZE = 16
TARGETED_DIVE_OUTPUT_SIZE = 29
TARGETED_DIVE_ANCHOR_FRAMES = 63


@dataclass(frozen=True)
class GoalkeeperTargetedDiveConfig:
    """Content-bound imitation curriculum for a task-conditioned option."""

    hidden_size: int = 192
    samples_per_epoch: int = 32_768
    epochs: int = 240
    minibatch_size: int = 1_024
    learning_rate: float = 1.5e-3
    weight_decay: float = 1.0e-5
    velocity_loss_weight: float = 0.08
    anchor_loss_weight: float = 0.005
    symmetry_loss_weight: float = 0.10
    tail_loss_weight: float = 0.30
    hard_mining_epochs: int = 200
    hard_mining_samples: int = 8_192
    hard_mining_learning_rate_scale: float = 0.10
    hard_mining_refresh_epochs: int = 20
    hard_mining_general_replay_fraction: float = 0.50
    failure_replay_fraction: float = 0.25
    low_height_crouch_scale: float = 1.0
    low_height_max_hip_flexion_delta_rad: float = 0.38
    low_height_max_knee_flexion_delta_rad: float = 0.70
    low_height_max_ankle_dorsiflexion_delta_rad: float = 0.30
    low_height_max_waist_roll_delta_rad: float = 0.30
    low_height_max_waist_pitch_delta_rad: float = 0.18
    low_height_crouch_start_lateral_progress: float = 0.35
    low_height_crouch_full_lateral_progress: float = 0.70
    recovery_tail_start_phase: float | None = None
    high_height_extension_scale: float = 1.0
    maximum_position_rmse_rad: float = 0.045
    maximum_absolute_error_rad: float = 0.22
    maximum_symmetry_rmse_rad: float = 0.030
    random_seed: int = 42_107
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_targeted_dive_config.v16"

    def __post_init__(self) -> None:
        if not 64 <= self.hidden_size <= 512:
            raise ValueError("targeted dive hidden size is invalid")
        if not 4_096 <= self.samples_per_epoch <= 1_048_576:
            raise ValueError("targeted dive sample count is invalid")
        if self.samples_per_epoch % 2:
            raise ValueError("targeted dive sample count must be even for mirrored pairs")
        if not 10 <= self.epochs <= 20_000:
            raise ValueError("targeted dive epoch count is invalid")
        if not 0 <= self.hard_mining_epochs <= 2_000:
            raise ValueError("targeted dive hard-mining epoch count is invalid")
        if not 512 <= self.hard_mining_samples <= 1_048_576:
            raise ValueError("targeted dive hard-mining sample count is invalid")
        if not 1 <= self.hard_mining_refresh_epochs <= 200:
            raise ValueError("targeted dive hard-mining refresh interval is invalid")
        if not 128 <= self.minibatch_size <= self.samples_per_epoch:
            raise ValueError("targeted dive minibatch size is invalid")
        values = (
            self.learning_rate,
            self.velocity_loss_weight,
            self.anchor_loss_weight,
            self.symmetry_loss_weight,
            self.tail_loss_weight,
            self.hard_mining_learning_rate_scale,
            self.maximum_position_rmse_rad,
            self.maximum_absolute_error_rad,
            self.maximum_symmetry_rmse_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("targeted dive settings must be finite and positive")
        if not math.isfinite(self.weight_decay) or not 0.0 <= self.weight_decay <= 0.01:
            raise ValueError("targeted dive weight decay is invalid")
        if not math.isfinite(self.failure_replay_fraction) or not (
            0.0 <= self.failure_replay_fraction <= 0.50
        ):
            raise ValueError("targeted dive failure replay fraction is invalid")
        if not (
            math.isfinite(self.low_height_crouch_scale)
            and 0.0 <= self.low_height_crouch_scale <= 1.50
            and math.isfinite(self.high_height_extension_scale)
            and 0.0 <= self.high_height_extension_scale <= 1.50
        ):
            raise ValueError("targeted dive height-conditioned posture scale is invalid")
        crouch_deltas = (
            self.low_height_max_hip_flexion_delta_rad,
            self.low_height_max_knee_flexion_delta_rad,
            self.low_height_max_ankle_dorsiflexion_delta_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in crouch_deltas) or (
            self.low_height_max_hip_flexion_delta_rad > 0.45
            or self.low_height_max_knee_flexion_delta_rad > 0.85
            or self.low_height_max_ankle_dorsiflexion_delta_rad > 0.40
        ):
            raise ValueError("targeted dive low-height crouch joint delta is invalid")
        torso_deltas = (
            self.low_height_max_waist_roll_delta_rad,
            self.low_height_max_waist_pitch_delta_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in torso_deltas) or any(
            value > 0.35 for value in torso_deltas
        ):
            raise ValueError("targeted dive low-height torso joint delta is invalid")
        if not (
            math.isfinite(self.low_height_crouch_start_lateral_progress)
            and math.isfinite(self.low_height_crouch_full_lateral_progress)
            and 0.0
            <= self.low_height_crouch_start_lateral_progress
            < self.low_height_crouch_full_lateral_progress
            <= 1.0
        ):
            raise ValueError("targeted dive low-height crouch lateral progress is invalid")
        if self.recovery_tail_start_phase is not None and not (
            math.isfinite(self.recovery_tail_start_phase)
            and 0.55 <= self.recovery_tail_start_phase <= 0.90
        ):
            raise ValueError("targeted dive recovery-tail phase is invalid")
        if self.hard_mining_learning_rate_scale > 1.0:
            raise ValueError("targeted dive hard-mining learning rate scale is invalid")
        if not math.isfinite(self.hard_mining_general_replay_fraction) or not (
            0.0 <= self.hard_mining_general_replay_fraction <= 0.90
        ):
            raise ValueError("targeted dive hard-mining general replay fraction is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("targeted dive option must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _ready_pose() -> NDArray[np.float64]:
    ready = np.zeros(29, dtype=np.float64)
    ready[np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)] = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    return ready


def targeted_dive_features_numpy(
    *,
    direction: NDArray[np.float64],
    phase: NDArray[np.float64],
    target_lateral_m: NDArray[np.float64],
    target_height_m: NDArray[np.float64],
    time_to_arrival_sec: NDArray[np.float64],
    root_lateral_m: NDArray[np.float64],
    root_lateral_speed_mps: NDArray[np.float64],
    pelvis_height_m: NDArray[np.float64],
    upright_projection: NDArray[np.float64],
    support_side: NDArray[np.float64],
    root_angular_speed_rad_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build normalized features shared by training, CUDA and CPU replay."""

    raw = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            direction,
            phase,
            target_lateral_m,
            target_height_m,
            time_to_arrival_sec,
            root_lateral_m,
            root_lateral_speed_mps,
            pelvis_height_m,
            upright_projection,
            support_side,
            root_angular_speed_rad_s,
        )
    )
    shape = raw[0].shape
    if any(value.shape != shape for value in raw) or any(
        not np.all(np.isfinite(value)) for value in raw
    ):
        raise ValueError("targeted dive feature arrays must be finite and shape-aligned")
    direction_value, phase_value = raw[:2]
    if np.any(np.abs(direction_value) != 1.0) or np.any((phase_value < 0) | (phase_value > 1)):
        raise ValueError("targeted dive direction/phase is invalid")
    pi = math.pi
    return np.stack(
        (
            direction_value,
            phase_value,
            np.square(phase_value),
            np.sin(pi * phase_value),
            np.cos(pi * phase_value),
            np.sin(2.0 * pi * phase_value),
            np.cos(2.0 * pi * phase_value),
            np.clip(raw[2] / 1.10, -1.0, 1.0),
            np.clip((raw[3] - 0.82) / 0.75, -1.0, 1.0),
            np.clip(raw[4] / 0.90, 0.0, 1.5),
            np.clip(raw[5] / 0.80, -1.25, 1.25),
            np.clip(raw[6] / 1.20, -1.5, 1.5),
            np.clip((raw[7] - 0.65) / 0.30, -1.5, 1.5),
            np.clip(raw[8], -1.0, 1.0),
            np.clip(raw[9], -1.0, 1.0),
            np.clip(raw[10] / 4.0, 0.0, 1.5),
        ),
        axis=-1,
    )


def targeted_dive_features_torch(
    *,
    torch: Any,
    direction: Any,
    phase: Any,
    target_lateral_m: Any,
    target_height_m: Any,
    time_to_arrival_sec: Any,
    root_lateral_m: Any,
    root_lateral_speed_mps: Any,
    pelvis_height_m: Any,
    upright_projection: Any,
    support_side: Any,
    root_angular_speed_rad_s: Any,
) -> Any:
    """Torch parity path without a CPU round trip."""

    pi = float(math.pi)
    return torch.stack(
        (
            direction,
            phase,
            phase.square(),
            torch.sin(pi * phase),
            torch.cos(pi * phase),
            torch.sin(2.0 * pi * phase),
            torch.cos(2.0 * pi * phase),
            torch.clamp(target_lateral_m / 1.10, -1.0, 1.0),
            torch.clamp((target_height_m - 0.82) / 0.75, -1.0, 1.0),
            torch.clamp(time_to_arrival_sec / 0.90, 0.0, 1.5),
            torch.clamp(root_lateral_m / 0.80, -1.25, 1.25),
            torch.clamp(root_lateral_speed_mps / 1.20, -1.5, 1.5),
            torch.clamp((pelvis_height_m - 0.65) / 0.30, -1.5, 1.5),
            torch.clamp(upright_projection, -1.0, 1.0),
            torch.clamp(support_side, -1.0, 1.0),
            torch.clamp(root_angular_speed_rad_s / 4.0, 0.0, 1.5),
        ),
        dim=-1,
    )


def build_targeted_dive_decoder(torch: Any, nn: Any, *, hidden_size: int) -> Any:
    """Residual MLP used identically by DDP training and deployment replay."""

    return nn.Sequential(
        nn.Linear(TARGETED_DIVE_INPUT_SIZE, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, TARGETED_DIVE_OUTPUT_SIZE),
    )


_MIRROR_ORDER = (
    6,
    7,
    8,
    9,
    10,
    11,
    0,
    1,
    2,
    3,
    4,
    5,
    12,
    13,
    14,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
)
_MIRROR_SIGN = (
    (1.0, -1.0, -1.0, 1.0, 1.0, -1.0) * 2
    + (-1.0, -1.0, 1.0)
    + (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0) * 2
)


def _mirror_features_torch(features: Any) -> Any:
    mirrored = features.clone()
    mirrored[:, (0, 7, 10, 11, 14)] *= -1.0
    return mirrored


def _mirror_joints_torch(torch: Any, joints: Any) -> Any:
    order = torch.as_tensor(_MIRROR_ORDER, dtype=torch.long, device=joints.device)
    sign = torch.as_tensor(_MIRROR_SIGN, dtype=joints.dtype, device=joints.device)
    return joints[:, order] * sign


def _mirror_joints_numpy(joints: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        joints[:, np.asarray(_MIRROR_ORDER, dtype=np.int64)]
        * np.asarray(_MIRROR_SIGN, dtype=np.float64)[None, :],
        dtype=np.float64,
    )


def _group_averaged_residual_torch(torch: Any, model: Any, features: Any) -> Any:
    """Replay the first exact-symmetry projection used by rejected evidence.

    This remains only so content-bound v15/v16 checkpoints cannot silently
    change semantics after the canonical-half-space decoder supersedes it.
    """

    direct = model(features)
    reflected = _mirror_joints_torch(torch, model(_mirror_features_torch(features)))
    return 0.5 * (direct + reflected)


def _equivariant_residual_torch(torch: Any, model: Any, features: Any) -> Any:
    """Decode one canonical half-space with exact mirror equivariance.

    Negative-direction observations are reflected into the positive canonical
    frame before inference, and their joint residuals are reflected back.  In
    contrast to group averaging, this preserves the MLP's full capacity while
    guaranteeing ``r(Mx) == M r(x)`` up to floating-point roundoff.
    """

    reflected_input = features[:, 0] < 0.0
    canonical_features = torch.where(
        reflected_input.unsqueeze(1),
        _mirror_features_torch(features),
        features,
    )
    canonical_residual = model(canonical_features)
    reflected_residual = _mirror_joints_torch(torch, canonical_residual)
    return torch.where(
        reflected_input.unsqueeze(1),
        reflected_residual,
        canonical_residual,
    )


def _smooth_window(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    rise = np.clip((phase - 0.20) / 0.32, 0.0, 1.0)
    fall = np.clip((1.0 - phase) / 0.20, 0.0, 1.0)
    rise = rise * rise * (3.0 - 2.0 * rise)
    fall = fall * fall * (3.0 - 2.0 * fall)
    return np.minimum(rise, fall)


def _interpolate_seed(
    seed_joint: NDArray[np.float64],
    direction: NDArray[np.float64],
    phase: NDArray[np.float64],
) -> NDArray[np.float64]:
    index = phase * (seed_joint.shape[1] - 1)
    low = np.floor(index).astype(np.int64)
    high = np.minimum(low + 1, seed_joint.shape[1] - 1)
    alpha = index - low
    side = (direction > 0.0).astype(np.int64)
    return np.asarray(
        seed_joint[side, low] * (1.0 - alpha[:, None]) + seed_joint[side, high] * alpha[:, None],
        dtype=np.float64,
    )


def _manufacture_curriculum(
    *,
    asset_root: Path,
    source_checkout: Path,
    sample_count: int,
    random_seed: int,
    low_height_crouch_scale: float,
    low_height_max_hip_flexion_delta_rad: float,
    low_height_max_knee_flexion_delta_rad: float,
    low_height_max_ankle_dorsiflexion_delta_rad: float,
    low_height_max_waist_roll_delta_rad: float,
    low_height_max_waist_pitch_delta_rad: float,
    low_height_crouch_start_lateral_progress: float,
    low_height_crouch_full_lateral_progress: float,
    recovery_tail_start_phase: float | None,
    high_height_extension_scale: float,
    failure_replay: list[dict[str, Any]] | None = None,
    failure_replay_fraction: float = 0.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
    """Make target labels from motion prior + qualified nonlinear hand IK."""

    atlas = load_official_goalkeeper_dive_atlas(checkout=source_checkout)
    seed = build_balanced_dive_imitation_seed(atlas)
    low_seed = build_balanced_dive_imitation_seed(
        atlas,
        window_profile="low_vertical_dip",
    )
    reach = build_g1_task_space_reach_atlas(
        asset_root,
        config=GoalkeeperReachConfig(
            damping=0.12,
            reach_gain=0.95,
            maximum_position_error_m=0.75,
            support_arm_scale=0.60,
            central_support_scale=0.95,
            residual_scale=1.0,
            arm_authority_scale=0.88,
            workspace_scale=2.50,
        ),
        # Four-neighbour interpolation is adequate as a runtime lookup but
        # creates sharp Voronoi boundaries in imitation labels.  A wider
        # neighbourhood makes the teacher continuous enough for a neural
        # muscle memory without changing its bounded workspace.
        atlas_config=GoalkeeperReachAtlasConfig(
            interpolation_neighbors=42,
            interpolation_kernel="gaussian",
            interpolation_temperature=0.75,
            multistart_count=12,
        ),
    )
    rng = np.random.default_rng(random_seed)
    direction = rng.choice((-1.0, 1.0), size=sample_count)
    phase = rng.uniform(0.0, 1.0, size=sample_count)
    magnitude = rng.uniform(0.72, 1.10, size=sample_count)
    target_lateral = direction * magnitude
    height_band = rng.integers(0, 3, size=sample_count)
    target_height = np.where(
        height_band == 0,
        rng.uniform(0.16, 0.60, size=sample_count),
        np.where(
            height_band == 1,
            rng.uniform(0.64, 1.10, size=sample_count),
            rng.uniform(1.14, 1.52, size=sample_count),
        ),
    )
    time_to_arrival = rng.uniform(0.0, 0.90, size=sample_count)
    root_lateral = target_lateral * rng.uniform(0.0, 0.82, size=sample_count)
    root_lateral += rng.normal(0.0, 0.04, size=sample_count)
    root_speed = direction * rng.uniform(0.0, 0.85, size=sample_count)
    pelvis_height = rng.uniform(0.54, 0.82, size=sample_count)
    upright = rng.uniform(0.70, 1.0, size=sample_count)
    support = rng.choice((-1.0, 0.0, 1.0), size=sample_count, p=(0.4, 0.2, 0.4))
    angular = rng.uniform(0.0, 3.2, size=sample_count)
    replay_count = 0
    if failure_replay and failure_replay_fraction > 0.0:
        replay_count = min(
            sample_count,
            int(round(sample_count * failure_replay_fraction)),
        )
        replay_indices = rng.integers(0, len(failure_replay), size=replay_count)
        for index, replay_index in enumerate(replay_indices):
            failure = failure_replay[int(replay_index)]
            target_y = float(failure["target_y_m"])
            target_z = float(failure["target_z_m"])
            flight_sec = float(failure["flight_sec"])
            sign = -1.0 if target_y < 0.0 else 1.0
            target_lateral[index] = sign * np.clip(
                abs(target_y) + rng.normal(0.0, 0.025), 0.72, 1.10
            )
            target_height[index] = np.clip(target_z + rng.normal(0.0, 0.035), 0.16, 1.52)
            time_to_arrival[index] = np.clip(
                flight_sec * (1.0 - phase[index]) + rng.normal(0.0, 0.018),
                0.0,
                0.90,
            )
            root_lateral[index] = target_lateral[index] * rng.uniform(0.0, 0.82)
            root_speed[index] = sign * rng.uniform(0.0, 0.85)
    # Manufacture exact group-paired examples.  The nonlinear IK teacher is
    # deterministic but can select different local minima on reflected poses;
    # allowing both labels into the corpus makes the same physical situation
    # demand contradictory muscle memories.  One member is the teacher result
    # and the other is its exact anatomical reflection.
    pair_count = sample_count // 2
    paired = slice(pair_count, sample_count)
    source = slice(0, pair_count)
    direction[paired] = -direction[source]
    phase[paired] = phase[source]
    target_lateral[paired] = -target_lateral[source]
    target_height[paired] = target_height[source]
    time_to_arrival[paired] = time_to_arrival[source]
    root_lateral[paired] = -root_lateral[source]
    root_speed[paired] = -root_speed[source]
    pelvis_height[paired] = pelvis_height[source]
    upright[paired] = upright[source]
    support[paired] = -support[source]
    angular[paired] = angular[source]
    features = targeted_dive_features_numpy(
        direction=direction,
        phase=phase,
        target_lateral_m=target_lateral,
        target_height_m=target_height,
        time_to_arrival_sec=time_to_arrival,
        root_lateral_m=root_lateral,
        root_lateral_speed_mps=root_speed,
        pelvis_height_m=pelvis_height,
        upright_projection=upright,
        support_side=support,
        root_angular_speed_rad_s=angular,
    )
    anchor_joint = seed.joint_position_rad[:, :TARGETED_DIVE_ANCHOR_FRAMES]
    anchor = _interpolate_seed(anchor_joint, direction, phase)
    low_anchor_joint = low_seed.joint_position_rad[:, :TARGETED_DIVE_ANCHOR_FRAMES]
    low_anchor = _interpolate_seed(low_anchor_joint, direction, phase)
    low_motion_blend = np.clip((0.60 - target_height) / 0.30, 0.0, 1.0)
    low_motion_blend = low_motion_blend * low_motion_blend * (3.0 - 2.0 * low_motion_blend)
    anchor = anchor + low_motion_blend[:, None] * (low_anchor - anchor)
    ready = _ready_pose()
    relative = np.stack(
        (
            np.full(sample_count, -0.08),
            target_lateral - root_lateral,
            target_height - pelvis_height,
        ),
        axis=1,
    )
    reach_action = task_space_reach_from_target_numpy(target_relative=relative, model=reach)
    reach_target = ready[15:29] + reach_action * np.asarray(
        tuple(reach.effective_arm_limits_rad) * 2, dtype=np.float64
    )
    window = _smooth_window(phase)
    urgency = np.clip((0.60 - time_to_arrival) / 0.38, 0.0, 1.0)
    # The offline teacher owns the desired motion manifold, not runtime
    # safety authority.  Attenuating labels by instantaneous posture here and
    # again in the runtime angular-speed guard double-suppresses exactly the
    # hard examples that must teach full high-corner reach.
    blend = window * (0.65 + 0.35 * urgency)
    target = anchor.copy()
    target[:, 15:29] += blend[:, None] * (reach_target - anchor[:, 15:29])
    # Bounded height and support conditioning.  This changes the whole-body
    # target without inventing an unconstrained lower-body residual policy.
    height_norm = np.clip((target_height - 0.82) / 0.70, -1.0, 1.0)
    low_crouch = low_height_crouch_scale * np.clip(
        (0.64 - target_height) / 0.48,
        0.0,
        1.0,
    )
    # Predict where the pelvis will be at interception from proprioceptive
    # position, velocity and remaining flight time.  Current progress alone
    # triggers the crouch after the ball has already arrived; projected
    # progress lets the decoder overlap the end of translation with descent.
    projected_root_lateral = root_lateral + root_speed * np.clip(
        time_to_arrival,
        0.0,
        0.55,
    )
    lateral_progress = np.clip(
        np.abs(projected_root_lateral) / np.maximum(np.abs(target_lateral), 1.0e-6),
        0.0,
        1.0,
    )
    crouch_delivery = np.clip(
        (lateral_progress - low_height_crouch_start_lateral_progress)
        / (low_height_crouch_full_lateral_progress - low_height_crouch_start_lateral_progress),
        0.0,
        1.0,
    )
    crouch_delivery = crouch_delivery * crouch_delivery * (3.0 - 2.0 * crouch_delivery)
    low_crouch *= crouch_delivery
    high_extension = high_height_extension_scale * np.clip(
        (target_height - 1.05) / 0.47,
        0.0,
        1.0,
    )
    posture_gate = window * (0.65 + 0.35 * urgency)
    low_rise = np.clip((phase - 0.05) / 0.25, 0.0, 1.0)
    low_fall = np.clip((1.0 - phase) / 0.20, 0.0, 1.0)
    low_rise = low_rise * low_rise * (3.0 - 2.0 * low_rise)
    low_fall = low_fall * low_fall * (3.0 - 2.0 * low_fall)
    low_posture_gate = np.minimum(low_rise, low_fall) * (0.65 + 0.35 * urgency)
    # Height is a whole-body reach problem.  These anatomically paired
    # teacher labels create arm workspace without giving a runtime heuristic
    # direct joint authority: low shots flex the stance, high shots extend it.
    for hip_pitch in (0, 6):
        target[:, hip_pitch] += (
            -low_posture_gate * low_height_max_hip_flexion_delta_rad * low_crouch
            + posture_gate * 0.05 * high_extension
        )
    for knee in (3, 9):
        target[:, knee] += (
            low_posture_gate * low_height_max_knee_flexion_delta_rad * low_crouch
            - posture_gate * 0.12 * high_extension
        )
    for ankle_pitch in (4, 10):
        target[:, ankle_pitch] += (
            -low_posture_gate * low_height_max_ankle_dorsiflexion_delta_rad * low_crouch
            + posture_gate * 0.06 * high_extension
        )
    target[:, 13] += window * direction * (0.05 + 0.04 * urgency)
    target[:, 13] += low_posture_gate * direction * low_height_max_waist_roll_delta_rad * low_crouch
    target[:, 14] += window * (0.05 * height_norm)
    target[:, 14] += low_posture_gate * low_height_max_waist_pitch_delta_rad * low_crouch
    support_match = support * direction
    for hip_roll in (1, 7):
        target[:, hip_roll] += window * direction * (0.025 + 0.015 * support_match)
    recovery_alpha = np.zeros(sample_count, dtype=np.float64)
    if recovery_tail_start_phase is not None:
        recovery_alpha = np.clip(
            (phase - recovery_tail_start_phase) / (1.0 - recovery_tail_start_phase),
            0.0,
            1.0,
        )
        recovery_alpha = recovery_alpha * recovery_alpha * (3.0 - 2.0 * recovery_alpha)
        anchor = ready + (anchor - ready) * (1.0 - recovery_alpha[:, None])
        target = ready + (target - ready) * (1.0 - recovery_alpha[:, None])
    anchor[paired] = _mirror_joints_numpy(anchor[source])
    target[paired] = _mirror_joints_numpy(target[source])
    target = np.asarray(target, dtype=np.float64)
    if not np.all(np.isfinite(target)):
        raise RuntimeError("targeted dive curriculum generated non-finite targets")
    reach_targets = np.asarray(reach.target_relative_m, dtype=np.float64)
    reach_errors = np.minimum(
        np.asarray(reach.left_terminal_error_m, dtype=np.float64),
        np.asarray(reach.right_terminal_error_m, dtype=np.float64),
    )
    reach_height_masks = {
        "low": reach_targets[:, 2] < -0.20,
        "mid": (reach_targets[:, 2] >= -0.20) & (reach_targets[:, 2] < 0.40),
        "high": reach_targets[:, 2] >= 0.40,
    }
    reach_quality = {
        name: {
            "mean_terminal_error_m": float(np.mean(reach_errors[mask])),
            "maximum_terminal_error_m": float(np.max(reach_errors[mask])),
        }
        for name, mask in reach_height_masks.items()
    }
    metadata = {
        "dive_seed_hash": seed.seed_hash,
        "low_dive_seed_hash": low_seed.seed_hash,
        "source_atlas_hash": atlas.atlas_hash,
        "source_window": [seed.source_start_frame, seed.source_start_frame + 62],
        "low_source_window": [
            low_seed.source_start_frame,
            low_seed.source_start_frame + 62,
        ],
        "reach_atlas_hash": reach.model_hash,
        "height_strata": ["far_corner_low", "far_corner_mid", "far_corner_high"],
        "minimum_lateral_target_m": 0.72,
        "maximum_lateral_target_m": 1.10,
        "maximum_sampled_root_lateral_fraction": 0.82,
        "teacher": "HEIGHT_ROUTED_DIVE_PRIORS_PLUS_EXPANDED_TARGET_SPACE_IK",
        "teacher_runtime_authority_separation": "FULL_LABELS_RUNTIME_GUARD_OWNS_ATTENUATION",
        "height_conditioned_posture_teacher": {
            "low": "BILATERAL_HIP_KNEE_FLEXION_PLUS_ANKLE_DORSIFLEXION",
            "high": "BILATERAL_STANCE_EXTENSION",
            "low_scale": low_height_crouch_scale,
            "high_scale": high_height_extension_scale,
            "low_max_joint_delta_rad": {
                "hip_flexion": low_height_max_hip_flexion_delta_rad,
                "knee_flexion": low_height_max_knee_flexion_delta_rad,
                "ankle_dorsiflexion": low_height_max_ankle_dorsiflexion_delta_rad,
                "waist_roll": low_height_max_waist_roll_delta_rad,
                "waist_pitch": low_height_max_waist_pitch_delta_rad,
            },
            "low_crouch_lateral_progress_window": [
                low_height_crouch_start_lateral_progress,
                low_height_crouch_full_lateral_progress,
            ],
            "low_crouch_semantics": "PROJECTED_TRANSLATE_THEN_DESCEND_SMOOTHSTEP",
            "low_crouch_prediction_horizon_sec": 0.55,
            "low_crouch_phase_window": [0.05, 0.30, 0.80, 1.00],
        },
        "recovery_tail": {
            "enabled": recovery_tail_start_phase is not None,
            "start_phase": recovery_tail_start_phase,
            "terminal_pose": "QUALIFIED_LOCOMOTION_READY",
            "blend": "CUBIC_SMOOTHSTEP",
        },
        "mirror_pairing": "EXACT_ANATOMICAL_GROUP_PAIRS",
        "reach_workspace_scale": 2.50,
        "reach_static_quality": reach_quality,
        "reach_static_quality_gate_passed": bool(
            reach_quality["mid"]["mean_terminal_error_m"] <= 0.30
            and reach_quality["high"]["mean_terminal_error_m"] <= 0.40
        ),
        "failure_replay_examples": replay_count,
        "failure_replay_source_cases": len(failure_replay or ()),
    }
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        np.asarray(anchor, dtype=np.float32),
        metadata,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _load_failure_replay(path: Path | None) -> tuple[list[dict[str, Any]], str | None]:
    """Load only bounded SIM_ONLY failures from an independent CPU report."""

    if path is None:
        return [], None
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("physics_backend") != "mujoco_cpu"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("failure replay must come from a SIM_ONLY CPU MuJoCo exam")
    failures = payload.get("candidate", {}).get("failures")
    if not isinstance(failures, list) or not failures:
        raise ValueError("failure replay report has no candidate failures")
    accepted: list[dict[str, Any]] = []
    for row in failures:
        try:
            target_y = float(row["target_y_m"])
            target_z = float(row["target_z_m"])
            flight_sec = float(row["flight_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("failure replay row is malformed") from exc
        if (
            not all(math.isfinite(value) for value in (target_y, target_z, flight_sec))
            or not 0.70 <= abs(target_y) <= 1.15
            or not 0.10 <= target_z <= 1.60
            or not 0.30 <= flight_sec <= 0.70
        ):
            raise ValueError("failure replay row exceeds the hard-shot envelope")
        accepted.append(
            {
                "seed": int(row.get("seed", -1)),
                "stratum": str(row.get("stratum", "unknown")),
                "target_y_m": target_y,
                "target_z_m": target_z,
                "flight_sec": flight_sec,
                "failure_category": str(row.get("failure_category", "UNCLASSIFIED")),
            }
        )
    return accepted, str(hash_bytes(source.read_bytes()))


def train_goalkeeper_targeted_dive(
    *,
    asset_root: Path,
    source_checkout: Path,
    output_dir: Path,
    config: GoalkeeperTargetedDiveConfig | None = None,
    device: str | None = None,
    failure_replay_path: Path | None = None,
    initialization_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Train on one or many GPUs; only rank zero writes canonical evidence."""

    import torch
    from torch import nn

    active = config or GoalkeeperTargetedDiveConfig()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("targeted dive DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        active_device = torch.device(f"cuda:{local_rank}")
    else:
        active_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(active.random_seed + rank)
    if active_device.type == "cuda":
        torch.cuda.manual_seed_all(active.random_seed + rank)
    failure_replay, failure_replay_hash = _load_failure_replay(failure_replay_path)
    features_np, target_np, anchor_np, metadata = _manufacture_curriculum(
        asset_root=asset_root,
        source_checkout=source_checkout,
        sample_count=active.samples_per_epoch,
        random_seed=active.random_seed,
        low_height_crouch_scale=active.low_height_crouch_scale,
        low_height_max_hip_flexion_delta_rad=(active.low_height_max_hip_flexion_delta_rad),
        low_height_max_knee_flexion_delta_rad=(active.low_height_max_knee_flexion_delta_rad),
        low_height_max_ankle_dorsiflexion_delta_rad=(
            active.low_height_max_ankle_dorsiflexion_delta_rad
        ),
        low_height_max_waist_roll_delta_rad=active.low_height_max_waist_roll_delta_rad,
        low_height_max_waist_pitch_delta_rad=active.low_height_max_waist_pitch_delta_rad,
        low_height_crouch_start_lateral_progress=(active.low_height_crouch_start_lateral_progress),
        low_height_crouch_full_lateral_progress=(active.low_height_crouch_full_lateral_progress),
        recovery_tail_start_phase=active.recovery_tail_start_phase,
        high_height_extension_scale=active.high_height_extension_scale,
        failure_replay=failure_replay,
        failure_replay_fraction=active.failure_replay_fraction,
    )
    metadata["failure_replay_report_hash"] = failure_replay_hash
    shard = np.arange(rank, active.samples_per_epoch, world_size, dtype=np.int64)
    features = torch.as_tensor(features_np[shard], device=active_device)
    target = torch.as_tensor(target_np[shard], device=active_device)
    anchor = torch.as_tensor(anchor_np[shard], device=active_device)
    scale = torch.maximum(
        torch.max(
            torch.abs(torch.as_tensor(target_np - anchor_np, device=active_device)),
            dim=0,
        ).values,
        torch.full((29,), 0.10, device=active_device),
    )
    mirror_order = torch.as_tensor(_MIRROR_ORDER, dtype=torch.long, device=active_device)
    scale = torch.maximum(scale, scale[mirror_order])
    raw_model = build_targeted_dive_decoder(torch, nn, hidden_size=active.hidden_size).to(
        active_device
    )
    initialization: dict[str, Any] | None = None
    if initialization_checkpoint_path is not None:
        initialization_path = initialization_checkpoint_path.expanduser().resolve()
        warmstart = torch.load(initialization_path, map_location="cpu", weights_only=True)
        if (
            warmstart.get("activation_ceiling") != "SIM_ONLY"
            or warmstart.get("hardware_authorized") is not False
            or warmstart.get("commercial_use_allowed") is not False
            or warmstart.get("decoder_symmetry_enforcement") != "MIRROR_CANONICAL_HALF_SPACE_V1"
            or int(warmstart.get("input_size", -1)) != TARGETED_DIVE_INPUT_SIZE
            or int(warmstart.get("output_size", -1)) != TARGETED_DIVE_OUTPUT_SIZE
            or int(warmstart.get("hidden_size", -1)) != active.hidden_size
        ):
            raise ValueError("targeted dive initialization checkpoint is incompatible")
        raw_model.load_state_dict(warmstart["model_state_dict"])
        initialization = {
            "checkpoint_hash": hash_bytes(initialization_path.read_bytes()),
            "role": "CONTENT_BOUND_MUSCLE_MEMORY_WARMSTART",
        }
    model: Any = raw_model
    if distributed:
        model = nn.parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=active.learning_rate, weight_decay=active.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=active.epochs,
        eta_min=active.learning_rate * 0.05,
    )
    generator = torch.Generator(device=active_device)
    generator.manual_seed(active.random_seed + 100_003 * rank)
    history: list[dict[str, Any]] = []
    for epoch in range(active.epochs):
        order = torch.randperm(features.shape[0], generator=generator, device=active_device)
        epoch_loss = torch.zeros((), device=active_device)
        batches = 0
        for start in range(0, features.shape[0], active.minibatch_size):
            ids = order[start : start + active.minibatch_size]
            residual = _equivariant_residual_torch(torch, model, features[ids])
            predicted = anchor[ids] + residual * scale
            error = torch.square(predicted - target[ids])
            correction = torch.max(torch.abs(target[ids] - anchor[ids]), dim=1).values
            importance = 1.0 + 2.0 * torch.clamp(correction, 0.0, 1.5)
            position = torch.mean(importance * torch.mean(error, dim=1))
            tail_count = max(1, error.numel() // 100)
            tail = torch.mean(torch.topk(error.flatten(), tail_count).values)
            anchor_loss = torch.mean(torch.square(predicted - anchor[ids]))
            velocity = torch.mean(
                torch.square(
                    (predicted[1:] - predicted[:-1]) - (target[ids][1:] - target[ids][:-1])
                )
            )
            mirrored_features = _mirror_features_torch(features[ids])
            mirrored_prediction = _mirror_joints_torch(torch, anchor[ids]) + (
                _equivariant_residual_torch(torch, model, mirrored_features) * scale
            )
            symmetry = torch.mean(
                torch.square(mirrored_prediction - _mirror_joints_torch(torch, predicted))
            )
            loss = (
                position
                + active.velocity_loss_weight * velocity
                + active.anchor_loss_weight * anchor_loss
                + active.symmetry_loss_weight * symmetry
                + active.tail_loss_weight * tail
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.detach()
            batches += 1
        scheduler.step()
        if distributed:
            torch.distributed.all_reduce(epoch_loss)
            epoch_loss /= world_size
        if rank == 0 and (epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == active.epochs):
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "stage": "global_curriculum",
                    "mean_loss": float(epoch_loss / batches),
                }
            )
    all_features = torch.as_tensor(features_np, device=active_device)
    all_target = torch.as_tensor(target_np, device=active_device)
    all_anchor = torch.as_tensor(anchor_np, device=active_device)
    hard_example_hashes: list[str] = []
    if active.hard_mining_epochs:
        for group in optimizer.param_groups:
            group["lr"] = active.learning_rate * active.hard_mining_learning_rate_scale
        hard_features = all_features[:0]
        hard_target = all_target[:0]
        hard_anchor = all_anchor[:0]
        for hard_epoch in range(active.hard_mining_epochs):
            if hard_epoch % active.hard_mining_refresh_epochs == 0:
                raw_model.eval()
                with torch.inference_mode():
                    refreshed = all_anchor + (
                        _equivariant_residual_torch(torch, raw_model, all_features) * scale
                    )
                    row_error = torch.max(torch.abs(refreshed - all_target), dim=1).values
                    hard_count = min(active.hard_mining_samples, active.samples_per_epoch)
                    hard_count -= hard_count % world_size
                    if hard_count < world_size:
                        raise ValueError("targeted dive hard-mining shard is empty")
                    hard_indices = torch.topk(row_error, k=hard_count, sorted=True).indices
                if rank == 0:
                    hard_example_hashes.append(
                        str(
                            hash_json(
                                [int(value) for value in hard_indices.detach().cpu().tolist()]
                            )
                        )
                    )
                hard_shard = hard_indices[rank::world_size]
                hard_features = all_features[hard_shard]
                hard_target = all_target[hard_shard]
                hard_anchor = all_anchor[hard_shard]
                raw_model.train()
            replay_fraction = active.hard_mining_general_replay_fraction
            general_count = min(
                features.shape[0],
                int(round(hard_features.shape[0] * replay_fraction / (1.0 - replay_fraction)))
                if replay_fraction > 0.0
                else 0,
            )
            if general_count:
                general_ids = torch.randperm(
                    features.shape[0], generator=generator, device=active_device
                )[:general_count]
                epoch_features = torch.cat((hard_features, features[general_ids]), dim=0)
                epoch_target = torch.cat((hard_target, target[general_ids]), dim=0)
                epoch_anchor = torch.cat((hard_anchor, anchor[general_ids]), dim=0)
            else:
                epoch_features = hard_features
                epoch_target = hard_target
                epoch_anchor = hard_anchor
            order = torch.randperm(
                epoch_features.shape[0], generator=generator, device=active_device
            )
            epoch_loss = torch.zeros((), device=active_device)
            batches = 0
            for start in range(0, epoch_features.shape[0], active.minibatch_size):
                ids = order[start : start + active.minibatch_size]
                residual = _equivariant_residual_torch(torch, model, epoch_features[ids])
                predicted = epoch_anchor[ids] + residual * scale
                error = torch.square(predicted - epoch_target[ids])
                correction = torch.max(
                    torch.abs(epoch_target[ids] - epoch_anchor[ids]), dim=1
                ).values
                importance = 1.0 + 2.0 * torch.clamp(correction, 0.0, 1.5)
                position = torch.mean(importance * torch.mean(error, dim=1))
                tail_count = max(1, error.numel() // 50)
                tail = torch.mean(torch.topk(error.flatten(), tail_count).values)
                anchor_loss = torch.mean(torch.square(predicted - epoch_anchor[ids]))
                velocity = torch.mean(
                    torch.square(
                        (predicted[1:] - predicted[:-1])
                        - (epoch_target[ids][1:] - epoch_target[ids][:-1])
                    )
                )
                mirrored_features = _mirror_features_torch(epoch_features[ids])
                mirrored_prediction = _mirror_joints_torch(torch, epoch_anchor[ids]) + (
                    _equivariant_residual_torch(torch, model, mirrored_features) * scale
                )
                symmetry = torch.mean(
                    torch.square(mirrored_prediction - _mirror_joints_torch(torch, predicted))
                )
                loss = (
                    position
                    + active.velocity_loss_weight * velocity
                    + active.anchor_loss_weight * anchor_loss
                    + active.symmetry_loss_weight * symmetry
                    + active.tail_loss_weight * tail
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.detach()
                batches += 1
            if distributed:
                torch.distributed.all_reduce(epoch_loss)
                epoch_loss /= world_size
            if rank == 0 and (
                hard_epoch == 0
                or (hard_epoch + 1) % 20 == 0
                or hard_epoch + 1 == active.hard_mining_epochs
            ):
                history.append(
                    {
                        "epoch": float(active.epochs + hard_epoch + 1),
                        "stage": "hard_negative_replay",
                        "mean_loss": float(epoch_loss / batches),
                    }
                )
    raw_model.eval()
    with torch.inference_mode():
        decoded = all_anchor + _equivariant_residual_torch(torch, raw_model, all_features) * scale
    error = decoded - all_target
    metrics_tensor = torch.stack(
        (
            torch.sqrt(torch.mean(torch.square(error))),
            torch.max(torch.abs(error)),
        )
    )
    if distributed:
        torch.distributed.broadcast(metrics_tensor, src=0)
    metrics = {
        "position_rmse_rad": float(metrics_tensor[0]),
        "maximum_absolute_error_rad": float(metrics_tensor[1]),
    }
    destination = output_dir.expanduser().resolve()
    checkpoint_path = destination / "goalkeeper-targeted-dive.pt"
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=True)
        # Mirrored task inputs must produce mirrored joint targets within the
        # declared fit tolerance; use paired canonical probes for the audit.
        with torch.inference_mode():
            mirrored_features = _mirror_features_torch(all_features)
            mirrored_decoded = _mirror_joints_torch(torch, all_anchor) + (
                _equivariant_residual_torch(torch, raw_model, mirrored_features) * scale
            )
            mirrored_reference = _mirror_joints_torch(torch, decoded)
        symmetry_rmse = float(
            torch.sqrt(torch.mean(torch.square(mirrored_decoded - mirrored_reference)))
        )
        metrics["symmetry_rmse_rad"] = symmetry_rmse
        source_atlas = load_official_goalkeeper_dive_atlas(checkout=source_checkout)
        source_seed = build_balanced_dive_imitation_seed(source_atlas)
        checkpoint_seed = np.asarray(
            source_seed.joint_position_rad[:, :TARGETED_DIVE_ANCHOR_FRAMES],
            dtype=np.float64,
        ).copy()
        if active.recovery_tail_start_phase is not None:
            seed_phase = np.linspace(0.0, 1.0, checkpoint_seed.shape[1])
            recovery_alpha = np.clip(
                (seed_phase - active.recovery_tail_start_phase)
                / (1.0 - active.recovery_tail_start_phase),
                0.0,
                1.0,
            )
            recovery_alpha = recovery_alpha * recovery_alpha * (3.0 - 2.0 * recovery_alpha)
            ready_seed = _ready_pose()[None, None, :]
            checkpoint_seed = ready_seed + (checkpoint_seed - ready_seed) * (
                1.0 - recovery_alpha[None, :, None]
            )
        checkpoint = {
            "schema_version": "rosclaw_soccer.goalkeeper_targeted_dive_checkpoint.v2",
            "input_size": TARGETED_DIVE_INPUT_SIZE,
            "output_size": TARGETED_DIVE_OUTPUT_SIZE,
            "hidden_size": active.hidden_size,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in raw_model.state_dict().items()
            },
            "target_scale": scale.detach().cpu(),
            "output_representation": "RESIDUAL_OVER_BALANCED_DIVE_ANCHOR",
            "decoder_symmetry_enforcement": "MIRROR_CANONICAL_HALF_SPACE_V1",
            "imitation_seed_joint_position": torch.as_tensor(
                checkpoint_seed,
                dtype=torch.float32,
            ),
            "training_config": asdict(active),
            "training_metadata": metadata,
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
            "commercial_use_allowed": False,
        }
        torch.save(checkpoint, checkpoint_path)
        checkpoint_hash = hash_bytes(checkpoint_path.read_bytes())
        fit_passed = bool(
            metrics["position_rmse_rad"] <= active.maximum_position_rmse_rad
            and metrics["maximum_absolute_error_rad"] <= active.maximum_absolute_error_rad
            and metrics["symmetry_rmse_rad"] <= active.maximum_symmetry_rmse_rad
        )
        report: dict[str, Any] = {
            "schema_version": "rosclaw_soccer.goalkeeper_targeted_dive_training.v2",
            "config": asdict(active),
            "config_hash": active.config_hash,
            "distributed_backend": "nccl" if distributed else None,
            "initialization": initialization,
            "world_size": world_size,
            "training_devices": [f"cuda:{index}" for index in range(world_size)]
            if distributed
            else [str(active_device)],
            "sample_count": active.samples_per_epoch,
            "curriculum": metadata,
            "history": history,
            "hard_negative_replay": {
                "enabled": bool(active.hard_mining_epochs),
                "epochs": active.hard_mining_epochs,
                "sample_count": active.hard_mining_samples,
                "refresh_epochs": active.hard_mining_refresh_epochs,
                "general_replay_fraction": active.hard_mining_general_replay_fraction,
                "example_index_hashes": hard_example_hashes,
                "cpu_exam_report_hash": failure_replay_hash,
                "cpu_failure_case_count": len(failure_replay),
            },
            "metrics": metrics,
            "fit_gate_passed": fit_passed,
            "checkpoint": checkpoint_path.name,
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_authority": "TRAINING_OPTION_CANDIDATE",
            "policy_integration_completed": False,
            "passed": fit_passed,
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
            "commercial_use_allowed": False,
        }
        report["report_hash"] = hash_json(report)
        _atomic_json(destination / "training-report.json", report)
    else:
        report = {"rank": rank, "status": "WORKER_COMPLETE"}
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return report


def load_goalkeeper_targeted_dive(
    *, checkpoint_path: Path, device: Any
) -> tuple[Any, dict[str, Any]]:
    """Load a bounded candidate with safe deserialization."""

    import torch
    from torch import nn

    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location=device, weights_only=True
    )
    if (
        checkpoint.get("activation_ceiling") != "SIM_ONLY"
        or checkpoint.get("hardware_authorized") is not False
        or checkpoint.get("commercial_use_allowed") is not False
        or int(checkpoint.get("input_size", -1)) != TARGETED_DIVE_INPUT_SIZE
        or int(checkpoint.get("output_size", -1)) != TARGETED_DIVE_OUTPUT_SIZE
        or checkpoint.get("output_representation") != "RESIDUAL_OVER_BALANCED_DIVE_ANCHOR"
    ):
        raise ValueError("targeted dive checkpoint boundary is invalid")
    seed = checkpoint.get("imitation_seed_joint_position")
    if (
        seed is None
        or seed.ndim != 3
        or seed.shape[0] != 2
        or not 50 <= seed.shape[1] <= 71
        or seed.shape[2] != 29
        or not torch.all(torch.isfinite(seed))
    ):
        raise ValueError("targeted dive checkpoint imitation anchor is invalid")
    model = build_targeted_dive_decoder(torch, nn, hidden_size=int(checkpoint["hidden_size"])).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def decode_goalkeeper_targeted_dive(
    *,
    model: Any,
    checkpoint: dict[str, Any],
    features: Any,
    residual_authority: Any = 1.0,
) -> Any:
    """Decode bounded residuals over the immutable bilateral motion anchor."""

    import torch

    if features.ndim != 2 or features.shape[1] != TARGETED_DIVE_INPUT_SIZE:
        raise ValueError("targeted dive decoder features have an invalid shape")
    if not torch.all(torch.isfinite(features)):
        raise ValueError("targeted dive decoder features must be finite")
    if isinstance(residual_authority, (int, float)):
        if not math.isfinite(residual_authority) or not 0.0 <= residual_authority <= 1.0:
            raise ValueError("targeted dive residual authority is invalid")
        authority: Any = float(residual_authority)
    else:
        authority = torch.as_tensor(
            residual_authority,
            device=features.device,
            dtype=features.dtype,
        )
        if tuple(authority.shape) not in {
            (TARGETED_DIVE_OUTPUT_SIZE,),
            (features.shape[0], TARGETED_DIVE_OUTPUT_SIZE),
        }:
            raise ValueError("targeted dive residual authority has an invalid shape")
        if not bool(torch.all(torch.isfinite(authority) & (authority >= 0.0) & (authority <= 1.0))):
            raise ValueError("targeted dive residual authority is invalid")
    phase = torch.clamp(features[:, 1], 0.0, 1.0)
    direction_index = (features[:, 0] > 0.0).to(torch.long)
    seed = checkpoint["imitation_seed_joint_position"].to(
        device=features.device, dtype=features.dtype
    )
    index = phase * (seed.shape[1] - 1)
    low = torch.floor(index).to(torch.long)
    high = torch.clamp(low + 1, max=seed.shape[1] - 1)
    alpha = index - low.to(index.dtype)
    anchor = seed[direction_index, low] * (1.0 - alpha.unsqueeze(1))
    anchor += seed[direction_index, high] * alpha.unsqueeze(1)
    scale = checkpoint["target_scale"].to(device=features.device, dtype=features.dtype)
    with torch.inference_mode():
        symmetry = checkpoint.get("decoder_symmetry_enforcement")
        if symmetry == "MIRROR_CANONICAL_HALF_SPACE_V1":
            residual = _equivariant_residual_torch(torch, model, features)
        elif symmetry == "MIRROR_GROUP_AVERAGE_V1":
            residual = _group_averaged_residual_torch(torch, model, features)
        else:
            # Backwards-compatible replay for content-bound legacy evidence.
            residual = model(features)
        return anchor + authority * residual * scale


class GoalkeeperTorchDiveMonitor:
    """GPU-vectorized, fail-closed phase authority for a neural dive."""

    def __init__(
        self,
        *,
        torch: Any,
        environment_count: int,
        device: Any,
        config: GoalkeeperDiveOptionConfig | None = None,
    ) -> None:
        if not 1 <= environment_count <= 262_144:
            raise ValueError("targeted dive environment count is invalid")
        self.torch = torch
        self.config = config or GoalkeeperDiveOptionConfig()
        self.count = environment_count
        self.device = device
        self.phase = torch.full(
            (environment_count,),
            int(GoalkeeperDivePhase.READY),
            dtype=torch.long,
            device=device,
        )
        self.active_threat = torch.zeros(environment_count, dtype=torch.long, device=device)
        self.active_steps = torch.zeros_like(self.active_threat)
        self.recovery_steps = torch.zeros_like(self.active_threat)
        self.completed_dives = torch.zeros_like(self.active_threat)

    def reset(self, environment_ids: Any | None = None) -> None:
        ids = (
            self.torch.arange(self.count, device=self.device)
            if environment_ids is None
            else environment_ids.to(device=self.device, dtype=self.torch.long)
        )
        self.phase[ids] = int(GoalkeeperDivePhase.READY)
        self.active_threat[ids] = 0
        self.active_steps[ids] = 0
        self.recovery_steps[ids] = 0
        self.completed_dives[ids] = 0

    def step(
        self,
        *,
        option_request: Any,
        threat_id: Any,
        threat_visible: Any,
        lateral_intercept_error_m: Any,
        pelvis_height_m: Any,
        upright_projection: Any,
        root_linear_speed_mps: Any,
        root_angular_speed_rad_s: Any,
        permitted_landing_contact: Any,
        forbidden_body_contact: Any,
    ) -> dict[str, Any]:
        torch = self.torch
        tensors = (
            option_request,
            threat_id,
            threat_visible,
            lateral_intercept_error_m,
            pelvis_height_m,
            upright_projection,
            root_linear_speed_mps,
            root_angular_speed_rad_s,
            permitted_landing_contact,
            forbidden_body_contact,
        )
        if any(tuple(value.shape) != (self.count,) for value in tensors):
            raise ValueError("targeted dive monitor tensors must have shape (N,)")
        numeric = tensors[3:8]
        finite = torch.ones(self.count, dtype=torch.bool, device=self.device)
        for value in numeric:
            finite &= torch.isfinite(value)
        cfg = self.config
        ready = self.phase == int(GoalkeeperDivePhase.READY)
        far = torch.abs(lateral_intercept_error_m) >= cfg.trigger_lateral_error_m
        started = ready & option_request.bool() & threat_visible.bool() & far & (threat_id > 0)
        self.phase[started] = int(GoalkeeperDivePhase.TAKEOFF)
        self.active_threat[started] = threat_id[started]
        self.active_steps[started] = 0
        self.recovery_steps[started] = 0
        active = (self.phase >= int(GoalkeeperDivePhase.TAKEOFF)) & (
            self.phase <= int(GoalkeeperDivePhase.RECOVERY)
        )
        self.active_steps[active] += 1
        changed = active & threat_visible.bool() & (threat_id != self.active_threat)
        strict = (
            finite
            & (pelvis_height_m >= cfg.standing_minimum_pelvis_height_m)
            & (upright_projection >= cfg.standing_minimum_upright_projection)
            & (root_linear_speed_mps <= cfg.recovered_maximum_linear_speed_mps)
            & (root_angular_speed_rad_s <= cfg.recovered_maximum_angular_speed_rad_s)
        )
        envelope = (
            finite
            & (pelvis_height_m >= cfg.dive_minimum_pelvis_height_m)
            & (upright_projection >= cfg.dive_minimum_upright_projection)
            & (root_linear_speed_mps <= cfg.dive_maximum_linear_speed_mps)
            & (root_angular_speed_rad_s <= cfg.dive_maximum_angular_speed_rad_s)
            & ~forbidden_body_contact.bool()
        )
        failed = active & (~envelope | changed | (self.active_steps > cfg.maximum_option_steps))
        self.phase[failed] = int(GoalkeeperDivePhase.FAILED)
        active &= ~failed
        takeoff = active & (self.phase == int(GoalkeeperDivePhase.TAKEOFF))
        airborne = takeoff & (
            ~strict | (root_linear_speed_mps >= cfg.minimum_takeoff_lateral_speed_mps)
        )
        self.phase[airborne] = int(GoalkeeperDivePhase.FLIGHT)
        flight_or_takeoff = (self.phase == int(GoalkeeperDivePhase.TAKEOFF)) | (
            self.phase == int(GoalkeeperDivePhase.FLIGHT)
        )
        landing = active & flight_or_takeoff & permitted_landing_contact.bool()
        self.phase[landing] = int(GoalkeeperDivePhase.LANDING)
        flight_or_landing = (self.phase == int(GoalkeeperDivePhase.FLIGHT)) | (
            self.phase == int(GoalkeeperDivePhase.LANDING)
        )
        recovery = active & flight_or_landing & strict
        self.phase[recovery] = int(GoalkeeperDivePhase.RECOVERY)
        recovering = active & (self.phase == int(GoalkeeperDivePhase.RECOVERY))
        self.recovery_steps = torch.where(
            recovering & strict,
            self.recovery_steps + 1,
            torch.where(recovering, torch.zeros_like(self.recovery_steps), self.recovery_steps),
        )
        lost = recovering & ~strict
        self.phase[lost] = int(GoalkeeperDivePhase.LANDING)
        recovered = recovering & strict & (self.recovery_steps >= cfg.recovery_hold_steps)
        self.phase[recovered] = int(GoalkeeperDivePhase.COMPLETE)
        self.completed_dives[recovered] += 1
        normalized_phase = torch.clamp(
            self.active_steps.to(torch.float32) / max(cfg.maximum_option_steps, 1),
            0.0,
            1.0,
        )
        result = {
            "phase": self.phase.clone(),
            "normalized_phase": normalized_phase,
            "posture_exception_granted": active & envelope,
            "unsafe": self.phase == int(GoalkeeperDivePhase.FAILED),
            "option_started_event": started,
            "recovered_event": recovered,
        }
        self.phase[recovered] = int(GoalkeeperDivePhase.READY)
        self.active_threat[recovered] = 0
        self.active_steps[recovered] = 0
        self.recovery_steps[recovered] = 0
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--samples", type=int, default=32_768)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--hard-mining-epochs", type=int, default=200)
    parser.add_argument("--hard-mining-refresh-epochs", type=int, default=20)
    parser.add_argument("--hard-mining-general-replay-fraction", type=float, default=0.50)
    parser.add_argument("--low-height-crouch-scale", type=float, default=1.0)
    parser.add_argument("--low-height-max-hip-flexion-delta-rad", type=float, default=0.38)
    parser.add_argument("--low-height-max-knee-flexion-delta-rad", type=float, default=0.70)
    parser.add_argument("--low-height-max-ankle-dorsiflexion-delta-rad", type=float, default=0.30)
    parser.add_argument("--low-height-max-waist-roll-delta-rad", type=float, default=0.30)
    parser.add_argument("--low-height-max-waist-pitch-delta-rad", type=float, default=0.18)
    parser.add_argument("--low-height-crouch-start-lateral-progress", type=float, default=0.35)
    parser.add_argument("--low-height-crouch-full-lateral-progress", type=float, default=0.70)
    parser.add_argument("--recovery-tail-start-phase", type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--failure-replay", type=Path, default=None)
    parser.add_argument("--initialization-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    report = train_goalkeeper_targeted_dive(
        asset_root=args.asset_root,
        source_checkout=args.source_checkout,
        output_dir=args.output_dir,
        config=GoalkeeperTargetedDiveConfig(
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            samples_per_epoch=args.samples,
            hard_mining_epochs=args.hard_mining_epochs,
            hard_mining_refresh_epochs=args.hard_mining_refresh_epochs,
            hard_mining_general_replay_fraction=(args.hard_mining_general_replay_fraction),
            low_height_crouch_scale=args.low_height_crouch_scale,
            low_height_max_hip_flexion_delta_rad=(args.low_height_max_hip_flexion_delta_rad),
            low_height_max_knee_flexion_delta_rad=(args.low_height_max_knee_flexion_delta_rad),
            low_height_max_ankle_dorsiflexion_delta_rad=(
                args.low_height_max_ankle_dorsiflexion_delta_rad
            ),
            low_height_max_waist_roll_delta_rad=args.low_height_max_waist_roll_delta_rad,
            low_height_max_waist_pitch_delta_rad=args.low_height_max_waist_pitch_delta_rad,
            low_height_crouch_start_lateral_progress=(
                args.low_height_crouch_start_lateral_progress
            ),
            low_height_crouch_full_lateral_progress=(args.low_height_crouch_full_lateral_progress),
            recovery_tail_start_phase=args.recovery_tail_start_phase,
        ),
        device=args.device,
        failure_replay_path=args.failure_replay,
        initialization_checkpoint_path=args.initialization_checkpoint,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TARGETED_DIVE_INPUT_SIZE",
    "TARGETED_DIVE_OUTPUT_SIZE",
    "TARGETED_DIVE_ANCHOR_FRAMES",
    "GoalkeeperTargetedDiveConfig",
    "GoalkeeperTorchDiveMonitor",
    "build_targeted_dive_decoder",
    "decode_goalkeeper_targeted_dive",
    "load_goalkeeper_targeted_dive",
    "targeted_dive_features_numpy",
    "targeted_dive_features_torch",
    "train_goalkeeper_targeted_dive",
]
