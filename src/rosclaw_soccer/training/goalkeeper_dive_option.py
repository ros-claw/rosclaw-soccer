"""Auditable imitation atlas and safety state machine for controlled dives.

Standing shuffles cannot cover regulation high/wide shots quickly enough.  A
goalkeeper therefore needs a deliberate dive option, but a low pelvis or a
tilted torso must not silently become a globally accepted state.  This module
keeps those concerns separate:

* the official Humanoid-Goalkeeper jump demonstrations are loaded as
  provenance-bound, train-only references; and
* :class:`GoalkeeperControlledDiveMonitor` grants a temporary posture
  exception only to a declared option with a bounded envelope and mandatory
  recovery.

Nothing in this module commands hardware or promotes a policy.  The source
dataset is non-commercial research material and remains ``SIM_ONLY``.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_combat_teacher import (
    OFFICIAL_GOALKEEPER_DEFAULT_QPOS,
)

_SOURCE_COMMIT = "976a81ff19b7306bafbe923d2890066b68a85271"
_LICENSE_HASH = "sha256:6c8cd1cdbe7accec4f63b6c3afb45ce0ffae9ed6abc0ca55acf5900b37970a82"
_MAPPING_HASH = "sha256:6b99f7217b2c5ce919b542697ef732a56535f11fa241a03fb29dcd5f1b689a79"
_CLIP_HASHES = {
    "left": "sha256:d88fa2d1e9df07191c370bc95f915d6a70e6e0625eaf28bbb43c4c2ad4ee11d4",
    "right": "sha256:980581b31e0bad141c5437cd4611d979079772f2261d7fe584c74023baf8e71c",
}
_DATASET_RELATIVE = Path("legged_gym/resources/datasets/goalkeeper")
_SOURCE_FPS = 30.0


class GoalkeeperDiveDirection(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class GoalkeeperDivePhase(IntEnum):
    """Environment-owned phase; an actor cannot declare itself safe."""

    READY = 0
    TAKEOFF = 1
    FLIGHT = 2
    LANDING = 3
    RECOVERY = 4
    COMPLETE = 5
    FAILED = 6


@dataclass(frozen=True)
class GoalkeeperDiveOptionConfig:
    """Content-bound envelope for a recoverable, simulation-only dive."""

    control_dt_sec: float = 0.02
    trigger_lateral_error_m: float = 0.62
    minimum_takeoff_lateral_speed_mps: float = 0.42
    standing_minimum_pelvis_height_m: float = 0.60
    standing_minimum_upright_projection: float = 0.78
    recovered_maximum_linear_speed_mps: float = 0.28
    recovered_maximum_angular_speed_rad_s: float = 0.55
    dive_minimum_pelvis_height_m: float = 0.28
    dive_minimum_upright_projection: float = 0.10
    dive_maximum_linear_speed_mps: float = 2.40
    dive_maximum_angular_speed_rad_s: float = 5.50
    maximum_option_duration_sec: float = 1.40
    recovery_hold_sec: float = 0.20
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_dive_option.v1"

    def __post_init__(self) -> None:
        values = (
            self.control_dt_sec,
            self.trigger_lateral_error_m,
            self.minimum_takeoff_lateral_speed_mps,
            self.standing_minimum_pelvis_height_m,
            self.standing_minimum_upright_projection,
            self.recovered_maximum_linear_speed_mps,
            self.recovered_maximum_angular_speed_rad_s,
            self.dive_minimum_pelvis_height_m,
            self.dive_minimum_upright_projection,
            self.dive_maximum_linear_speed_mps,
            self.dive_maximum_angular_speed_rad_s,
            self.maximum_option_duration_sec,
            self.recovery_hold_sec,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("goalkeeper dive settings must be finite and positive")
        if not 0.45 <= self.trigger_lateral_error_m <= 0.95:
            raise ValueError("goalkeeper dive trigger is outside the far-shot region")
        if not 0.20 <= self.dive_minimum_pelvis_height_m < self.standing_minimum_pelvis_height_m:
            raise ValueError("goalkeeper dive pelvis envelope is invalid")
        if not (
            0.0
            < self.dive_minimum_upright_projection
            < self.standing_minimum_upright_projection
            <= 1.0
        ):
            raise ValueError("goalkeeper dive upright envelope is invalid")
        if not 1.0 <= self.dive_maximum_linear_speed_mps <= 3.0:
            raise ValueError("goalkeeper dive linear-speed envelope is invalid")
        if not 3.0 <= self.dive_maximum_angular_speed_rad_s <= 6.0:
            raise ValueError("goalkeeper dive angular-speed envelope is invalid")
        if not 0.60 <= self.maximum_option_duration_sec <= 3.60:
            raise ValueError("goalkeeper dive duration is invalid")
        if not 0.10 <= self.recovery_hold_sec <= 0.50:
            raise ValueError("goalkeeper dive recovery hold is invalid")
        if self.recovery_hold_sec >= self.maximum_option_duration_sec:
            raise ValueError("goalkeeper dive recovery hold exceeds its option duration")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("goalkeeper dive option is SIM_ONLY")

    @property
    def maximum_option_steps(self) -> int:
        return int(math.ceil(self.maximum_option_duration_sec / self.control_dt_sec))

    @property
    def recovery_hold_steps(self) -> int:
        return int(math.ceil(self.recovery_hold_sec / self.control_dt_sec))

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class OfficialGoalkeeperDiveClip:
    """One resampled 29-DoF research demonstration."""

    direction: GoalkeeperDiveDirection
    joint_position_rad: NDArray[np.float64]
    joint_velocity_rad_s: NDArray[np.float64]
    root_displacement_m: NDArray[np.float64]
    root_quaternion_xyzw: NDArray[np.float64]
    source_frames: int
    source_fps: float
    resampled_fps: float
    source_hash: str

    def __post_init__(self) -> None:
        frames = self.joint_position_rad.shape[0]
        if self.joint_position_rad.shape != (frames, 29):
            raise ValueError("goalkeeper dive joint position must have shape (T, 29)")
        if self.joint_velocity_rad_s.shape != (frames, 29):
            raise ValueError("goalkeeper dive joint velocity must have shape (T, 29)")
        if self.root_displacement_m.shape != (frames, 3):
            raise ValueError("goalkeeper dive root displacement must have shape (T, 3)")
        if self.root_quaternion_xyzw.shape != (frames, 4):
            raise ValueError("goalkeeper dive root quaternion must have shape (T, 4)")
        arrays = (
            self.joint_position_rad,
            self.joint_velocity_rad_s,
            self.root_displacement_m,
            self.root_quaternion_xyzw,
        )
        if frames < 100 or any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("goalkeeper dive clip is too short or non-finite")
        if self.source_frames < 100 or self.source_fps != _SOURCE_FPS:
            raise ValueError("goalkeeper dive source timing changed")
        if not 40.0 <= self.resampled_fps <= 100.0:
            raise ValueError("goalkeeper dive resampling frequency is invalid")
        if not self.source_hash.startswith("sha256:"):
            raise ValueError("goalkeeper dive source hash is invalid")

    @property
    def duration_sec(self) -> float:
        frame_count = int(self.joint_position_rad.shape[0])
        return float((frame_count - 1) / self.resampled_fps)

    @property
    def maximum_lateral_displacement_m(self) -> float:
        values: list[float] = [abs(float(value)) for value in self.root_displacement_m[:, 1]]
        return max(values)


@dataclass(frozen=True)
class OfficialGoalkeeperDiveAtlas:
    source_commit: str
    source_license_hash: str
    mapping_hash: str
    joint_order: tuple[str, ...]
    clips: tuple[OfficialGoalkeeperDiveClip, ...]
    training_use_only: bool = True
    commercial_use_allowed: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.goalkeeper_dive_atlas.v1"

    def __post_init__(self) -> None:
        if self.source_commit != _SOURCE_COMMIT:
            raise ValueError("goalkeeper dive source commit changed")
        if self.source_license_hash != _LICENSE_HASH or self.mapping_hash != _MAPPING_HASH:
            raise ValueError("goalkeeper dive provenance changed")
        if self.joint_order != G1_DDS_JOINT_NAMES:
            raise ValueError("goalkeeper dive joint order changed")
        if {clip.direction for clip in self.clips} != set(GoalkeeperDiveDirection):
            raise ValueError("goalkeeper dive atlas requires left and right clips")
        if (
            not self.training_use_only
            or self.commercial_use_allowed
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("goalkeeper dive atlas must remain train-only SIM_ONLY")

    @property
    def atlas_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "source_license_hash": self.source_license_hash,
            "mapping_hash": self.mapping_hash,
            "joint_order": self.joint_order,
            "clips": [
                {
                    "direction": clip.direction.value,
                    "source_frames": clip.source_frames,
                    "source_fps": clip.source_fps,
                    "resampled_fps": clip.resampled_fps,
                    "source_hash": clip.source_hash,
                    "resampled_frames": clip.joint_position_rad.shape[0],
                    "duration_sec": clip.duration_sec,
                    "maximum_lateral_displacement_m": (clip.maximum_lateral_displacement_m),
                }
                for clip in self.clips
            ],
            "training_use_only": self.training_use_only,
            "commercial_use_allowed": self.commercial_use_allowed,
            "activation_ceiling": self.activation_ceiling,
        }
        return str(hash_json(payload))


_G1_MIRROR_ORDER = (
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
_G1_MIRROR_SIGN = (
    (
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
    )
    * 2
    + (-1.0, -1.0, 1.0)
    + (
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
    )
    * 2
)


@dataclass(frozen=True)
class GoalkeeperBalancedDiveSeed:
    """A symmetric, train-only seed extracted from the safer source side."""

    source_atlas_hash: str
    source_direction: GoalkeeperDiveDirection
    source_start_frame: int
    source_end_frame: int
    joint_position_rad: NDArray[np.float64]
    root_displacement_m: NDArray[np.float64]
    frame_rate_hz: float
    source_lateral_displacement_m: float
    window_profile: str = "maximum_lateral"
    training_use_only: bool = True
    commercial_use_allowed: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.goalkeeper_balanced_dive_seed.v1"

    def __post_init__(self) -> None:
        frames = self.source_end_frame - self.source_start_frame + 1
        if self.joint_position_rad.shape != (2, frames, 29):
            raise ValueError("balanced goalkeeper dive joints have an invalid shape")
        if self.root_displacement_m.shape != (2, frames, 3):
            raise ValueError("balanced goalkeeper dive roots have an invalid shape")
        arrays = (self.joint_position_rad, self.root_displacement_m)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("balanced goalkeeper dive seed must be finite")
        if not self.source_atlas_hash.startswith("sha256:"):
            raise ValueError("balanced goalkeeper dive source hash is invalid")
        if not 40.0 <= self.frame_rate_hz <= 100.0:
            raise ValueError("balanced goalkeeper dive frame rate is invalid")
        if not 0.30 <= self.source_lateral_displacement_m <= 0.80:
            raise ValueError("balanced goalkeeper dive displacement is invalid")
        if self.window_profile not in {"maximum_lateral", "low_vertical_dip"}:
            raise ValueError("balanced goalkeeper dive window profile is invalid")
        if (
            not self.training_use_only
            or self.commercial_use_allowed
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("balanced goalkeeper dive seed must remain train-only SIM_ONLY")

    @property
    def seed_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "source_atlas_hash": self.source_atlas_hash,
                    "source_direction": self.source_direction.value,
                    "source_start_frame": self.source_start_frame,
                    "source_end_frame": self.source_end_frame,
                    "frame_rate_hz": self.frame_rate_hz,
                    "source_lateral_displacement_m": self.source_lateral_displacement_m,
                    "window_profile": self.window_profile,
                    "joint_position_hash": hash_bytes(self.joint_position_rad.tobytes()),
                    "root_displacement_hash": hash_bytes(self.root_displacement_m.tobytes()),
                    "training_use_only": self.training_use_only,
                    "commercial_use_allowed": self.commercial_use_allowed,
                    "activation_ceiling": self.activation_ceiling,
                }
            )
        )


def mirror_g1_joint_positions(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mirror G1 DDS joint positions across the sagittal plane."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 29 or not np.all(np.isfinite(array)):
        raise ValueError("G1 mirror expects finite joint positions ending in 29 DoF")
    mirrored: NDArray[np.float64] = np.asarray(
        array[..., np.asarray(_G1_MIRROR_ORDER)] * np.asarray(_G1_MIRROR_SIGN),
        dtype=np.float64,
    )
    return mirrored


def build_balanced_dive_imitation_seed(
    atlas: OfficialGoalkeeperDiveAtlas,
    *,
    config: GoalkeeperDiveOptionConfig | None = None,
    window_profile: str = "maximum_lateral",
) -> GoalkeeperBalancedDiveSeed:
    """Extract the strongest bounded source window and manufacture its mirror.

    The two upstream sides are not assumed to have equal physical quality.
    The source window is selected by lateral displacement, while actual
    MuJoCo qualification remains a separate downstream gate.
    """

    active = config or GoalkeeperDiveOptionConfig()
    if atlas.activation_ceiling != "SIM_ONLY" or not atlas.training_use_only:
        raise ValueError("balanced dive seed requires a train-only SIM_ONLY atlas")
    left = next(
        (clip for clip in atlas.clips if clip.direction is GoalkeeperDiveDirection.LEFT),
        None,
    )
    if left is None:
        raise ValueError("balanced dive seed requires the left source clip")
    if not math.isclose(
        left.resampled_fps * active.control_dt_sec,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("balanced dive seed timing does not match the option controller")
    window_steps = active.maximum_option_steps
    if left.joint_position_rad.shape[0] <= window_steps:
        raise ValueError("balanced dive source is shorter than the option window")
    displacement = (
        left.root_displacement_m[window_steps:, 1] - left.root_displacement_m[:-window_steps, 1]
    )
    if window_profile == "maximum_lateral":
        score = displacement
    elif window_profile == "low_vertical_dip":
        vertical_dip = np.asarray(
            [
                min(
                    0.0,
                    float(
                        np.min(
                            left.root_displacement_m[index : index + window_steps + 1, 2]
                            - left.root_displacement_m[index, 2]
                        )
                    ),
                )
                for index in range(displacement.shape[0])
            ],
            dtype=np.float64,
        )
        end_rise = (
            left.root_displacement_m[window_steps:, 2] - left.root_displacement_m[:-window_steps, 2]
        )
        # Keep a useful lateral dive while selecting the demonstrated segment
        # that actually lowers the body.  The former lateral-only criterion
        # selected an upward jump window for low balls.
        score = displacement + 3.0 * (-vertical_dip) - 0.5 * np.maximum(end_rise, 0.0)
        score = np.where(displacement >= 0.30, score, -np.inf)
    else:
        raise ValueError("balanced goalkeeper dive window profile is invalid")
    start = int(np.argmax(score))
    end = start + window_steps
    lateral = float(displacement[start])
    source_joint = left.joint_position_rad[start : end + 1].copy()
    mirrored_joint = mirror_g1_joint_positions(source_joint)
    source_root = left.root_displacement_m[start : end + 1].copy()
    source_root -= source_root[0]
    mirrored_root = source_root.copy()
    mirrored_root[:, 1] *= -1.0
    joint = np.stack((source_joint, mirrored_joint), axis=0)
    root = np.stack((source_root, mirrored_root), axis=0)
    joint.setflags(write=False)
    root.setflags(write=False)
    return GoalkeeperBalancedDiveSeed(
        source_atlas_hash=atlas.atlas_hash,
        source_direction=GoalkeeperDiveDirection.LEFT,
        source_start_frame=start,
        source_end_frame=end,
        joint_position_rad=joint,
        root_displacement_m=root,
        frame_rate_hz=left.resampled_fps,
        source_lateral_displacement_m=lateral,
        window_profile=window_profile,
    )


def balanced_dive_qualified_impedance() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the exact 29-DoF impedance used by the CPU qualification.

    Keeping this contract in one function prevents an integration from
    replaying the qualified joint positions with unrelated locomotion gains
    or a non-zero velocity target and then treating the resulting fall as a
    failure of the motion source.
    """

    from rosclaw_soccer.training.goalkeeper_mjwarp import (
        _LOCO_KD,
        _LOCO_KP,
        _LOCO_TO_MOTOR,
    )

    order = np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)
    kp = np.zeros(29, dtype=np.float64)
    kd = np.zeros(29, dtype=np.float64)
    kp[order] = np.asarray(_LOCO_KP, dtype=np.float64)
    kd[order] = np.asarray(_LOCO_KD, dtype=np.float64)
    kp[12:] = np.asarray((150.0, 150.0, 150.0) + ((150.0,) * 4 + (20.0,) * 3) * 2)
    kd[12:] = np.asarray((2.0, 2.0, 2.0) + ((2.0,) * 4 + (0.5,) * 3) * 2)
    kp.setflags(write=False)
    kd.setflags(write=False)
    return kp, kd


def qualify_balanced_dive_seed_cpu_mujoco(
    *,
    asset_root: Path,
    source_checkout: Path,
    output_path: Path,
    joint_position_rad: NDArray[np.float64] | None = None,
    trajectory_kind: str = "balanced_imitation_seed",
    torque_limit_fraction: float = 1.0,
) -> dict[str, Any]:
    """Physics-qualify a balanced seed or its distilled trajectories.

    Passing an alternate trajectory never upgrades its authority: it remains
    a train-only, simulation-only candidate tied to the same source seed.
    """

    import json

    import mujoco

    from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS
    from rosclaw_soccer.training.goalkeeper_mjwarp import _LOCO_DEFAULT, _LOCO_TO_MOTOR
    from rosclaw_soccer.world.field import build_g1_stadium_model, g1_stadium_scene_hash

    atlas = load_official_goalkeeper_dive_atlas(checkout=source_checkout)
    seed = build_balanced_dive_imitation_seed(atlas)
    trajectory = (
        seed.joint_position_rad
        if joint_position_rad is None
        else np.asarray(joint_position_rad, dtype=np.float64)
    )
    if (
        trajectory.shape != seed.joint_position_rad.shape
        or not np.all(np.isfinite(trajectory))
        or not trajectory_kind
    ):
        raise ValueError("balanced goalkeeper dive trajectory is invalid")
    if not math.isfinite(torque_limit_fraction) or not 0.75 <= torque_limit_fraction <= 1.0:
        raise ValueError("balanced goalkeeper dive torque fraction must be in [0.75, 1]")
    model = build_g1_stadium_model(asset_root.expanduser().resolve())
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in G1_DDS_JOINT_NAMES],
        dtype=np.int64,
    )
    if np.any(joint_ids < 0):
        raise ValueError("balanced goalkeeper dive Body joint contract changed")
    joint_qpos = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    joint_limited = model.jnt_limited[joint_ids].astype(bool)
    joint_ranges = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    ready = np.zeros(29, dtype=np.float64)
    order = np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)
    ready[order] = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    kp, kd = balanced_dive_qualified_impedance()
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    blend_steps = 25
    outcomes: list[dict[str, Any]] = []
    for direction_index, direction in enumerate(
        (GoalkeeperDiveDirection.LEFT, GoalkeeperDiveDirection.RIGHT)
    ):
        direction_trajectory = trajectory[direction_index]
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = (4.52, 0.0, 0.793, 0.0, 0.0, 0.0, 1.0)
        data.qpos[7:36] = ready
        data.qpos[36:43] = (-20.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        targets = [
            ready + (direction_trajectory[0] - ready) * ((index + 1) / blend_steps)
            for index in range(blend_steps)
        ] + [frame for frame in direction_trajectory]
        minimum_pelvis = math.inf
        maximum_root_speed = 0.0
        maximum_root_angular_speed = 0.0
        maximum_torque_fraction = 0.0
        joint_limit_violation = False
        for target in targets:
            for _ in range(10):
                requested_torque = kp * (target - data.qpos[7:36]) - kd * data.qvel[6:35]
                maximum_torque_fraction = max(
                    maximum_torque_fraction,
                    float(np.max(np.abs(requested_torque) / limits)),
                )
                data.ctrl[:] = np.clip(
                    requested_torque,
                    -limits * torque_limit_fraction,
                    limits * torque_limit_fraction,
                )
                mujoco.mj_step(model, data)
                q = np.asarray(data.qpos[joint_qpos], dtype=np.float64)
                joint_limit_violation = joint_limit_violation or bool(
                    np.any(q[joint_limited] < joint_ranges[joint_limited, 0] - 1.0e-6)
                    or np.any(q[joint_limited] > joint_ranges[joint_limited, 1] + 1.0e-6)
                )
                minimum_pelvis = min(minimum_pelvis, float(data.qpos[2]))
                maximum_root_speed = max(maximum_root_speed, float(np.linalg.norm(data.qvel[:3])))
                maximum_root_angular_speed = max(
                    maximum_root_angular_speed,
                    float(np.linalg.norm(data.qvel[3:6])),
                )
        finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        outcomes.append(
            {
                "direction": direction.value,
                "finite_state": finite,
                "final_lateral_displacement_m": float(data.qpos[1]),
                "minimum_pelvis_height_m": minimum_pelvis,
                "maximum_root_speed_mps": maximum_root_speed,
                "maximum_root_angular_speed_rad_s": maximum_root_angular_speed,
                "maximum_requested_torque_fraction": maximum_torque_fraction,
                "joint_limit_violation": joint_limit_violation,
                "passed_training_seed_gate": bool(
                    finite
                    and not joint_limit_violation
                    and minimum_pelvis >= 0.55
                    and maximum_root_speed <= 1.50
                    and maximum_root_angular_speed <= 3.50
                    and abs(float(data.qpos[1])) >= 0.15
                ),
            }
        )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.goalkeeper_balanced_dive_cpu_exam.v2",
        "physics_backend": "mujoco_cpu",
        "physics_scene_hash": g1_stadium_scene_hash(asset_root.expanduser().resolve()),
        "dive_seed_hash": seed.seed_hash,
        "trajectory_kind": trajectory_kind,
        "trajectory_hash": hash_bytes(trajectory.tobytes()),
        "torque_limit_fraction": torque_limit_fraction,
        "source_atlas_hash": atlas.atlas_hash,
        "source_window": [seed.source_start_frame, seed.source_end_frame],
        "source_lateral_displacement_m": seed.source_lateral_displacement_m,
        "outcomes": outcomes,
        "passed": all(bool(item["passed_training_seed_gate"]) for item in outcomes),
        "authority": "TRAINING_REFERENCE_ONLY",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    report["report_hash"] = hash_json(report)
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report


@dataclass(frozen=True)
class GoalkeeperDiveStepResult:
    phase: NDArray[np.int64]
    posture_exception_granted: NDArray[np.bool_]
    unsafe: NDArray[np.bool_]
    recovered_event: NDArray[np.bool_]
    option_started_event: NDArray[np.bool_]


class GoalkeeperControlledDiveMonitor:
    """Fail-closed state machine for a learned dive option.

    The monitor, not the actor, owns phase transitions.  A request is accepted
    only for a far active shot.  Changing to a new shot before recovery,
    forbidden body contact, envelope violations, or overstaying the bounded
    option all fail immediately.
    """

    def __init__(
        self,
        environment_count: int,
        config: GoalkeeperDiveOptionConfig | None = None,
    ) -> None:
        if not 1 <= environment_count <= 262_144:
            raise ValueError("goalkeeper dive environment count is invalid")
        self.environment_count = environment_count
        self.config = config or GoalkeeperDiveOptionConfig()
        self.phase = np.full(environment_count, GoalkeeperDivePhase.READY, dtype=np.int64)
        self.active_shot = np.zeros(environment_count, dtype=np.int64)
        self._active_steps = np.zeros(environment_count, dtype=np.int64)
        self._recovery_steps = np.zeros(environment_count, dtype=np.int64)
        self.completed_dives = np.zeros(environment_count, dtype=np.int64)

    def reset(self, environment_ids: NDArray[np.int64] | None = None) -> None:
        ids = (
            np.arange(self.environment_count, dtype=np.int64)
            if environment_ids is None
            else np.asarray(environment_ids, dtype=np.int64)
        )
        if ids.ndim != 1 or np.any((ids < 0) | (ids >= self.environment_count)):
            raise ValueError("goalkeeper dive reset ids are invalid")
        self.phase[ids] = GoalkeeperDivePhase.READY
        self.active_shot[ids] = 0
        self._active_steps[ids] = 0
        self._recovery_steps[ids] = 0
        self.completed_dives[ids] = 0

    def step(
        self,
        *,
        option_request: NDArray[np.bool_],
        shot_index: NDArray[np.int64],
        lateral_intercept_error_m: NDArray[np.float64],
        pelvis_height_m: NDArray[np.float64],
        upright_projection: NDArray[np.float64],
        root_linear_speed_mps: NDArray[np.float64],
        root_angular_speed_rad_s: NDArray[np.float64],
        permitted_landing_contact: NDArray[np.bool_],
        forbidden_body_contact: NDArray[np.bool_],
    ) -> GoalkeeperDiveStepResult:
        arrays: dict[str, NDArray[Any]] = {
            "option_request": np.asarray(option_request),
            "shot_index": np.asarray(shot_index),
            "lateral_intercept_error_m": np.asarray(lateral_intercept_error_m),
            "pelvis_height_m": np.asarray(pelvis_height_m),
            "upright_projection": np.asarray(upright_projection),
            "root_linear_speed_mps": np.asarray(root_linear_speed_mps),
            "root_angular_speed_rad_s": np.asarray(root_angular_speed_rad_s),
            "permitted_landing_contact": np.asarray(permitted_landing_contact),
            "forbidden_body_contact": np.asarray(forbidden_body_contact),
        }
        if any(value.shape != (self.environment_count,) for value in arrays.values()):
            raise ValueError("goalkeeper dive step arrays must have shape (N,)")
        numeric = tuple(
            arrays[name]
            for name in (
                "lateral_intercept_error_m",
                "pelvis_height_m",
                "upright_projection",
                "root_linear_speed_mps",
                "root_angular_speed_rad_s",
            )
        )
        if any(not np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("goalkeeper dive step must be finite")
        shot = arrays["shot_index"].astype(np.int64, copy=False)
        if np.any((shot < 0) | (shot > 2)):
            raise ValueError("goalkeeper dive shot index must be in [0, 2]")

        cfg = self.config
        ready = self.phase == GoalkeeperDivePhase.READY
        request = arrays["option_request"].astype(np.bool_, copy=False)
        far_threat = np.abs(arrays["lateral_intercept_error_m"]) >= cfg.trigger_lateral_error_m
        started = ready & request & far_threat & (shot > 0)
        self.phase[started] = GoalkeeperDivePhase.TAKEOFF
        self.active_shot[started] = shot[started]
        self._active_steps[started] = 0
        self._recovery_steps[started] = 0

        active = np.isin(
            self.phase,
            (
                GoalkeeperDivePhase.TAKEOFF,
                GoalkeeperDivePhase.FLIGHT,
                GoalkeeperDivePhase.LANDING,
                GoalkeeperDivePhase.RECOVERY,
            ),
        )
        changed_shot = active & (shot > 0) & (shot != self.active_shot)
        self._active_steps[active] += 1

        pelvis = arrays["pelvis_height_m"]
        upright = arrays["upright_projection"]
        linear = arrays["root_linear_speed_mps"]
        angular = arrays["root_angular_speed_rad_s"]
        strict_standing = (
            (pelvis >= cfg.standing_minimum_pelvis_height_m)
            & (upright >= cfg.standing_minimum_upright_projection)
            & (linear <= cfg.recovered_maximum_linear_speed_mps)
            & (angular <= cfg.recovered_maximum_angular_speed_rad_s)
        )
        envelope_safe = (
            (pelvis >= cfg.dive_minimum_pelvis_height_m)
            & (upright >= cfg.dive_minimum_upright_projection)
            & (linear <= cfg.dive_maximum_linear_speed_mps)
            & (angular <= cfg.dive_maximum_angular_speed_rad_s)
            & ~arrays["forbidden_body_contact"].astype(np.bool_, copy=False)
        )
        timed_out = active & (self._active_steps > cfg.maximum_option_steps)
        failed = active & (~envelope_safe | changed_shot | timed_out)
        self.phase[failed] = GoalkeeperDivePhase.FAILED

        active = active & ~failed
        takeoff = active & (self.phase == GoalkeeperDivePhase.TAKEOFF)
        airborne = takeoff & (~strict_standing | (linear >= cfg.minimum_takeoff_lateral_speed_mps))
        self.phase[airborne] = GoalkeeperDivePhase.FLIGHT
        landing = (
            active
            & np.isin(
                self.phase,
                (GoalkeeperDivePhase.TAKEOFF, GoalkeeperDivePhase.FLIGHT),
            )
            & arrays["permitted_landing_contact"].astype(np.bool_, copy=False)
        )
        self.phase[landing] = GoalkeeperDivePhase.LANDING
        recovery = (
            active
            & np.isin(
                self.phase,
                (GoalkeeperDivePhase.FLIGHT, GoalkeeperDivePhase.LANDING),
            )
            & strict_standing
        )
        self.phase[recovery] = GoalkeeperDivePhase.RECOVERY

        recovering = active & (self.phase == GoalkeeperDivePhase.RECOVERY)
        self._recovery_steps = np.where(
            recovering & strict_standing,
            self._recovery_steps + 1,
            np.where(recovering, 0, self._recovery_steps),
        )
        lost_recovery = recovering & ~strict_standing
        self.phase[lost_recovery] = GoalkeeperDivePhase.LANDING
        recovered = recovering & strict_standing & (self._recovery_steps >= cfg.recovery_hold_steps)
        self.phase[recovered] = GoalkeeperDivePhase.COMPLETE
        self.completed_dives[recovered] += 1
        self.active_shot[recovered] = 0
        self._active_steps[recovered] = 0
        self._recovery_steps[recovered] = 0

        posture_exception = (
            np.isin(
                self.phase,
                (
                    GoalkeeperDivePhase.TAKEOFF,
                    GoalkeeperDivePhase.FLIGHT,
                    GoalkeeperDivePhase.LANDING,
                    GoalkeeperDivePhase.RECOVERY,
                ),
            )
            & envelope_safe
        )
        unsafe = self.phase == GoalkeeperDivePhase.FAILED
        result = GoalkeeperDiveStepResult(
            phase=self.phase.copy(),
            posture_exception_granted=posture_exception,
            unsafe=unsafe.copy(),
            recovered_event=recovered.copy(),
            option_started_event=started.copy(),
        )
        # COMPLETE is an observable one-step event.  The next request may then
        # start from READY, including from the keeper's non-central position.
        self.phase[self.phase == GoalkeeperDivePhase.COMPLETE] = GoalkeeperDivePhase.READY
        return result


def load_official_goalkeeper_dive_atlas(
    *,
    checkout: Path,
    resampled_fps: float = 50.0,
) -> OfficialGoalkeeperDiveAtlas:
    """Load pinned left/right demonstrations with ``weights_only=True``."""

    root = checkout.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Humanoid-Goalkeeper dive checkout is missing")
    commit = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != _SOURCE_COMMIT:
        raise ValueError("Humanoid-Goalkeeper dive checkout is not pinned")
    if not 40.0 <= resampled_fps <= 100.0 or not math.isfinite(resampled_fps):
        raise ValueError("goalkeeper dive resampling frequency is invalid")
    license_path = root / "LICENSE"
    mapping_path = root / _DATASET_RELATIVE / "joint_id.txt"
    if hash_bytes(license_path.read_bytes()) != _LICENSE_HASH:
        raise ValueError("Humanoid-Goalkeeper dive license changed")
    if hash_bytes(mapping_path.read_bytes()) != _MAPPING_HASH:
        raise ValueError("Humanoid-Goalkeeper dive joint mapping changed")
    mapping = _read_joint_mapping(mapping_path)

    clips = tuple(
        _load_clip(
            path=root / _DATASET_RELATIVE / f"{direction.value}jump.pt",
            direction=direction,
            mapping=mapping,
            resampled_fps=resampled_fps,
        )
        for direction in GoalkeeperDiveDirection
    )
    return OfficialGoalkeeperDiveAtlas(
        source_commit=commit,
        source_license_hash=_LICENSE_HASH,
        mapping_hash=_MAPPING_HASH,
        joint_order=G1_DDS_JOINT_NAMES,
        clips=clips,
    )


def _read_joint_mapping(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) != 2 or not fields[0].isdigit():
            raise ValueError("Humanoid-Goalkeeper dive joint mapping is malformed")
        index = int(fields[0])
        name = fields[1]
        if name in mapping or index in mapping.values():
            raise ValueError("Humanoid-Goalkeeper dive joint mapping is duplicated")
        mapping[name] = index
    if len(mapping) != 21 or sorted(mapping.values()) != list(range(21)):
        raise ValueError("Humanoid-Goalkeeper dive joint mapping is incomplete")
    if any(name not in G1_DDS_JOINT_NAMES for name in mapping):
        raise ValueError("Humanoid-Goalkeeper dive mapping contains an unknown joint")
    return mapping


def _load_clip(
    *,
    path: Path,
    direction: GoalkeeperDiveDirection,
    mapping: dict[str, int],
    resampled_fps: float,
) -> OfficialGoalkeeperDiveClip:
    expected_hash = _CLIP_HASHES[direction.value]
    if hash_bytes(path.read_bytes()) != expected_hash:
        raise ValueError(f"Humanoid-Goalkeeper {direction.value} dive clip changed")
    import importlib

    torch = importlib.import_module("torch")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Humanoid-Goalkeeper dive payload changed")
    required_shapes = {
        "base_pose": 4,
        "base_position": 3,
        "joint_position": 21,
        "joint_velocity": 21,
    }
    arrays: dict[str, NDArray[np.float64]] = {}
    frame_count: int | None = None
    for key, width in required_shapes.items():
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"Humanoid-Goalkeeper dive field {key} changed")
        array = value.detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Humanoid-Goalkeeper dive field {key} is non-finite")
        if frame_count is None:
            frame_count = int(array.shape[0])
        elif array.shape[0] != frame_count:
            raise ValueError("Humanoid-Goalkeeper dive frame counts differ")
        arrays[key] = array
    if frame_count is None or frame_count < 100:
        raise ValueError("Humanoid-Goalkeeper dive clip is too short")

    source_joint = arrays["joint_position"]
    full_joint = np.broadcast_to(
        np.asarray(OFFICIAL_GOALKEEPER_DEFAULT_QPOS, dtype=np.float64),
        (frame_count, 29),
    ).copy()
    for motor_index, name in enumerate(G1_DDS_JOINT_NAMES):
        source_index = mapping.get(name)
        if source_index is not None:
            full_joint[:, motor_index] = source_joint[:, source_index]
    root = arrays["base_position"] - arrays["base_position"][0]
    source_time = np.arange(frame_count, dtype=np.float64) / _SOURCE_FPS
    resampled_count = int(round(source_time[-1] * resampled_fps)) + 1
    target_time = np.arange(resampled_count, dtype=np.float64) / resampled_fps
    target_time[-1] = source_time[-1]
    joint = _interp_columns(source_time, full_joint, target_time)
    root = _interp_columns(source_time, root, target_time)
    quaternion = _interp_columns(source_time, arrays["base_pose"], target_time)
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1.0e-12)
    velocity = np.gradient(joint, target_time, axis=0, edge_order=2)
    for array in (joint, velocity, root, quaternion):
        array.setflags(write=False)
    return OfficialGoalkeeperDiveClip(
        direction=direction,
        joint_position_rad=joint,
        joint_velocity_rad_s=velocity,
        root_displacement_m=root,
        root_quaternion_xyzw=quaternion,
        source_frames=frame_count,
        source_fps=_SOURCE_FPS,
        resampled_fps=resampled_fps,
        source_hash=expected_hash,
    )


def _interp_columns(
    source_time: NDArray[np.float64],
    source: NDArray[np.float64],
    target_time: NDArray[np.float64],
) -> NDArray[np.float64]:
    result = np.empty((target_time.shape[0], source.shape[1]), dtype=np.float64)
    for column in range(source.shape[1]):
        result[:, column] = np.interp(target_time, source_time, source[:, column])
    return result


__all__ = [
    "GoalkeeperBalancedDiveSeed",
    "GoalkeeperControlledDiveMonitor",
    "GoalkeeperDiveDirection",
    "GoalkeeperDiveOptionConfig",
    "GoalkeeperDivePhase",
    "GoalkeeperDiveStepResult",
    "OfficialGoalkeeperDiveAtlas",
    "OfficialGoalkeeperDiveClip",
    "balanced_dive_qualified_impedance",
    "build_balanced_dive_imitation_seed",
    "load_official_goalkeeper_dive_atlas",
    "mirror_g1_joint_positions",
    "qualify_balanced_dive_seed_cpu_mujoco",
]
