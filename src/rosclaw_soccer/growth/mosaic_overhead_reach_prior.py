"""Content-bound MOSAIC SE50 overhead-reach prior for G1 simulation.

The artifact distilled here is a data prior, not a deployable controller.  It
is built from repeated overhead-set motions, converted from the raw Isaac Lab
tensor order, and independently checked by reconstructing source body poses in
MuJoCo.  Runtime consumers may only blend it through bounded PD targets while
the environment retains timing and safety ownership.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.mosaic_g1_contract import (
    MOSAIC_G1_CANONICAL_BODY_NAMES,
    MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
    canonicalize_mosaic_g1_bodies,
    canonicalize_mosaic_g1_joints,
)
from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_SOURCE_RELATIVE_PATH = "G1/optical_mocap/g1_SE50_stageii.npz"
_LEFT_HAND_BODY_INDEX = MOSAIC_G1_CANONICAL_BODY_NAMES.index("left_wrist_yaw_link")
_RIGHT_HAND_BODY_INDEX = MOSAIC_G1_CANONICAL_BODY_NAMES.index("right_wrist_yaw_link")


@dataclass(frozen=True)
class G1MosaicOverheadReachEvent:
    center_frame: int
    center_time_sec: float
    bilateral_hand_height_relative_pelvis_m: float
    left_hand_world_height_m: float
    right_hand_world_height_m: float
    leg_velocity_rms_rad_s: float
    root_planar_speed_mps: float

    def __post_init__(self) -> None:
        values = (
            self.center_time_sec,
            self.bilateral_hand_height_relative_pelvis_m,
            self.left_hand_world_height_m,
            self.right_hand_world_height_m,
            self.leg_velocity_rms_rad_s,
            self.root_planar_speed_mps,
        )
        if self.center_frame < 1 or not all(
            math.isfinite(value) and value >= 0.0 for value in values
        ):
            raise ValueError("MOSAIC overhead event is malformed")


@dataclass(frozen=True)
class G1MosaicOverheadReachPrior:
    dataset_readme_hash: str
    source_hash: str
    semantic_contract_hash: str
    physics_scene_hash: str
    body_hash: str
    joint_names: tuple[str, ...]
    reference_times_sec: tuple[float, ...]
    whole_body_position_reference_rad: tuple[tuple[float, ...], ...]
    whole_body_velocity_reference_rad_s: tuple[tuple[float, ...], ...]
    selected_events: tuple[G1MosaicOverheadReachEvent, ...]
    forward_kinematics_mean_error_m: float
    forward_kinematics_maximum_error_m: float
    reference_peak_bilateral_hand_height_m: float
    source_dataset: str = "MOSAIC"
    source_skill_id: str = "SE50"
    source_skill_description: str = "repeated_bimanual_overhead_set"
    dataset_license: str = "CDLA-Permissive-2.0"
    source_quaternion_order: str = "wxyz"
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.g1_mosaic_overhead_reach_prior.v1"

    def __post_init__(self) -> None:
        for value in (
            self.dataset_readme_hash,
            self.source_hash,
            self.semantic_contract_hash,
            self.physics_scene_hash,
            self.body_hash,
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError("MOSAIC overhead prior hashes must be sha256 content hashes")
        if self.semantic_contract_hash != MOSAIC_G1_SEMANTIC_CONTRACT_HASH:
            raise ValueError("MOSAIC overhead prior semantic contract is unsupported")
        if self.joint_names != G1_DDS_JOINT_NAMES:
            raise ValueError("MOSAIC overhead prior requires canonical G1 joints")
        times = np.asarray(self.reference_times_sec, dtype=np.float64)
        position = np.asarray(self.whole_body_position_reference_rad, dtype=np.float64)
        velocity = np.asarray(self.whole_body_velocity_reference_rad_s, dtype=np.float64)
        if (
            times.ndim != 1
            or len(times) < 41
            or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0.0)
            or not times[0] < 0.0 < times[-1]
            or position.shape != (len(times), 29)
            or velocity.shape != position.shape
            or not np.isfinite(position).all()
            or not np.isfinite(velocity).all()
        ):
            raise ValueError("MOSAIC overhead reference trajectory is malformed")
        if len(self.selected_events) < 8:
            raise ValueError("MOSAIC overhead prior requires at least eight repetitions")
        metrics = (
            self.forward_kinematics_mean_error_m,
            self.forward_kinematics_maximum_error_m,
            self.reference_peak_bilateral_hand_height_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in metrics):
            raise ValueError("MOSAIC overhead prior metrics are malformed")
        if self.forward_kinematics_maximum_error_m > 2.0e-4:
            raise ValueError("MOSAIC overhead source failed semantic reconstruction")
        if self.reference_peak_bilateral_hand_height_m < 1.30:
            raise ValueError("MOSAIC overhead source does not demonstrate high reach")
        if (
            self.source_dataset != "MOSAIC"
            or self.source_skill_id != "SE50"
            or self.dataset_license != "CDLA-Permissive-2.0"
            or self.source_quaternion_order != "wxyz"
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
        ):
            raise ValueError("MOSAIC overhead prior authority boundary is invalid")

    @property
    def prior_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "prior_hash": self.prior_hash}


def load_g1_mosaic_overhead_reach_prior(path: Path) -> G1MosaicOverheadReachPrior:
    """Load one no-pickle, content-bound overhead-reach artifact."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not 1 <= resolved.stat().st_size <= 8 * 1024 * 1024:
        raise ValueError("MOSAIC overhead prior is missing, empty, or too large")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MOSAIC overhead prior root must be an object")
    claimed_hash = payload.pop("prior_hash", None)
    try:
        payload["joint_names"] = tuple(payload["joint_names"])
        payload["reference_times_sec"] = tuple(payload["reference_times_sec"])
        payload["whole_body_position_reference_rad"] = tuple(
            tuple(row) for row in payload["whole_body_position_reference_rad"]
        )
        payload["whole_body_velocity_reference_rad_s"] = tuple(
            tuple(row) for row in payload["whole_body_velocity_reference_rad_s"]
        )
        payload["selected_events"] = tuple(
            G1MosaicOverheadReachEvent(**event) for event in payload["selected_events"]
        )
        prior = G1MosaicOverheadReachPrior(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MOSAIC overhead prior payload is invalid") from exc
    if claimed_hash != prior.prior_hash:
        raise ValueError("MOSAIC overhead prior content hash mismatch")
    return prior


def derive_g1_mosaic_overhead_reach_prior(
    *,
    mosaic_root: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
) -> G1MosaicOverheadReachPrior:
    """Distil repeated SE50 reaches after an independent MuJoCo semantic proof."""

    root = mosaic_root.expanduser().resolve()
    assets = asset_root.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("MOSAIC overhead prior must be outside the source checkout")
    if output.exists():
        raise ValueError("MOSAIC overhead prior output already exists")
    readme = root / "README.md"
    if (
        not readme.is_file()
        or "license: cdla-permissive-2.0" not in readme.read_text(encoding="utf-8").lower()
    ):
        raise ValueError("MOSAIC README with CDLA-Permissive-2.0 is required")
    source = root / _SOURCE_RELATIVE_PATH
    if not source.is_file():
        raise ValueError("MOSAIC SE50 G1 source is missing")
    with np.load(source, allow_pickle=False) as data:
        required = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
        }
        if not required.issubset(data.files):
            raise ValueError("MOSAIC SE50 source lacks required arrays")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        raw_position = np.asarray(data["joint_pos"], dtype=np.float64)
        raw_velocity = np.asarray(data["joint_vel"], dtype=np.float64)
        raw_body_position = np.asarray(data["body_pos_w"], dtype=np.float64)
        raw_body_quaternion = np.asarray(data["body_quat_w"], dtype=np.float64)
        raw_body_velocity = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
    frame_count = raw_position.shape[0]
    if (
        not 30.0 <= fps <= 240.0
        or raw_position.shape != (frame_count, 29)
        or raw_velocity.shape != raw_position.shape
        or raw_body_position.shape != (frame_count, 30, 3)
        or raw_body_quaternion.shape != (frame_count, 30, 4)
        or raw_body_velocity.shape != raw_body_position.shape
        or not all(
            np.isfinite(value).all()
            for value in (
                raw_position,
                raw_velocity,
                raw_body_position,
                raw_body_quaternion,
                raw_body_velocity,
            )
        )
    ):
        raise ValueError("MOSAIC SE50 source shape, clock, or values are invalid")
    position = canonicalize_mosaic_g1_joints(raw_position)
    velocity = canonicalize_mosaic_g1_joints(raw_velocity)
    body_position = canonicalize_mosaic_g1_bodies(raw_body_position)
    body_quaternion = canonicalize_mosaic_g1_bodies(raw_body_quaternion)
    body_velocity = canonicalize_mosaic_g1_bodies(raw_body_velocity)
    scene = assets / "g1_description" / "scene_with_ball.xml"
    mean_fk_error, maximum_fk_error = _verify_forward_kinematics(
        scene=scene,
        position=position,
        body_position=body_position,
        body_quaternion=body_quaternion,
    )
    if maximum_fk_error > 2.0e-4:
        raise ValueError("MOSAIC SE50 joint/body semantics failed MuJoCo reconstruction")

    root_position = body_position[:, 0]
    left_hand = body_position[:, _LEFT_HAND_BODY_INDEX]
    right_hand = body_position[:, _RIGHT_HAND_BODY_INDEX]
    bilateral_height = np.minimum(left_hand[:, 2], right_hand[:, 2])
    relative_bilateral_height = np.minimum(
        left_hand[:, 2] - root_position[:, 2],
        right_hand[:, 2] - root_position[:, 2],
    )
    leg_speed = np.sqrt(np.mean(np.square(velocity[:, :12]), axis=1))
    root_speed = np.linalg.norm(body_velocity[:, 0, :2], axis=1)
    score = relative_bilateral_height - 0.025 * leg_speed - 0.05 * root_speed
    local_radius = max(10, int(round(0.30 * fps)))
    candidates = [
        frame
        for frame in range(local_radius, frame_count - local_radius)
        if relative_bilateral_height[frame] >= 0.52
        and score[frame]
        >= float(np.max(score[frame - local_radius : frame + local_radius + 1]))
    ]
    separation = int(round(1.50 * fps))
    selected: list[int] = []
    for frame in sorted(candidates, key=lambda item: float(score[item]), reverse=True):
        if all(abs(frame - previous) >= separation for previous in selected):
            selected.append(frame)
        if len(selected) == 16:
            break
    selected.sort()
    if len(selected) < 8:
        raise ValueError("MOSAIC SE50 lacks eight stable, separated overhead reaches")
    times = np.linspace(-0.56, 0.66, 62, dtype=np.float64)
    position_windows: list[NDArray[np.float64]] = []
    velocity_windows: list[NDArray[np.float64]] = []
    events: list[G1MosaicOverheadReachEvent] = []
    for center in selected:
        indices = np.rint(center + times * fps).astype(np.int64)
        if int(indices[0]) < 0 or int(indices[-1]) >= frame_count:
            raise ValueError("MOSAIC overhead event window exceeds the source")
        position_windows.append(position[indices])
        velocity_windows.append(velocity[indices])
        events.append(
            G1MosaicOverheadReachEvent(
                center_frame=center,
                center_time_sec=center / fps,
                bilateral_hand_height_relative_pelvis_m=float(
                    relative_bilateral_height[center]
                ),
                left_hand_world_height_m=float(left_hand[center, 2]),
                right_hand_world_height_m=float(right_hand[center, 2]),
                leg_velocity_rms_rad_s=float(leg_speed[center]),
                root_planar_speed_mps=float(root_speed[center]),
            )
        )
    reference_position = np.median(np.asarray(position_windows), axis=0)
    reference_velocity = np.median(np.asarray(velocity_windows), axis=0)
    if np.max(np.abs(reference_position)) > 3.20:
        raise ValueError("MOSAIC overhead reference exceeds the bounded G1 pose contract")
    prior = G1MosaicOverheadReachPrior(
        dataset_readme_hash=_file_hash(readme),
        source_hash=_file_hash(source),
        semantic_contract_hash=MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
        physics_scene_hash=hash_bytes(scene.read_bytes()),
        body_hash=g1_body_hash(assets),
        joint_names=G1_DDS_JOINT_NAMES,
        reference_times_sec=tuple(float(value) for value in times),
        whole_body_position_reference_rad=tuple(
            tuple(float(value) for value in row) for row in reference_position
        ),
        whole_body_velocity_reference_rad_s=tuple(
            tuple(float(value) for value in row) for row in reference_velocity
        ),
        selected_events=tuple(events),
        forward_kinematics_mean_error_m=mean_fk_error,
        forward_kinematics_maximum_error_m=maximum_fk_error,
        reference_peak_bilateral_hand_height_m=float(np.median(bilateral_height[selected])),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(prior.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return prior


def blend_g1_mosaic_overhead_reach_target(
    *,
    target: NDArray[np.float64],
    prior: G1MosaicOverheadReachPrior,
    time_to_arrival_sec: float,
    target_height_m: float,
    blend: float,
    minimum_target_height_m: float = 1.10,
    full_target_height_m: float = 1.25,
    joint_scales: tuple[float, ...] = (0.0,) * 12 + (0.25,) * 3 + (1.0,) * 14,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Blend a smooth, height-conditioned SE50 pose through a PD target."""

    base = np.asarray(target, dtype=np.float64)
    if base.shape != (29,) or not np.isfinite(base).all():
        raise ValueError("MOSAIC overhead target must contain 29 finite joints")
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("MOSAIC overhead blend must be in [0, 1]")
    if (
        not math.isfinite(time_to_arrival_sec)
        or not math.isfinite(target_height_m)
        or not 0.80 <= minimum_target_height_m < full_target_height_m <= 1.60
    ):
        raise ValueError("MOSAIC overhead timing or height contract is invalid")
    scales = np.asarray(joint_scales, dtype=np.float64)
    if (
        scales.shape != (29,)
        or not np.isfinite(scales).all()
        or np.any((scales < 0) | (scales > 1))
    ):
        raise ValueError("MOSAIC overhead joint scales must contain 29 values in [0, 1]")
    relative_time = -time_to_arrival_sec
    times = np.asarray(prior.reference_times_sec, dtype=np.float64)
    if (
        blend == 0.0
        or target_height_m < minimum_target_height_m
        or relative_time < times[0]
        or relative_time > times[-1]
    ):
        return base.copy(), np.zeros(29, dtype=np.float64), 0.0
    height_gate = _smoothstep(
        (target_height_m - minimum_target_height_m)
        / (full_target_height_m - minimum_target_height_m)
    )
    rise = _smoothstep((relative_time - times[0]) / -times[0])
    hold_end = min(0.18, float(times[-1]) - 0.05)
    decay = 1.0 - _smoothstep((relative_time - hold_end) / (times[-1] - hold_end))
    gate = blend * height_gate * min(rise, decay)
    reference = np.asarray(prior.whole_body_position_reference_rad, dtype=np.float64)
    desired = np.asarray(
        [np.interp(relative_time, times, reference[:, joint]) for joint in range(29)],
        dtype=np.float64,
    )
    maximum = np.asarray((0.35,) * 12 + (0.45,) * 3 + (2.80,) * 14)
    correction = gate * scales * np.clip(desired - base, -maximum, maximum)
    return base + correction, correction, float(gate)


def _verify_forward_kinematics(
    *,
    scene: Path,
    position: NDArray[np.float64],
    body_position: NDArray[np.float64],
    body_quaternion: NDArray[np.float64],
) -> tuple[float, float]:
    import mujoco

    if not scene.is_file():
        raise ValueError("G1 MuJoCo semantic-validation scene is missing")
    model = mujoco.MjModel.from_xml_path(str(scene))
    body_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index))
        for index in range(1, 31)
    )
    joint_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index))
        for index in range(1, 30)
    )
    if body_names != MOSAIC_G1_CANONICAL_BODY_NAMES or joint_names != G1_DDS_JOINT_NAMES:
        raise ValueError("G1 MuJoCo scene does not match the canonical MOSAIC contract")
    data = mujoco.MjData(model)
    errors: list[float] = []
    for frame in np.linspace(0, len(position) - 1, 9, dtype=np.int64):
        data.qpos[:3] = body_position[frame, 0]
        # Isaac Lab exports root/body quaternions as wxyz for this dataset.
        data.qpos[3:7] = body_quaternion[frame, 0]
        data.qpos[7:36] = position[frame]
        mujoco.mj_forward(model, data)
        errors.extend(
            np.linalg.norm(data.xpos[1:31] - body_position[frame], axis=1).tolist()
        )
    return float(np.mean(errors)), float(np.max(errors))


def _smoothstep(value: float) -> float:
    clipped = min(1.0, max(0.0, float(value)))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "G1MosaicOverheadReachEvent",
    "G1MosaicOverheadReachPrior",
    "blend_g1_mosaic_overhead_reach_target",
    "derive_g1_mosaic_overhead_reach_prior",
    "load_g1_mosaic_overhead_reach_prior",
]
