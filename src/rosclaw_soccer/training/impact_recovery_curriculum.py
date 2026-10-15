"""Content-bound impact-recovery curriculum from complete soccer episodes.

The full four-player simulation is the causal source of impact states, but it
is far too expensive to use as the inner loop of every policy update.  This
module extracts a compact, pickle-free curriculum at the goalkeeper boundary:
root/joint state, support, contact-relative time, target heading and a frozen
successful joint-memory route.  Failed episodes contribute reset states only;
they are never mislabeled as teachers.

Historical evidence is verified against its own immutable request and
trajectory bytes.  Its old source closure is preserved as provenance rather
than recomputed with today's checkout.  Consequently, this corpus is training
input, not a retroactive promotion claim.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash, trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

ImpactRecoveryUse = Literal["RETENTION_ANCHOR", "ACQUISITION_FAILURE"]

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_JOINT_COUNT = 29
_ROOT_QPOS = 7
_ROOT_QVEL = 6


@dataclass(frozen=True)
class ImpactRecoverySource:
    """One immutable full-chain episode admitted to the curriculum."""

    source_id: str
    evidence_path: Path
    use: ImpactRecoveryUse
    expected_success: bool

    def __post_init__(self) -> None:
        if (
            not self.source_id
            or len(self.source_id) > 96
            or not self.source_id.replace("-", "").replace("_", "").isalnum()
            or self.use not in {"RETENTION_ANCHOR", "ACQUISITION_FAILURE"}
            or (self.use == "RETENTION_ANCHOR") is not self.expected_success
        ):
            raise ValueError("impact-recovery source contract is invalid")


@dataclass(frozen=True)
class ImpactRecoveryCurriculumConfig:
    """Bounded extraction and frozen-memory settings."""

    first_offset_sec: float = 0.80
    last_offset_sec: float = 5.00
    sample_stride_sec: float = 0.10
    control_dt_sec: float = 0.02
    memory_horizon_steps: int = 250
    desired_heading_rad: float = math.pi
    dynamic_gain_memory_enabled: bool = False
    teacher_state_memory_enabled: bool = False
    teacher_phase_search_radius_sec: float = 0.0
    teacher_phase_deviation_penalty: float = 0.05
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_curriculum_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.first_offset_sec,
            self.last_offset_sec,
            self.sample_stride_sec,
            self.control_dt_sec,
            self.desired_heading_rad,
            self.teacher_phase_search_radius_sec,
            self.teacher_phase_deviation_penalty,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or not 0.0 <= self.first_offset_sec < self.last_offset_sec <= 8.0
            or not 0.02 <= self.sample_stride_sec <= 1.0
            or not 0.002 <= self.control_dt_sec <= 0.05
            or not 8 <= self.memory_horizon_steps <= 1_000
            or not isinstance(self.dynamic_gain_memory_enabled, bool)
            or not isinstance(self.teacher_state_memory_enabled, bool)
            or (self.teacher_state_memory_enabled and not self.dynamic_gain_memory_enabled)
            or not 0.0 <= self.teacher_phase_search_radius_sec <= 3.0
            or not 0.0 <= self.teacher_phase_deviation_penalty <= 1.0
            or (
                self.teacher_phase_search_radius_sec > 0.0 and not self.teacher_state_memory_enabled
            )
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery curriculum config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class _BoundEpisode:
    source: ImpactRecoverySource
    evidence_file_hash: str
    evidence_report_hash: str
    request_file_hash: str
    request_config_hash: str
    reproducibility_closure_hash: str
    trajectory_file_hash: str
    trajectory_semantic_digest: str
    trajectory: dict[str, NDArray[Any]]
    contact_index: int
    succeeded: bool


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return cast(dict[str, Any], value)


def _load_historical_episode(source: ImpactRecoverySource) -> _BoundEpisode:
    """Verify historical bytes without pretending the old source is current."""

    evidence_path = source.evidence_path.expanduser().resolve()
    request_path = evidence_path.parent / "request.json"
    if not evidence_path.is_file() or not request_path.is_file():
        raise FileNotFoundError("impact-recovery source evidence is incomplete")
    evidence = _load_json(evidence_path)
    request = _load_json(request_path)
    recorded_report_hash = evidence.pop("report_hash", None)
    try:
        replays = evidence.get("replays")
        config = request.get("config")
        if (
            request.get("schema_version")
            != "rosclaw_soccer.role_isolated_second_striker_probe_request.v1"
            or not isinstance(config, dict)
            or request.get("config_hash") != hash_json(config)
            or request.get("activation_ceiling") != "SIM_ONLY"
            or request.get("hardware_command_sent") is not False
            or evidence.get("schema_version")
            != "rosclaw_soccer.role_isolated_second_striker_probe_evidence.v1"
            or evidence.get("activation_ceiling") != "SIM_ONLY"
            or evidence.get("hardware_command_sent") is not False
            or evidence.get("pixels_used_for_scoring") is not False
            or not isinstance(replays, list)
            or len(replays) != 2
            or not isinstance(recorded_report_hash, str)
            or recorded_report_hash != hash_json(evidence)
        ):
            raise ValueError("impact-recovery historical authority contract is invalid")
        request_hash = hash_bytes(request_path.read_bytes())
        closure_hash = str(request.get("reproducibility_closure_hash", ""))
        if (
            evidence.get("request_hash") != request_hash
            or evidence.get("reproducibility_closure_hash") != closure_hash
            or not _SHA256.fullmatch(closure_hash)
        ):
            raise ValueError("impact-recovery historical request binding changed")

        loaded: list[tuple[dict[str, Any], dict[str, NDArray[Any]], str, str]] = []
        for replay in replays:
            if not isinstance(replay, dict):
                raise ValueError("impact-recovery historical replay is invalid")
            trajectory_path = evidence_path.parent / str(replay.get("trajectory_file", ""))
            file_hash = hash_bytes(trajectory_path.read_bytes())
            if replay.get("trajectory_hash") != file_hash:
                raise ValueError("impact-recovery trajectory bytes changed")
            with np.load(trajectory_path, allow_pickle=False) as archive:
                trajectory = {name: np.asarray(archive[name]) for name in archive.files}
            digest = trajectory_digest(trajectory)
            if replay.get("trajectory_digest") != digest:
                raise ValueError("impact-recovery trajectory semantics changed")
            loaded.append((cast(dict[str, Any], replay), trajectory, file_hash, digest))
        first_replay, first_trajectory, trajectory_hash, semantic_digest = loaded[0]
        second_replay, second_trajectory, _, second_digest = loaded[1]
        if (
            semantic_digest != second_digest
            or first_replay.get("result") != second_replay.get("result")
            or first_replay.get("evaluation") != second_replay.get("evaluation")
            or first_replay.get("candidate_diagnostics")
            != second_replay.get("candidate_diagnostics")
            or set(first_trajectory) != set(second_trajectory)
        ):
            raise ValueError("impact-recovery source does not have strict paired replay")
        required = {
            "time",
            "second_ball_pose",
            "second_ball_velocity",
            "goalkeeper_pelvis_pose",
            "goalkeeper_root_velocity",
            "goalkeeper_foot_contact",
            "goalkeeper_joint_position",
            "goalkeeper_joint_velocity",
            "goalkeeper_executed_torque",
            "goalkeeper_policy_action",
            "goalkeeper_second_ball_contact",
        }
        if not required.issubset(first_trajectory):
            raise ValueError("impact-recovery trajectory state is incomplete")
        time = np.asarray(first_trajectory["time"], dtype=np.float64)
        contact = np.asarray(first_trajectory["goalkeeper_second_ball_contact"], dtype=np.bool_)
        row_count = time.size
        shaped = {
            "second_ball_pose": (row_count, 7),
            "second_ball_velocity": (row_count, 6),
            "goalkeeper_pelvis_pose": (row_count, 7),
            "goalkeeper_root_velocity": (row_count, 6),
            "goalkeeper_foot_contact": (row_count, 2),
            "goalkeeper_joint_position": (row_count, _JOINT_COUNT),
            "goalkeeper_joint_velocity": (row_count, _JOINT_COUNT),
            "goalkeeper_executed_torque": (row_count, _JOINT_COUNT),
            "goalkeeper_policy_action": (row_count, _JOINT_COUNT),
        }
        gain_names = {"goalkeeper_kp", "goalkeeper_kd"}
        available_gains = gain_names.intersection(first_trajectory)
        if available_gains and available_gains != gain_names:
            raise ValueError("impact-recovery trajectory gain state is incomplete")
        if available_gains:
            shaped.update({name: (row_count, _JOINT_COUNT) for name in gain_names})
        if (
            time.ndim != 1
            or row_count < 10
            or contact.shape != (row_count,)
            or not np.any(contact)
            or any(
                np.asarray(first_trajectory[name]).shape != shape for name, shape in shaped.items()
            )
            or any(
                not np.all(np.isfinite(np.asarray(first_trajectory[name], dtype=np.float64)))
                for name in shaped
            )
        ):
            raise ValueError("impact-recovery trajectory arrays are invalid")
        succeeded = bool(evidence.get("candidate_promoted") is True)
        if succeeded is not source.expected_success:
            raise ValueError("impact-recovery source outcome does not match its declared use")
        return _BoundEpisode(
            source=source,
            evidence_file_hash=hash_bytes(evidence_path.read_bytes()),
            evidence_report_hash=recorded_report_hash,
            request_file_hash=request_hash,
            request_config_hash=str(request["config_hash"]),
            reproducibility_closure_hash=closure_hash,
            trajectory_file_hash=trajectory_hash,
            trajectory_semantic_digest=semantic_digest,
            trajectory=first_trajectory,
            contact_index=int(np.flatnonzero(contact)[0]),
            succeeded=succeeded,
        )
    finally:
        if recorded_report_hash is not None:
            evidence["report_hash"] = recorded_report_hash


def _yaw_error(quaternion_wxyz: NDArray[np.float64], desired_heading_rad: float) -> float:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not math.isfinite(norm) or norm <= 1.0e-8:
        raise ValueError("impact-recovery root quaternion is invalid")
    w, x, y, z = quaternion / norm
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.atan2(
        math.sin(desired_heading_rad - yaw),
        math.cos(desired_heading_rad - yaw),
    )


def _projected_gravity(quaternion_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not math.isfinite(norm) or norm <= 1.0e-8:
        raise ValueError("impact-recovery root quaternion is invalid")
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return rotation.T @ np.asarray((0.0, 0.0, -1.0), dtype=np.float64)


def _sample_indexes(episode: _BoundEpisode, config: ImpactRecoveryCurriculumConfig) -> list[int]:
    time = np.asarray(episode.trajectory["time"], dtype=np.float64)
    contact_time = float(time[episode.contact_index])
    if contact_time + config.last_offset_sec > float(time[-1]) + 1.0e-9:
        raise ValueError("impact-recovery sampling exceeds episode duration")
    offsets = np.arange(
        config.first_offset_sec,
        config.last_offset_sec + 0.5 * config.sample_stride_sec,
        config.sample_stride_sec,
        dtype=np.float64,
    )
    indexes = [int(np.argmin(np.abs(time - (contact_time + offset)))) for offset in offsets]
    if len(indexes) != len(set(indexes)):
        raise ValueError("impact-recovery sampling aliases multiple offsets to one frame")
    return indexes


def _route_distance(
    trajectory: dict[str, NDArray[Any]],
    index: int,
    reference: dict[str, NDArray[Any]],
    reference_index: int,
    *,
    desired_heading_rad: float,
    dynamics_complete: bool,
) -> float:
    pose = np.asarray(trajectory["goalkeeper_pelvis_pose"][index], dtype=np.float64)
    ref_pose = np.asarray(reference["goalkeeper_pelvis_pose"][reference_index], dtype=np.float64)
    velocity = np.asarray(trajectory["goalkeeper_root_velocity"][index], dtype=np.float64)
    ref_velocity = np.asarray(
        reference["goalkeeper_root_velocity"][reference_index], dtype=np.float64
    )
    joints = np.asarray(trajectory["goalkeeper_joint_position"][index, :15], dtype=np.float64)
    ref_joints = np.asarray(
        reference["goalkeeper_joint_position"][reference_index, :15], dtype=np.float64
    )
    yaw = _yaw_error(pose[3:7], desired_heading_rad)
    ref_yaw = _yaw_error(ref_pose[3:7], desired_heading_rad)
    distance = float(
        ((yaw - ref_yaw) / math.pi) ** 2
        + np.mean(((velocity - ref_velocity) / np.asarray((1, 1, 1, 2, 2, 2))) ** 2)
        + np.mean(((joints - ref_joints) / 0.5) ** 2)
    )
    if not dynamics_complete:
        return distance
    gravity = _projected_gravity(pose[3:7])
    ref_gravity = _projected_gravity(ref_pose[3:7])
    joint_velocity = np.asarray(
        trajectory["goalkeeper_joint_velocity"][index, :15], dtype=np.float64
    )
    ref_joint_velocity = np.asarray(
        reference["goalkeeper_joint_velocity"][reference_index, :15], dtype=np.float64
    )
    next_index = min(reference_index + 1, len(reference["goalkeeper_policy_action"]) - 1)
    action = np.asarray(trajectory["goalkeeper_policy_action"][index], dtype=np.float64)
    next_action = np.asarray(reference["goalkeeper_policy_action"][next_index], dtype=np.float64)
    continuity = float(np.mean(((action - next_action) / 0.25) ** 2))
    gain_continuity = 0.0
    if all(name in trajectory and name in reference for name in ("goalkeeper_kp", "goalkeeper_kd")):
        kp = np.asarray(trajectory["goalkeeper_kp"][index], dtype=np.float64)
        kd = np.asarray(trajectory["goalkeeper_kd"][index], dtype=np.float64)
        next_kp = np.asarray(reference["goalkeeper_kp"][next_index], dtype=np.float64)
        next_kd = np.asarray(reference["goalkeeper_kd"][next_index], dtype=np.float64)
        gain_continuity = float(
            np.mean(((kp - next_kp) / 20.0) ** 2) + np.mean(((kd - next_kd) / 1.0) ** 2)
        )
    return float(
        distance
        + ((pose[2] - ref_pose[2]) / 0.10) ** 2
        + np.mean(((gravity - ref_gravity) / 0.25) ** 2)
        + np.mean(((joint_velocity - ref_joint_velocity) / 3.0) ** 2)
        + 0.50 * continuity
        + 0.20 * gain_continuity
    )


def _memory_route(
    episode: _BoundEpisode,
    index: int,
    successful: Sequence[_BoundEpisode],
    *,
    config: ImpactRecoveryCurriculumConfig,
) -> tuple[
    int,
    int,
    float,
    NDArray[np.float32],
    NDArray[np.float32] | None,
    NDArray[np.float32] | None,
    NDArray[np.float32] | None,
    NDArray[np.float32] | None,
]:
    time = np.asarray(episode.trajectory["time"], dtype=np.float64)
    elapsed = float(time[index] - time[episode.contact_index])
    candidates: list[tuple[float, int, int, float]] = []
    for route_index, reference in enumerate(successful):
        reference_time = np.asarray(reference.trajectory["time"], dtype=np.float64)
        reference_contact_time = float(reference_time[reference.contact_index])
        target_time = reference_contact_time + elapsed
        center = int(np.argmin(np.abs(reference_time - target_time)))
        if config.teacher_phase_search_radius_sec > 0.0:
            eligible = np.flatnonzero(
                (reference_time >= target_time - config.teacher_phase_search_radius_sec)
                & (reference_time <= target_time + config.teacher_phase_search_radius_sec)
                & (np.arange(reference_time.size) >= reference.contact_index)
            )
            search_indexes = eligible.tolist() or [center]
        else:
            search_indexes = [center]
        for reference_index in search_indexes:
            reference_elapsed = float(reference_time[reference_index] - reference_contact_time)
            phase_offset = reference_elapsed - elapsed
            phase_cost = (
                config.teacher_phase_deviation_penalty
                * (phase_offset / config.teacher_phase_search_radius_sec) ** 2
                if config.teacher_phase_search_radius_sec > 0.0
                else 0.0
            )
            candidates.append(
                (
                    _route_distance(
                        episode.trajectory,
                        index,
                        reference.trajectory,
                        reference_index,
                        desired_heading_rad=config.desired_heading_rad,
                        dynamics_complete=config.teacher_phase_search_radius_sec > 0.0,
                    )
                    + phase_cost,
                    route_index,
                    reference_index,
                    phase_offset,
                )
            )
    _, route_index, start, phase_offset = min(candidates)
    motor_target = np.asarray(
        successful[route_index].trajectory["goalkeeper_policy_action"], dtype=np.float32
    )
    # Trace rows are appended after the target has already driven the physics
    # transition into that row.  Continuing from the recorded qpos/qvel must
    # therefore start with the next target, not replay the already-consumed
    # target and introduce a one-control-period causal delay.
    steps = np.arange(config.memory_horizon_steps, dtype=np.int64)
    source_indexes = np.minimum(start + 1 + steps, motor_target.shape[0] - 1)
    state_indexes = np.minimum(start + steps, motor_target.shape[0] - 1)
    if config.dynamic_gain_memory_enabled:
        reference_trajectory = successful[route_index].trajectory
        if (
            "goalkeeper_kp" not in reference_trajectory
            or "goalkeeper_kd" not in reference_trajectory
        ):
            raise ValueError("impact-recovery dynamic gain teacher is unavailable")
        memory_kp = np.asarray(
            reference_trajectory["goalkeeper_kp"][source_indexes], dtype=np.float32
        )
        memory_kd = np.asarray(
            reference_trajectory["goalkeeper_kd"][source_indexes], dtype=np.float32
        )
    else:
        memory_kp = None
        memory_kd = None
    if config.teacher_state_memory_enabled:
        reference_trajectory = successful[route_index].trajectory
        memory_qpos = np.concatenate(
            (
                reference_trajectory["goalkeeper_pelvis_pose"][state_indexes],
                reference_trajectory["goalkeeper_joint_position"][state_indexes],
            ),
            axis=1,
        ).astype(np.float32)
        memory_qvel = np.concatenate(
            (
                reference_trajectory["goalkeeper_root_velocity"][state_indexes],
                reference_trajectory["goalkeeper_joint_velocity"][state_indexes],
            ),
            axis=1,
        ).astype(np.float32)
    else:
        memory_qpos = None
        memory_qvel = None
    return (
        route_index,
        start,
        phase_offset,
        np.asarray(motor_target[source_indexes], dtype=np.float32),
        memory_kp,
        memory_kd,
        memory_qpos,
        memory_qvel,
    )


def build_impact_recovery_curriculum(
    *,
    sources: Sequence[ImpactRecoverySource],
    asset_root: Path,
    source_checkout: Path,
    output_dir: Path,
    config: ImpactRecoveryCurriculumConfig | None = None,
) -> dict[str, Any]:
    """Build a new external reset/memory corpus from sealed episode bytes."""

    active = config or ImpactRecoveryCurriculumConfig()
    if len(sources) < 2 or len({item.source_id for item in sources}) != len(sources):
        raise ValueError("impact-recovery curriculum requires unique success and failure sources")
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if (
        destination.exists()
        or destination == checkout
        or checkout in destination.parents
        or destination in checkout.parents
    ):
        raise ValueError("impact-recovery curriculum output must be new and external")
    episodes = tuple(_load_historical_episode(item) for item in sources)
    successful = tuple(item for item in episodes if item.succeeded)
    failures = tuple(item for item in episodes if not item.succeeded)
    if not successful or not failures:
        raise ValueError("impact-recovery curriculum needs retention anchors and failures")

    root = asset_root.expanduser().resolve()
    model_path = root / "g1_description" / "g1_liao.xml"
    scene_path = root / "g1_description" / "scene_with_ball.xml"
    if not model_path.is_file() or not scene_path.is_file():
        raise FileNotFoundError("impact-recovery G1 assets are incomplete")
    body_hash = g1_body_hash(root)
    model_hash = hash_bytes(model_path.read_bytes())
    scene_hash = hash_bytes(scene_path.read_bytes())

    qpos_rows: list[NDArray[np.float32]] = []
    qvel_rows: list[NDArray[np.float32]] = []
    support_rows: list[NDArray[np.bool_]] = []
    ball_position_rows: list[NDArray[np.float32]] = []
    ball_velocity_rows: list[NDArray[np.float32]] = []
    executed_torque_rows: list[NDArray[np.float32]] = []
    initial_motor_target_rows: list[NDArray[np.float32]] = []
    elapsed_rows: list[float] = []
    yaw_error_rows: list[float] = []
    source_index_rows: list[int] = []
    successful_rows: list[bool] = []
    memory_route_rows: list[int] = []
    memory_targets: list[NDArray[np.float32]] = []
    initial_kp_rows: list[NDArray[np.float32]] = []
    initial_kd_rows: list[NDArray[np.float32]] = []
    memory_kp_rows: list[NDArray[np.float32]] = []
    memory_kd_rows: list[NDArray[np.float32]] = []
    memory_qpos_rows: list[NDArray[np.float32]] = []
    memory_qvel_rows: list[NDArray[np.float32]] = []
    rows: list[dict[str, Any]] = []

    for source_index, episode in enumerate(episodes):
        time = np.asarray(episode.trajectory["time"], dtype=np.float64)
        contact_time = float(time[episode.contact_index])
        for index in _sample_indexes(episode, active):
            pelvis = np.asarray(
                episode.trajectory["goalkeeper_pelvis_pose"][index], dtype=np.float32
            )
            joints = np.asarray(
                episode.trajectory["goalkeeper_joint_position"][index], dtype=np.float32
            )
            root_velocity = np.asarray(
                episode.trajectory["goalkeeper_root_velocity"][index], dtype=np.float32
            )
            joint_velocity = np.asarray(
                episode.trajectory["goalkeeper_joint_velocity"][index], dtype=np.float32
            )
            qpos = np.concatenate((pelvis, joints)).astype(np.float32)
            qvel = np.concatenate((root_velocity, joint_velocity)).astype(np.float32)
            if qpos.shape != (_ROOT_QPOS + _JOINT_COUNT,) or qvel.shape != (
                _ROOT_QVEL + _JOINT_COUNT,
            ):
                raise ValueError("impact-recovery extracted state has an invalid shape")
            (
                route_index,
                route_reference_frame,
                route_phase_offset_sec,
                memory,
                memory_kp,
                memory_kd,
                memory_qpos,
                memory_qvel,
            ) = _memory_route(
                episode,
                index,
                successful,
                config=active,
            )
            yaw_error = _yaw_error(
                np.asarray(pelvis[3:7], dtype=np.float64), active.desired_heading_rad
            )
            archive_row = len(qpos_rows)
            qpos_rows.append(qpos)
            qvel_rows.append(qvel)
            support_rows.append(
                np.asarray(episode.trajectory["goalkeeper_foot_contact"][index], dtype=np.bool_)
            )
            ball_position_rows.append(
                np.asarray(episode.trajectory["second_ball_pose"][index, :3], dtype=np.float32)
            )
            ball_velocity_rows.append(
                np.asarray(episode.trajectory["second_ball_velocity"][index, :3], dtype=np.float32)
            )
            executed_torque_rows.append(
                np.asarray(
                    episode.trajectory["goalkeeper_executed_torque"][index], dtype=np.float32
                )
            )
            initial_motor_target_rows.append(
                np.asarray(episode.trajectory["goalkeeper_policy_action"][index], dtype=np.float32)
            )
            elapsed_rows.append(float(time[index] - contact_time))
            yaw_error_rows.append(yaw_error)
            source_index_rows.append(source_index)
            successful_rows.append(episode.succeeded)
            memory_route_rows.append(route_index)
            memory_targets.append(memory)
            if active.dynamic_gain_memory_enabled:
                if (
                    memory_kp is None
                    or memory_kd is None
                    or "goalkeeper_kp" not in episode.trajectory
                    or "goalkeeper_kd" not in episode.trajectory
                ):
                    raise ValueError("impact-recovery dynamic reset gain is unavailable")
                initial_kp_rows.append(
                    np.asarray(episode.trajectory["goalkeeper_kp"][index], dtype=np.float32)
                )
                initial_kd_rows.append(
                    np.asarray(episode.trajectory["goalkeeper_kd"][index], dtype=np.float32)
                )
                memory_kp_rows.append(memory_kp)
                memory_kd_rows.append(memory_kd)
            if active.teacher_state_memory_enabled:
                if memory_qpos is None or memory_qvel is None:
                    raise ValueError("impact-recovery teacher state memory is unavailable")
                memory_qpos_rows.append(memory_qpos)
                memory_qvel_rows.append(memory_qvel)
            rows.append(
                {
                    "archive_row": archive_row,
                    "source_id": episode.source.source_id,
                    "source_use": episode.source.use,
                    "source_frame": index,
                    "elapsed_since_contact_sec": float(time[index] - contact_time),
                    "succeeded": episode.succeeded,
                    "memory_route_source_id": successful[route_index].source.source_id,
                    "memory_route_reference_frame": route_reference_frame,
                    "memory_route_phase_offset_sec": route_phase_offset_sec,
                }
            )

    arrays: dict[str, NDArray[Any]] = {
        "qpos": np.stack(qpos_rows).astype(np.float32),
        "qvel": np.stack(qvel_rows).astype(np.float32),
        "foot_contact": np.stack(support_rows).astype(np.bool_),
        "ball_position_m": np.stack(ball_position_rows).astype(np.float32),
        "ball_velocity_mps": np.stack(ball_velocity_rows).astype(np.float32),
        "executed_torque_nm": np.stack(executed_torque_rows).astype(np.float32),
        "initial_motor_target_rad": np.stack(initial_motor_target_rows).astype(np.float32),
        "elapsed_since_contact_sec": np.asarray(elapsed_rows, dtype=np.float32),
        "target_heading_error_rad": np.asarray(yaw_error_rows, dtype=np.float32),
        "source_index": np.asarray(source_index_rows, dtype=np.int32),
        "source_succeeded": np.asarray(successful_rows, dtype=np.bool_),
        "memory_route_index": np.asarray(memory_route_rows, dtype=np.int32),
        "frozen_memory_target_rad": np.stack(memory_targets).astype(np.float32),
    }
    if active.dynamic_gain_memory_enabled:
        arrays.update(
            initial_kp=np.stack(initial_kp_rows).astype(np.float32),
            initial_kd=np.stack(initial_kd_rows).astype(np.float32),
            frozen_memory_kp=np.stack(memory_kp_rows).astype(np.float32),
            frozen_memory_kd=np.stack(memory_kd_rows).astype(np.float32),
        )
    if active.teacher_state_memory_enabled:
        arrays.update(
            frozen_memory_qpos=np.stack(memory_qpos_rows).astype(np.float32),
            frozen_memory_qvel=np.stack(memory_qvel_rows).astype(np.float32),
        )
    if any(not np.all(np.isfinite(value)) for value in arrays.values() if value.dtype.kind == "f"):
        raise ValueError("impact-recovery curriculum contains non-finite values")

    destination.mkdir(parents=True)
    archive_path = destination / "impact-recovery-curriculum.npz"
    temporary = destination / ".impact-recovery-curriculum.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, archive_path)
    source_rows = [
        {
            "source_id": episode.source.source_id,
            "source_use": episode.source.use,
            "expected_success": episode.source.expected_success,
            "evidence_path": str(episode.source.evidence_path.expanduser().resolve()),
            "evidence_file_hash": episode.evidence_file_hash,
            "evidence_report_hash": episode.evidence_report_hash,
            "request_file_hash": episode.request_file_hash,
            "request_config_hash": episode.request_config_hash,
            "historical_reproducibility_closure_hash": episode.reproducibility_closure_hash,
            "trajectory_file_hash": episode.trajectory_file_hash,
            "trajectory_semantic_digest": episode.trajectory_semantic_digest,
        }
        for episode in episodes
    ]
    manifest: dict[str, Any] = {
        "schema_version": (
            "rosclaw_soccer.impact_recovery_curriculum.v6"
            if active.teacher_phase_search_radius_sec > 0.0
            else "rosclaw_soccer.impact_recovery_curriculum.v5"
            if active.teacher_state_memory_enabled
            else "rosclaw_soccer.impact_recovery_curriculum.v4"
            if active.dynamic_gain_memory_enabled
            else "rosclaw_soccer.impact_recovery_curriculum.v3"
        ),
        "config": asdict(active),
        "config_hash": active.config_hash,
        "archive": archive_path.name,
        "archive_hash": hash_bytes(archive_path.read_bytes()),
        "snapshot_count": len(rows),
        "retention_snapshot_count": sum(successful_rows),
        "acquisition_failure_snapshot_count": len(successful_rows) - sum(successful_rows),
        "body_hash": body_hash,
        "training_model_hash": model_hash,
        "full_chain_scene_hash": scene_hash,
        "sources": source_rows,
        "rows": rows,
        "teacher_memory_sources": [item.source.source_id for item in successful],
        "teacher_memory_semantics": "RECORDED_SUCCESSFUL_NEXT_PD_MOTOR_TARGET_SEQUENCE",
        "reset_motor_target_semantics": "RECORDED_SOURCE_PD_MOTOR_TARGET",
        "teacher_gain_semantics": (
            "RECORDED_SUCCESSFUL_NEXT_DYNAMIC_PD_GAINS"
            if active.dynamic_gain_memory_enabled
            else None
        ),
        "reset_gain_semantics": (
            "RECORDED_SOURCE_DYNAMIC_PD_GAINS" if active.dynamic_gain_memory_enabled else None
        ),
        "teacher_state_semantics": (
            "RECORDED_SUCCESSFUL_CAUSAL_PROPRIOCEPTIVE_SEQUENCE"
            if active.teacher_state_memory_enabled
            else None
        ),
        "teacher_retrieval_semantics": (
            "NEAREST_SUCCESSFUL_STATE_WITH_BOUNDED_PHASE_SEARCH"
            if active.teacher_phase_search_radius_sec > 0.0
            else "CONTACT_RELATIVE_TIME_ALIGNED"
        ),
        "failed_sources_used_as_teacher_count": 0,
        "historical_source_closures_recomputed": False,
        "pixels_used_for_training": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path = destination / "impact-recovery-curriculum.json"
    manifest_tmp = destination / ".impact-recovery-curriculum.json.tmp"
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_tmp, manifest_path)
    return validate_impact_recovery_curriculum(manifest_path)


def validate_impact_recovery_curriculum(path: Path) -> dict[str, Any]:
    """Fully verify an impact-recovery curriculum before training reads it."""

    resolved = path.expanduser().resolve()
    payload = _load_json(resolved)
    expected_hash = payload.pop("manifest_hash", None)
    try:
        config_value = payload.get("config")
        if not isinstance(config_value, dict):
            raise ValueError("impact-recovery curriculum config is missing")
        config = ImpactRecoveryCurriculumConfig(**config_value)
        archive_name = payload.get("archive")
        if archive_name != "impact-recovery-curriculum.npz":
            raise ValueError("impact-recovery curriculum archive name changed")
        archive_path = resolved.parent / archive_name
        schema_version = payload.get("schema_version")
        if (
            schema_version
            not in {
                "rosclaw_soccer.impact_recovery_curriculum.v1",
                "rosclaw_soccer.impact_recovery_curriculum.v2",
                "rosclaw_soccer.impact_recovery_curriculum.v3",
                "rosclaw_soccer.impact_recovery_curriculum.v4",
                "rosclaw_soccer.impact_recovery_curriculum.v5",
                "rosclaw_soccer.impact_recovery_curriculum.v6",
            }
            or expected_hash != hash_json(payload)
            or payload.get("config_hash") != hash_json(config_value)
            or payload.get("archive_hash") != hash_bytes(archive_path.read_bytes())
            or payload.get("failed_sources_used_as_teacher_count") != 0
            or payload.get("historical_source_closures_recomputed") is not False
            or payload.get("pixels_used_for_training") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_authorized") is not False
            or payload.get("hardware_command_sent") is not False
            or not _SHA256.fullmatch(str(payload.get("body_hash", "")))
            or not _SHA256.fullmatch(str(payload.get("training_model_hash", "")))
            or not _SHA256.fullmatch(str(payload.get("full_chain_scene_hash", "")))
        ):
            raise ValueError("impact-recovery curriculum authority or integrity changed")
        required = {
            "qpos",
            "qvel",
            "foot_contact",
            "ball_position_m",
            "ball_velocity_mps",
            "executed_torque_nm",
            "elapsed_since_contact_sec",
            "target_heading_error_rad",
            "source_index",
            "source_succeeded",
            "memory_route_index",
            "frozen_memory_target_rad",
        }
        if schema_version in {
            "rosclaw_soccer.impact_recovery_curriculum.v2",
            "rosclaw_soccer.impact_recovery_curriculum.v3",
            "rosclaw_soccer.impact_recovery_curriculum.v4",
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        }:
            required.add("initial_motor_target_rad")
        if schema_version in {
            "rosclaw_soccer.impact_recovery_curriculum.v4",
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        }:
            required.update({"initial_kp", "initial_kd", "frozen_memory_kp", "frozen_memory_kd"})
        if schema_version in {
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        }:
            required.update({"frozen_memory_qpos", "frozen_memory_qvel"})
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ValueError("impact-recovery curriculum arrays are incomplete")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        count = int(payload.get("snapshot_count", -1))
        shapes = {
            "qpos": (count, _ROOT_QPOS + _JOINT_COUNT),
            "qvel": (count, _ROOT_QVEL + _JOINT_COUNT),
            "foot_contact": (count, 2),
            "ball_position_m": (count, 3),
            "ball_velocity_mps": (count, 3),
            "executed_torque_nm": (count, _JOINT_COUNT),
            "elapsed_since_contact_sec": (count,),
            "target_heading_error_rad": (count,),
            "source_index": (count,),
            "source_succeeded": (count,),
            "memory_route_index": (count,),
            "frozen_memory_target_rad": (count, config.memory_horizon_steps, _JOINT_COUNT),
        }
        if schema_version in {
            "rosclaw_soccer.impact_recovery_curriculum.v2",
            "rosclaw_soccer.impact_recovery_curriculum.v3",
            "rosclaw_soccer.impact_recovery_curriculum.v4",
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        }:
            shapes["initial_motor_target_rad"] = (count, _JOINT_COUNT)
        if schema_version in {
            "rosclaw_soccer.impact_recovery_curriculum.v4",
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        }:
            shapes.update(
                initial_kp=(count, _JOINT_COUNT),
                initial_kd=(count, _JOINT_COUNT),
                frozen_memory_kp=(count, config.memory_horizon_steps, _JOINT_COUNT),
                frozen_memory_kd=(count, config.memory_horizon_steps, _JOINT_COUNT),
            )
        if schema_version in {
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        }:
            shapes.update(
                frozen_memory_qpos=(
                    count,
                    config.memory_horizon_steps,
                    _ROOT_QPOS + _JOINT_COUNT,
                ),
                frozen_memory_qvel=(
                    count,
                    config.memory_horizon_steps,
                    _ROOT_QVEL + _JOINT_COUNT,
                ),
            )
        if (
            count <= 0
            or any(arrays[name].shape != shape for name, shape in shapes.items())
            or any(
                not np.all(np.isfinite(value))
                for value in arrays.values()
                if value.dtype.kind == "f"
            )
            or int(np.count_nonzero(arrays["source_succeeded"]))
            != payload.get("retention_snapshot_count")
            or int(np.count_nonzero(~arrays["source_succeeded"].astype(np.bool_)))
            != payload.get("acquisition_failure_snapshot_count")
        ):
            raise ValueError("impact-recovery curriculum tensor contract changed")
        expected_teacher_semantics = (
            "RECORDED_SUCCESSFUL_NEXT_PD_MOTOR_TARGET_SEQUENCE"
            if schema_version
            in {
                "rosclaw_soccer.impact_recovery_curriculum.v3",
                "rosclaw_soccer.impact_recovery_curriculum.v4",
                "rosclaw_soccer.impact_recovery_curriculum.v5",
                "rosclaw_soccer.impact_recovery_curriculum.v6",
            }
            else "RECORDED_SUCCESSFUL_PD_MOTOR_TARGET_SEQUENCE"
        )
        if schema_version in {
            "rosclaw_soccer.impact_recovery_curriculum.v2",
            "rosclaw_soccer.impact_recovery_curriculum.v3",
            "rosclaw_soccer.impact_recovery_curriculum.v4",
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        } and (
            payload.get("teacher_memory_semantics") != expected_teacher_semantics
            or payload.get("reset_motor_target_semantics") != "RECORDED_SOURCE_PD_MOTOR_TARGET"
        ):
            raise ValueError("impact-recovery motor-target semantics changed")
        if schema_version in {
            "rosclaw_soccer.impact_recovery_curriculum.v4",
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        } and (
            config.dynamic_gain_memory_enabled is not True
            or payload.get("teacher_gain_semantics") != "RECORDED_SUCCESSFUL_NEXT_DYNAMIC_PD_GAINS"
            or payload.get("reset_gain_semantics") != "RECORDED_SOURCE_DYNAMIC_PD_GAINS"
            or np.any(arrays["initial_kp"] <= 0.0)
            or np.any(arrays["initial_kd"] <= 0.0)
            or np.any(arrays["frozen_memory_kp"] <= 0.0)
            or np.any(arrays["frozen_memory_kd"] <= 0.0)
        ):
            raise ValueError("impact-recovery dynamic gain semantics changed")
        if schema_version in {
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        } and (
            config.teacher_state_memory_enabled is not True
            or payload.get("teacher_state_semantics")
            != "RECORDED_SUCCESSFUL_CAUSAL_PROPRIOCEPTIVE_SEQUENCE"
        ):
            raise ValueError("impact-recovery teacher state semantics changed")
        if schema_version == "rosclaw_soccer.impact_recovery_curriculum.v6" and (
            config.teacher_phase_search_radius_sec <= 0.0
            or payload.get("teacher_retrieval_semantics")
            != "NEAREST_SUCCESSFUL_STATE_WITH_BOUNDED_PHASE_SEARCH"
        ):
            raise ValueError("impact-recovery teacher retrieval semantics changed")
        sources = payload.get("sources")
        rows = payload.get("rows")
        teacher_sources = payload.get("teacher_memory_sources")
        if (
            not isinstance(sources, list)
            or not isinstance(rows, list)
            or len(rows) != count
            or not isinstance(teacher_sources, list)
            or not teacher_sources
            or arrays["source_index"].dtype.kind not in "iu"
            or arrays["memory_route_index"].dtype.kind not in "iu"
            or np.any(arrays["source_index"] < 0)
            or np.any(arrays["source_index"] >= len(sources))
            or np.any(arrays["memory_route_index"] < 0)
            or np.any(arrays["memory_route_index"] >= len(teacher_sources))
            or any(
                not isinstance(source, dict)
                or any(
                    not _SHA256.fullmatch(str(source.get(key, "")))
                    for key in (
                        "evidence_file_hash",
                        "evidence_report_hash",
                        "request_file_hash",
                        "request_config_hash",
                        "historical_reproducibility_closure_hash",
                        "trajectory_file_hash",
                        "trajectory_semantic_digest",
                    )
                )
                for source in sources
            )
        ):
            raise ValueError("impact-recovery curriculum provenance is incomplete")
        source_ids = [source.get("source_id") for source in sources]
        successful_source_ids = {
            source.get("source_id") for source in sources if source.get("expected_success") is True
        }
        if (
            len(source_ids) != len(set(source_ids))
            or set(teacher_sources) != successful_source_ids
            or any(
                not isinstance(row, dict)
                or row.get("archive_row") != index
                or row.get("source_id") != source_ids[int(arrays["source_index"][index])]
                or row.get("memory_route_source_id")
                != teacher_sources[int(arrays["memory_route_index"][index])]
                or row.get("succeeded") is not bool(arrays["source_succeeded"][index])
                or (
                    schema_version == "rosclaw_soccer.impact_recovery_curriculum.v6"
                    and (
                        not isinstance(row.get("memory_route_reference_frame"), int)
                        or int(row["memory_route_reference_frame"]) < 0
                        or not isinstance(row.get("memory_route_phase_offset_sec"), float)
                        or not math.isfinite(float(row["memory_route_phase_offset_sec"]))
                        or abs(float(row["memory_route_phase_offset_sec"]))
                        > config.teacher_phase_search_radius_sec + config.control_dt_sec
                    )
                )
                for index, row in enumerate(rows)
            )
        ):
            raise ValueError("impact-recovery curriculum row binding changed")
        return payload
    finally:
        if expected_hash is not None:
            payload["manifest_hash"] = expected_hash


__all__ = [
    "ImpactRecoveryCurriculumConfig",
    "ImpactRecoverySource",
    "build_impact_recovery_curriculum",
    "validate_impact_recovery_curriculum",
]
