"""Evidence-first rolling diagnostics for the Soccer ball contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

_PHYSICS_DT_SEC = 0.002
_TRACE_DT_SEC = 0.02


@dataclass(frozen=True)
class RollingAuthenticityThresholds:
    maximum_median_slip_ratio: float = 0.12
    maximum_p95_slip_ratio: float = 0.25
    minimum_rolling_fraction: float = 0.80
    minimum_evaluated_samples: int = 20
    minimum_speed_mps: float = 0.20
    ground_height_tolerance_m: float = 0.012
    schema_version: str = "rosclaw_soccer.rolling_authenticity_thresholds.v1"

    def __post_init__(self) -> None:
        values = (
            self.maximum_median_slip_ratio,
            self.maximum_p95_slip_ratio,
            self.minimum_rolling_fraction,
            self.minimum_speed_mps,
            self.ground_height_tolerance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rolling authenticity thresholds must be finite")
        if not 0.02 <= self.maximum_median_slip_ratio <= 0.40:
            raise ValueError("rolling median slip threshold is invalid")
        if not self.maximum_median_slip_ratio <= self.maximum_p95_slip_ratio <= 0.60:
            raise ValueError("rolling p95 slip threshold is invalid")
        if not 0.50 <= self.minimum_rolling_fraction <= 1.0:
            raise ValueError("rolling fraction threshold is invalid")
        if not 10 <= self.minimum_evaluated_samples <= 10000:
            raise ValueError("rolling sample threshold is invalid")
        if not 0.05 <= self.minimum_speed_mps <= 2.0:
            raise ValueError("rolling minimum speed is invalid")
        if not 0.002 <= self.ground_height_tolerance_m <= 0.05:
            raise ValueError("rolling ground tolerance is invalid")


@dataclass(frozen=True)
class RollingAuthenticityMetrics:
    passed: bool
    evaluated_samples: int
    evaluated_duration_sec: float
    distance_m: float
    median_linear_speed_mps: float
    median_surface_speed_mps: float
    median_slip_speed_mps: float
    median_slip_ratio: float
    p95_slip_ratio: float
    rolling_fraction: float
    rotation_observed: bool
    schema_version: str = "rosclaw_soccer.rolling_authenticity_metrics.v1"


@dataclass(frozen=True)
class RollingAuditResult:
    output_path: str
    request_path: str
    trajectory_path: str
    request_hash: str
    trajectory_hash: str
    trajectory_digest: str
    source_evidence_hash: str
    source_trajectory_hash: str
    implementation_hash: str
    legacy: RollingAuthenticityMetrics
    corrected: RollingAuthenticityMetrics
    strict_replay: bool
    passed: bool
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    pixels_used_for_scoring: bool = False
    schema_version: str = "rosclaw_soccer.rolling_audit.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_rolling_authenticity(
    *,
    time: NDArray[np.floating[Any]],
    ball_pose: NDArray[np.floating[Any]],
    ball_velocity: NDArray[np.floating[Any]],
    ball_radius_m: float,
    thresholds: RollingAuthenticityThresholds | None = None,
    ignore_initial_sec: float = 0.20,
) -> tuple[RollingAuthenticityMetrics, NDArray[np.float64]]:
    """Measure no-slip consistency from physical linear and angular velocity."""

    gate = thresholds or RollingAuthenticityThresholds()
    timestamps = np.asarray(time, dtype=np.float64)
    pose = np.asarray(ball_pose, dtype=np.float64)
    velocity = np.asarray(ball_velocity, dtype=np.float64)
    if (
        timestamps.ndim != 1
        or pose.shape != (len(timestamps), 7)
        or velocity.shape != (len(timestamps), 6)
        or len(timestamps) < 2
    ):
        raise ValueError("rolling trace shapes are invalid")
    if not all(np.all(np.isfinite(value)) for value in (timestamps, pose, velocity)):
        raise ValueError("rolling trace contains non-finite values")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("rolling trace time must be strictly increasing")
    if not math.isfinite(ball_radius_m) or not 0.105 <= ball_radius_m <= 0.115:
        raise ValueError("rolling audit ball radius is outside the football range")
    linear_xy = velocity[:, :2]
    angular = velocity[:, 3:6]
    surface_xy = np.column_stack((ball_radius_m * angular[:, 1], -ball_radius_m * angular[:, 0]))
    linear_speed = np.linalg.norm(linear_xy, axis=1)
    surface_speed = np.linalg.norm(surface_xy, axis=1)
    slip_speed = np.linalg.norm(linear_xy - surface_xy, axis=1)
    slip_ratio = slip_speed / np.maximum(linear_speed, gate.minimum_speed_mps)
    active = (
        (np.abs(pose[:, 2] - ball_radius_m) <= gate.ground_height_tolerance_m)
        & (linear_speed >= gate.minimum_speed_mps)
        & (timestamps >= timestamps[0] + ignore_initial_sec)
    )
    selected = np.flatnonzero(active)
    if len(selected) < gate.minimum_evaluated_samples:
        raise ValueError("rolling trace has insufficient moving ground-contact samples")
    ratio = slip_ratio[selected]
    median_ratio = float(np.median(ratio))
    p95_ratio = float(np.quantile(ratio, 0.95))
    rolling_fraction = float(np.mean(ratio <= gate.maximum_median_slip_ratio))
    rotation_observed = bool(np.max(surface_speed[selected]) >= gate.minimum_speed_mps)
    first, last = int(selected[0]), int(selected[-1])
    distance = float(np.sum(np.linalg.norm(np.diff(pose[first : last + 1, :2], axis=0), axis=1)))
    metrics = RollingAuthenticityMetrics(
        passed=bool(
            median_ratio <= gate.maximum_median_slip_ratio
            and p95_ratio <= gate.maximum_p95_slip_ratio
            and rolling_fraction >= gate.minimum_rolling_fraction
            and rotation_observed
        ),
        evaluated_samples=len(selected),
        evaluated_duration_sec=float(timestamps[last] - timestamps[first]),
        distance_m=distance,
        median_linear_speed_mps=float(np.median(linear_speed[selected])),
        median_surface_speed_mps=float(np.median(surface_speed[selected])),
        median_slip_speed_mps=float(np.median(slip_speed[selected])),
        median_slip_ratio=median_ratio,
        p95_slip_ratio=p95_ratio,
        rolling_fraction=rolling_fraction,
        rotation_observed=rotation_observed,
    )
    return metrics, np.asarray(slip_ratio, dtype=np.float64)


def audit_pass_rolling_physics(
    *,
    asset_root: Path,
    source_evidence_path: Path,
    output_dir: Path,
    source_checkout: Path,
    duration_sec: float = 2.20,
) -> RollingAuditResult:
    """Replay the measured pass launch under legacy and corrected damping."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    evidence_path = source_evidence_path.expanduser().resolve()
    if root == checkout or checkout in root.parents or root.exists():
        raise ValueError("rolling audit output must be a new path outside the source checkout")
    if evidence_path == checkout or checkout in evidence_path.parents:
        raise ValueError("rolling audit source evidence must remain outside the source checkout")
    if not 1.0 <= duration_sec <= 5.0:
        raise ValueError("rolling audit duration must be in [1, 5] sec")
    source = _load_json(evidence_path)
    trajectory_source = evidence_path.parent / "trajectory.npz"
    if (
        source.get("activation_ceiling") != "SIM_ONLY"
        or source.get("physics_authority") != "CPU_MUJOCO"
    ):
        raise ValueError("rolling audit source lacks SIM_ONLY CPU MuJoCo authority")
    if source.get("hardware_command_sent") is not False:
        raise ValueError("rolling audit source contains a hardware command claim")
    if source.get("trajectory_hash") != _file_hash(trajectory_source):
        raise ValueError("rolling audit source trajectory hash mismatch")
    result = source.get("result")
    if not isinstance(result, dict) or result.get("pass_contact_observed") is not True:
        raise ValueError("rolling audit source has no measured pass contact")
    contact = _number(result.get("pass_contact_time_sec"), "pass contact time")
    with np.load(trajectory_source, allow_pickle=False) as archive:
        source_time = np.asarray(archive["time"], dtype=np.float64)
        source_pose = np.asarray(archive["ball_pose"], dtype=np.float64)
        source_velocity = np.asarray(archive["ball_velocity"], dtype=np.float64)
    index = int(np.searchsorted(source_time, contact + 0.02, side="left"))
    initial_pose = source_pose[index].copy()
    initial_pose[1] += 3.0
    initial_velocity = source_velocity[index].copy()
    request = {
        "schema_version": "rosclaw_soccer.rolling_audit_request.v1",
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "hardware_command_sent": False,
        "source_evidence_hash": _file_hash(evidence_path),
        "source_trajectory_hash": _file_hash(trajectory_source),
        "source_frame_index": index,
        "initial_ball_pose_wxyz": initial_pose.tolist(),
        "initial_ball_velocity": initial_velocity.tolist(),
        "duration_sec": duration_sec,
        "legacy_angular_damping_n_m_s_rad": 0.02,
        "corrected_angular_damping_n_m_s_rad": 0.00002,
        "ball_ground_sliding_friction": 0.10,
        "pixels_used_for_scoring": False,
    }
    root.mkdir(parents=True)
    request_path = root / "request.json"
    _write_json(request_path, request)
    goal = G1TrainingGoalSpec(
        plane_x_m=7.5,
        width_m=3.0,
        height_m=2.0,
        depth_m=1.2,
        target_y_m=0.89,
        target_z_m=0.115,
        precision_radius_m=0.10,
        ball_contact_sliding_friction=0.10,
        ball_sliding_friction=0.10,
    )
    traces: dict[str, dict[str, NDArray[np.float64]]] = {}
    metrics: dict[str, RollingAuthenticityMetrics] = {}
    strict = True
    for name, damping in (("legacy", 0.02), ("corrected", 0.00002)):
        trace = _simulate_roll(
            asset_root, goal, initial_pose, initial_velocity, damping, duration_sec
        )
        replay = _simulate_roll(
            asset_root, goal, initial_pose, initial_velocity, damping, duration_sec
        )
        strict = strict and trajectory_digest(trace) == trajectory_digest(replay)
        metrics[name], trace["slip_ratio"] = measure_rolling_authenticity(
            time=trace["time"],
            ball_pose=trace["ball_pose"],
            ball_velocity=trace["ball_velocity"],
            ball_radius_m=goal.ball_radius_m,
        )
        traces[name] = trace
    combined = {
        f"{name}_{key}": value for name, trace in traces.items() for key, value in trace.items()
    }
    trajectory_path = root / "rolling-audit-trajectory.npz"
    np.savez_compressed(trajectory_path, **combined)  # type: ignore[arg-type]
    passed = bool(strict and not metrics["legacy"].passed and metrics["corrected"].passed)
    unsigned = {
        "schema_version": "rosclaw_soccer.rolling_audit.v1",
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
        "source_evidence_hash": request["source_evidence_hash"],
        "source_trajectory_hash": request["source_trajectory_hash"],
        "request_hash": _file_hash(request_path),
        "trajectory_hash": _file_hash(trajectory_path),
        "trajectory_digest": trajectory_digest(combined),
        "implementation_hash": _implementation_hash(),
        "legacy": asdict(metrics["legacy"]),
        "corrected": asdict(metrics["corrected"]),
        "strict_replay": strict,
        "passed": passed,
    }
    output_path = root / "rolling-audit.json"
    _write_json(output_path, unsigned)
    return RollingAuditResult(
        output_path=str(output_path),
        request_path=str(request_path),
        trajectory_path=str(trajectory_path),
        request_hash=str(unsigned["request_hash"]),
        trajectory_hash=str(unsigned["trajectory_hash"]),
        trajectory_digest=str(unsigned["trajectory_digest"]),
        source_evidence_hash=str(unsigned["source_evidence_hash"]),
        source_trajectory_hash=str(unsigned["source_trajectory_hash"]),
        implementation_hash=str(unsigned["implementation_hash"]),
        legacy=metrics["legacy"],
        corrected=metrics["corrected"],
        strict_replay=strict,
        passed=passed,
    )


def _simulate_roll(
    asset_root: Path,
    goal: G1TrainingGoalSpec,
    initial_pose: NDArray[np.float64],
    initial_velocity: NDArray[np.float64],
    angular_damping: float,
    duration_sec: float,
) -> dict[str, NDArray[np.float64]]:
    import mujoco

    model = build_g1_stadium_model(asset_root, goal)
    model.opt.timestep = _PHYSICS_DT_SEC
    data = mujoco.MjData(model)
    joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"))
    qpos, qvel = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    model.dof_damping[qvel + 3 : qvel + 6] = angular_damping
    data.qpos[:] = model.qpos0
    data.qpos[qpos : qpos + 7] = initial_pose
    data.qvel[qvel : qvel + 6] = initial_velocity
    mujoco.mj_forward(model, data)
    frames = int(round(duration_sec / _TRACE_DT_SEC)) + 1
    time: NDArray[np.float64] = np.empty(frames, dtype=np.float64)
    pose: NDArray[np.float64] = np.empty((frames, 7), dtype=np.float64)
    velocity: NDArray[np.float64] = np.empty((frames, 6), dtype=np.float64)
    for frame in range(frames):
        time[frame] = data.time
        pose[frame] = data.qpos[qpos : qpos + 7]
        velocity[frame] = data.qvel[qvel : qvel + 6]
        for _ in range(int(round(_TRACE_DT_SEC / _PHYSICS_DT_SEC))):
            mujoco.mj_step(model, data)
    return {"time": time, "ball_pose": pose, "ball_velocity": velocity}


def _implementation_hash() -> str:
    return str(
        hash_json(
            {
                "rolling": hash_bytes(Path(__file__).read_bytes()),
                "field": hash_bytes(
                    Path(__file__).resolve().parents[1].joinpath("world/field.py").read_bytes()
                ),
            }
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or not 1 <= path.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError("rolling audit source evidence is missing or oversized")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("rolling audit source evidence must be a JSON object")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"rolling audit {label} is invalid")
    return float(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "RollingAuditResult",
    "RollingAuthenticityMetrics",
    "RollingAuthenticityThresholds",
    "audit_pass_rolling_physics",
    "measure_rolling_authenticity",
]
