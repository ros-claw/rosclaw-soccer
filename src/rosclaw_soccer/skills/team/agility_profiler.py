"""Layer-by-layer temporal agility diagnostics for humanoid policies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class AgilityProfilerConfig:
    response_target_delta_rad: float = 0.02
    idle_target_velocity_rms_rad_s: float = 0.05
    idle_actual_velocity_rms_rad_s: float = 0.05
    torque_projection_tolerance_nm: float = 1e-6
    actuator_tracking_tolerance_nm: float = 0.5
    velocity_limits_rad_s: tuple[float, ...] | None = None
    schema_version: str = "rosclaw_soccer.agility_profiler_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.response_target_delta_rad,
            self.idle_target_velocity_rms_rad_s,
            self.idle_actual_velocity_rms_rad_s,
            self.torque_projection_tolerance_nm,
            self.actuator_tracking_tolerance_nm,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("agility profiler thresholds must be finite and positive")
        if self.velocity_limits_rad_s is not None and (
            not self.velocity_limits_rad_s
            or any(not math.isfinite(value) or value <= 0.0 for value in self.velocity_limits_rad_s)
        ):
            raise ValueError("agility velocity limits must be finite and positive")


@dataclass(frozen=True)
class AgilityRuntimeTelemetry:
    """Nondeterministic wall-clock telemetry kept outside trajectory hashes."""

    inference_latency_sec: tuple[float, ...] = ()
    handoff_requested_sec: float | None = None
    schema_version: str = "rosclaw_soccer.agility_runtime_telemetry.v1"

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0.0 for value in self.inference_latency_sec):
            raise ValueError("inference latencies must be finite and non-negative")
        if self.handoff_requested_sec is not None and (
            not math.isfinite(self.handoff_requested_sec) or self.handoff_requested_sec < 0.0
        ):
            raise ValueError("handoff request time must be finite and non-negative")


@dataclass(frozen=True)
class TemporalAgilityProfile:
    role: str
    frame_count: int
    duration_sec: float
    reaction_latency_sec: float | None
    skill_handoff_latency_sec: float | None
    motion_pause_time_sec: float
    idle_ratio: float
    target_velocity_clip_fraction: float | None
    torque_projection_fraction: float
    actuator_tracking_miss_fraction: float
    target_joint_speed_rms_rad_s: float
    actual_joint_speed_rms_rad_s: float
    actual_joint_acceleration_rms_rad_s2: float
    peak_actual_joint_acceleration_rad_s2: float
    mean_abs_pd_tracking_error_nm: float
    inference_latency_p50_ms: float | None
    inference_latency_p90_ms: float | None
    bottlenecks: tuple[str, ...]
    complete: bool
    missing_channels: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.temporal_agility_profile.v1"

    def __post_init__(self) -> None:
        if not self.role.strip() or self.frame_count < 2 or self.duration_sec <= 0.0:
            raise ValueError("agility profile requires a valid role and time window")
        values = (
            self.motion_pause_time_sec,
            self.idle_ratio,
            self.torque_projection_fraction,
            self.actuator_tracking_miss_fraction,
            self.target_joint_speed_rms_rad_s,
            self.actual_joint_speed_rms_rad_s,
            self.actual_joint_acceleration_rms_rad_s2,
            self.peak_actual_joint_acceleration_rad_s2,
            self.mean_abs_pd_tracking_error_nm,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("agility profile metrics must be finite and non-negative")
        for fraction in (
            self.idle_ratio,
            self.torque_projection_fraction,
            self.actuator_tracking_miss_fraction,
        ):
            if fraction > 1.0:
                raise ValueError("agility fractions must be in [0, 1]")
        if self.target_velocity_clip_fraction is not None and not (
            0.0 <= self.target_velocity_clip_fraction <= 1.0
        ):
            raise ValueError("target velocity clipping fraction must be in [0, 1]")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_command_sent:
            raise ValueError("agility profiles are SIM_ONLY evidence")

    @property
    def profile_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_temporal_agility(
    trajectory: dict[str, np.ndarray],
    *,
    role: str,
    observation_event_sec: float | None = None,
    window_start_sec: float | None = None,
    window_end_sec: float | None = None,
    config: AgilityProfilerConfig | None = None,
    telemetry: AgilityRuntimeTelemetry | None = None,
) -> TemporalAgilityProfile:
    """Diagnose desired motion, PD, safety projection, execution, and response."""

    active = config or AgilityProfilerConfig()
    runtime = telemetry or AgilityRuntimeTelemetry()
    required = {
        "time": "time",
        "position": f"{role}_joint_position",
        "velocity": f"{role}_joint_velocity",
        "action": f"{role}_policy_action",
        "target_velocity": f"{role}_target_velocity",
        "commanded": f"{role}_commanded_torque",
        "projected": f"{role}_safety_projected_torque",
        "executed": f"{role}_executed_torque",
    }
    missing = tuple(name for name, key in required.items() if key not in trajectory)
    if any(name in missing for name in ("time", "position", "velocity", "action")):
        raise ValueError(f"agility trajectory is missing required channels: {missing}")

    time = _matrix(trajectory[required["time"]], columns=None, label="time").reshape(-1)
    position = _matrix(trajectory[required["position"]], columns=None, label="position")
    velocity = _matrix(
        trajectory[required["velocity"]], columns=position.shape[1], label="velocity"
    )
    action = _matrix(trajectory[required["action"]], columns=position.shape[1], label="action")
    if len(time) != len(position) or len(time) < 2:
        raise ValueError("agility trajectory channels must share at least two frames")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("agility trajectory time must be strictly increasing")

    start = float(time[0]) if window_start_sec is None else window_start_sec
    end = float(time[-1]) if window_end_sec is None else window_end_sec
    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("agility profiling window must be finite and increasing")
    mask = (time >= start) & (time <= end)
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        raise ValueError("agility profiling window contains fewer than two frames")
    time_w = time[indices]
    velocity_w = velocity[indices]
    action_w = action[indices]
    dt = np.diff(time_w)
    policy_target_speed = np.diff(action_w, axis=0) / dt[:, None]
    acceleration = np.diff(velocity_w, axis=0) / dt[:, None]

    target_velocity = _optional_channel(
        trajectory,
        required["target_velocity"],
        len(time),
        position.shape[1],
    )
    if target_velocity is not None:
        explicit_target_velocity = target_velocity[indices][1:]
    else:
        explicit_target_velocity = policy_target_speed

    commanded = _optional_channel(trajectory, required["commanded"], len(time), position.shape[1])
    projected = _optional_channel(trajectory, required["projected"], len(time), position.shape[1])
    executed = _optional_channel(trajectory, required["executed"], len(time), position.shape[1])
    torque_projection_fraction = _difference_fraction(
        commanded,
        projected,
        indices,
        active.torque_projection_tolerance_nm,
    )
    actuator_tracking_fraction = _difference_fraction(
        projected,
        executed,
        indices,
        active.actuator_tracking_tolerance_nm,
    )
    pd_tracking_error = (
        0.0
        if commanded is None or executed is None
        else float(np.mean(np.abs(commanded[indices] - executed[indices])))
    )
    velocity_clip_fraction: float | None = None
    if active.velocity_limits_rad_s is not None:
        limits = np.asarray(active.velocity_limits_rad_s, dtype=np.float64)
        if limits.shape != (position.shape[1],):
            raise ValueError("velocity limits must match the trajectory joint dimension")
        velocity_clip_fraction = float(np.mean(np.abs(explicit_target_velocity) > limits[None, :]))

    target_rms_by_frame = np.sqrt(np.mean(np.square(policy_target_speed), axis=1))
    actual_rms_by_frame = np.sqrt(np.mean(np.square(velocity_w[1:]), axis=1))
    idle = (target_rms_by_frame < active.idle_target_velocity_rms_rad_s) & (
        actual_rms_by_frame < active.idle_actual_velocity_rms_rad_s
    )
    pause = _longest_duration(idle, dt)
    reaction = _response_latency(
        time=time,
        action=action,
        event_sec=observation_event_sec,
        threshold=active.response_target_delta_rad,
    )
    handoff = _response_latency(
        time=time,
        action=action,
        event_sec=runtime.handoff_requested_sec,
        threshold=active.response_target_delta_rad,
    )
    inference = np.asarray(runtime.inference_latency_sec, dtype=np.float64)
    bottlenecks: list[str] = []
    if float(np.mean(target_rms_by_frame)) < active.idle_target_velocity_rms_rad_s:
        bottlenecks.append("policy_desired_motion_limited")
    if torque_projection_fraction > 0.01:
        bottlenecks.append("safety_projection_limited")
    if actuator_tracking_fraction > 0.01:
        bottlenecks.append("actuator_tracking_limited")
    if pd_tracking_error > 5.0 and torque_projection_fraction <= 0.01:
        bottlenecks.append("pd_or_dynamics_tracking_limited")
    if float(np.mean(idle)) > 0.10:
        bottlenecks.append("motion_pause")

    complete = not missing and bool(runtime.inference_latency_sec)
    if not runtime.inference_latency_sec:
        missing = (*missing, "inference_latency")
    return TemporalAgilityProfile(
        role=role,
        frame_count=len(indices),
        duration_sec=float(time_w[-1] - time_w[0]),
        reaction_latency_sec=reaction,
        skill_handoff_latency_sec=handoff,
        motion_pause_time_sec=pause,
        idle_ratio=float(np.mean(idle)),
        target_velocity_clip_fraction=velocity_clip_fraction,
        torque_projection_fraction=torque_projection_fraction,
        actuator_tracking_miss_fraction=actuator_tracking_fraction,
        target_joint_speed_rms_rad_s=float(np.sqrt(np.mean(np.square(policy_target_speed)))),
        actual_joint_speed_rms_rad_s=float(np.sqrt(np.mean(np.square(velocity_w)))),
        actual_joint_acceleration_rms_rad_s2=float(np.sqrt(np.mean(np.square(acceleration)))),
        peak_actual_joint_acceleration_rad_s2=float(np.max(np.abs(acceleration))),
        mean_abs_pd_tracking_error_nm=pd_tracking_error,
        inference_latency_p50_ms=(
            None if not inference.size else float(np.quantile(inference, 0.50) * 1000.0)
        ),
        inference_latency_p90_ms=(
            None if not inference.size else float(np.quantile(inference, 0.90) * 1000.0)
        ),
        bottlenecks=tuple(bottlenecks),
        complete=complete,
        missing_channels=tuple(dict.fromkeys(missing)),
    )


def _matrix(value: np.ndarray, *, columns: int | None, label: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if label == "time":
        if result.ndim != 1:
            raise ValueError("agility time must be one-dimensional")
    elif result.ndim != 2 or (columns is not None and result.shape[1] != columns):
        raise ValueError(f"agility {label} has an invalid shape")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"agility {label} must be finite")
    return result


def _optional_channel(
    trajectory: dict[str, np.ndarray],
    key: str,
    frames: int,
    joints: int,
) -> NDArray[np.float64] | None:
    if key not in trajectory:
        return None
    value = _matrix(trajectory[key], columns=joints, label=key)
    if value.shape != (frames, joints):
        raise ValueError(f"agility {key} must match the trajectory frame count")
    return value


def _difference_fraction(
    left: NDArray[np.float64] | None,
    right: NDArray[np.float64] | None,
    indices: NDArray[np.intp],
    tolerance: float,
) -> float:
    if left is None or right is None:
        return 0.0
    return float(np.mean(np.abs(left[indices] - right[indices]) > tolerance))


def _response_latency(
    *,
    time: NDArray[np.float64],
    action: NDArray[np.float64],
    event_sec: float | None,
    threshold: float,
) -> float | None:
    if event_sec is None:
        return None
    if not math.isfinite(event_sec) or not float(time[0]) <= event_sec <= float(time[-1]):
        raise ValueError("response event must be inside the trajectory")
    event_index = int(np.searchsorted(time, event_sec, side="left"))
    reference_index = max(0, event_index - 1)
    delta = np.sqrt(np.mean(np.square(action[event_index:] - action[reference_index]), axis=1))
    responsive = np.flatnonzero(delta >= threshold)
    if not responsive.size:
        return None
    response_time = float(time[event_index + int(responsive[0])])
    return max(0.0, response_time - event_sec)


def _longest_duration(active: NDArray[np.bool_], dt: NDArray[np.float64]) -> float:
    longest = current = 0.0
    for is_active, duration in zip(active, dt, strict=True):
        current = current + float(duration) if is_active else 0.0
        longest = max(longest, current)
    return longest


__all__ = [
    "AgilityProfilerConfig",
    "AgilityRuntimeTelemetry",
    "TemporalAgilityProfile",
    "profile_temporal_agility",
]
