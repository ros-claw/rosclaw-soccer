"""Fail-closed validation of one passer/shooter/goalkeeper evidence bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.physics.rolling_authenticity import measure_rolling_authenticity
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest

Array: TypeAlias = NDArray[np.generic]
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024
_ROLES = ("passer", "shooter", "goalkeeper")


@dataclass(frozen=True)
class ThreePlayerEvidenceBundle:
    evidence_path: Path
    request_path: Path
    trajectory_path: Path
    report: dict[str, Any]
    request: dict[str, Any]
    trajectory: dict[str, Array]
    evidence_hash: str
    request_hash: str
    trajectory_hash: str
    trajectory_digest: str


def validate_three_player_evidence(
    evidence_path: Path,
    *,
    source_checkout: Path,
) -> ThreePlayerEvidenceBundle:
    """Validate immutable shared-world evidence before any rendering starts."""

    checkout = source_checkout.expanduser().resolve()
    if not checkout.is_dir() or not (checkout / "pyproject.toml").is_file():
        raise ValueError("three-player source checkout is missing or invalid")
    evidence = evidence_path.expanduser().resolve()
    _require_outside_checkout(evidence, checkout, "evidence")
    request_path = evidence.parent / "request.json"
    trajectory_path = evidence.parent / "trajectory.npz"
    for path, label in ((request_path, "request"), (trajectory_path, "trajectory")):
        _require_outside_checkout(path, checkout, label)

    report = _load_json(evidence, "evidence")
    request = _load_json(request_path, "request")
    trajectory = load_three_player_trajectory(trajectory_path)
    request_hash = _file_hash(request_path)
    trajectory_hash = _file_hash(trajectory_path)
    digest = trajectory_digest(trajectory)

    if report.get("request_hash") != request_hash:
        raise ValueError("three-player request hash mismatch")
    if report.get("trajectory_hash") != trajectory_hash:
        raise ValueError("three-player trajectory file hash mismatch")
    if report.get("trajectory_digest") != digest:
        raise ValueError("three-player trajectory content digest mismatch")
    _validate_authority(report)
    _validate_metrics(report)
    _validate_request(request, report)
    _validate_pass_roll(report, request, trajectory)

    return ThreePlayerEvidenceBundle(
        evidence_path=evidence,
        request_path=request_path,
        trajectory_path=trajectory_path,
        report=report,
        request=request,
        trajectory=trajectory,
        evidence_hash=_file_hash(evidence),
        request_hash=request_hash,
        trajectory_hash=trajectory_hash,
        trajectory_digest=digest,
    )


def load_three_player_trajectory(path: Path) -> dict[str, Array]:
    """Load a bounded no-pickle three-G1 trajectory on one strict timeline."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not 1 <= resolved.stat().st_size <= _MAX_TRAJECTORY_BYTES:
        raise ValueError("three-player trajectory is missing, empty, or too large")
    with np.load(resolved, allow_pickle=False) as archive:
        value = {name: archive[name] for name in archive.files}
    expected: dict[str, tuple[int, ...]] = {
        "time": (),
        "ball_pose": (7,),
        "ball_velocity": (6,),
    }
    for role in _ROLES:
        expected[f"{role}_pelvis_pose"] = (7,)
        expected[f"{role}_joint_position"] = (29,)
    for name, shape in expected.items():
        array = np.asarray(value.get(name))
        if array.ndim != len(shape) + 1 or array.shape[1:] != shape:
            raise ValueError(f"three-player trajectory {name} has invalid shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"three-player trajectory {name} is non-finite")
    if len({len(np.asarray(value[name])) for name in expected}) != 1:
        raise ValueError("three-player trajectory arrays do not share one timeline")
    time = np.asarray(value["time"], dtype=np.float64)
    if len(time) < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("three-player trajectory time must be strictly increasing")
    for name in ("ball_pose", *(f"{role}_pelvis_pose" for role in _ROLES)):
        norms = np.linalg.norm(np.asarray(value[name])[:, 3:], axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError(f"three-player trajectory {name} has a zero quaternion")
    return value


def _validate_authority(report: dict[str, Any]) -> None:
    required_true = (
        "passed",
        "strict_replay",
        "simultaneous_three_body_physics",
        "shared_ball_state",
        "unified_physics_and_render_scene",
    )
    if any(report.get(name) is not True for name in required_true):
        raise ValueError("three-player evidence is not a passing strict shared-world replay")
    if report.get("activation_ceiling") != "SIM_ONLY":
        raise ValueError("three-player evidence is not SIM_ONLY")
    if report.get("physics_authority") != "CPU_MUJOCO":
        raise ValueError("three-player evidence does not declare CPU MuJoCo authority")
    if report.get("hardware_command_sent") is not False:
        raise ValueError("three-player evidence contains a hardware command claim")
    claims = report.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("three-player evidence claims are missing")
    if claims.get("real_hardware") is not False:
        raise ValueError("three-player evidence contains a real-hardware claim")
    if claims.get("pixels_used_for_promotion") is not False:
        raise ValueError("three-player evidence must not use pixels for scoring")


def _validate_metrics(report: dict[str, Any]) -> None:
    result = report.get("result")
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise ValueError("three-player result is not passing")
    required_true = (
        "finite_state",
        "pass_contact_observed",
        "shot_contact_observed",
        "goal_crossed",
        "goalkeeper_enabled",
        "pass_precision_passed",
    )
    if any(result.get(name) is not True for name in required_true):
        raise ValueError("three-player result is missing a required physical event")
    required_false = (
        "joint_limit_violation",
        "torque_limit_violation",
        "actuator_saturation",
        "passer_post_kick_fall",
        "shooter_post_kick_fall",
        "goalkeeper_joint_limit_violation",
    )
    if any(result.get(name) is not False for name in required_false):
        raise ValueError("three-player result violates a safety or stability gate")
    bounds = (
        (report.get("pass_distance_m"), 2.75, "pass distance"),
        (report.get("shot_distance_m"), 6.25, "shot distance"),
        (result.get("pass_delivery_error_m"), 0.10, "pass delivery error", True),
        (result.get("target_error_m"), 0.10, "shot target error", True),
        (result.get("goalkeeper_lateral_displacement_m"), 0.75, "goalkeeper displacement"),
        (result.get("goalkeeper_min_pelvis_height_m"), 0.65, "goalkeeper height"),
    )
    for item in bounds:
        value, limit, label, *upper = item
        number = _finite_number(value, str(label))
        if not np.isfinite(number) or (number > limit if upper else number < limit):
            raise ValueError(f"three-player {label} fails its gate")
    if _finite_number(report.get("pass_speed_max_positive_step_mps"), "pass speed step") > 0.03:
        raise ValueError("three-player pass speed contains a non-physical positive jump")
    pass_time = _finite_number(result.get("pass_contact_time_sec"), "pass contact time")
    shot_time = _finite_number(result.get("shot_contact_time_sec"), "shot contact time")
    if not (np.isfinite(pass_time) and np.isfinite(shot_time) and 0.0 < pass_time < shot_time):
        raise ValueError("three-player contact order is invalid")


def _validate_request(request: dict[str, Any], report: dict[str, Any]) -> None:
    if request.get("activation_ceiling") != "SIM_ONLY":
        raise ValueError("three-player request is not SIM_ONLY")
    if request.get("physics_authority") != "CPU_MUJOCO":
        raise ValueError("three-player request does not declare CPU MuJoCo authority")
    if request.get("body_hash") != report.get("body_hash"):
        raise ValueError("three-player request Body hash does not match evidence")
    goal = request.get("goal_spec")
    target = request.get("physical_scoring_target_m")
    if not isinstance(goal, dict) or not isinstance(target, list | tuple) or len(target) != 3:
        raise ValueError("three-player request goal or physical target is missing")
    expected = (
        _finite_number(goal.get("plane_x_m"), "goal plane"),
        _finite_number(goal.get("target_y_m"), "goal target y"),
        _finite_number(goal.get("target_z_m"), "goal target z"),
    )
    actual = tuple(_finite_number(value, "physical scoring target") for value in target)
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-9):
        raise ValueError("three-player physical target does not match the goal contract")


def _validate_pass_roll(
    report: dict[str, Any],
    request: dict[str, Any],
    trajectory: dict[str, Array],
) -> None:
    result = report["result"]
    pass_time = _finite_number(result.get("pass_contact_time_sec"), "pass contact time")
    shot_time = _finite_number(result.get("shot_contact_time_sec"), "shot contact time")
    goal = request["goal_spec"]
    radius = _finite_number(goal.get("ball_radius_m", 0.115), "ball radius")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    start = int(np.searchsorted(time, pass_time + 0.10, side="left"))
    end = int(np.searchsorted(time, shot_time - 0.15, side="right"))
    if end - start < 2:
        raise ValueError("three-player pass roll interval is too short")
    metrics, _ = measure_rolling_authenticity(
        time=time[start:end],
        ball_pose=np.asarray(trajectory["ball_pose"])[start:end],
        ball_velocity=np.asarray(trajectory["ball_velocity"])[start:end],
        ball_radius_m=radius,
        ignore_initial_sec=0.0,
    )
    if not metrics.passed:
        raise ValueError(
            "three-player pass is sliding rather than rolling "
            f"(median_slip_ratio={metrics.median_slip_ratio:.3f})"
        )


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"three-player {label} is missing or invalid")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"three-player {label} is non-finite")
    return converted


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or not 1 <= path.stat().st_size <= _MAX_JSON_BYTES:
        raise ValueError(f"three-player {label} is missing, empty, or too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"three-player {label} must be a JSON object")
    return value


def _require_outside_checkout(path: Path, checkout: Path, label: str) -> None:
    if path == checkout or checkout in path.parents:
        raise ValueError(f"three-player raw {label} must be outside the source checkout")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "ThreePlayerEvidenceBundle",
    "load_three_player_trajectory",
    "validate_three_player_evidence",
]
