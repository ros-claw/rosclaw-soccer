"""Strict-replay evidence for the three-role goalkeeper development loop."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.physics.rolling_authenticity import (
    RollingAuthenticityMetrics,
    measure_rolling_authenticity,
)
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SharedWorldResult,
    simulate_shared_world,
    trained_three_role_skill_simulation_kwargs,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_PASSER_ORIGIN = (5.10, -0.16406006503921598, 0.0)
_PASSER_BALL_LOCAL_XY = (1.205, -0.16)
_PHYSICAL_TARGET = (7.50, 0.89, 0.115)
_POLICY_TARGET = (7.50, 0.70, 0.50)


@dataclass(frozen=True)
class ThreeRoleDevelopmentEvidence:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    implementation_hash: str
    request_hash: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    result: G1SharedWorldResult
    reactive_goalkeeper_lateral_displacement_m: float
    anticipation_displacement_improvement_fraction: float
    pass_distance_m: float
    shot_distance_m: float
    pass_speed_start_mps: float
    pass_speed_end_mps: float
    pass_speed_max_positive_step_mps: float
    pass_speed_positive_step_count: int
    rolling_authenticity_passed: bool
    rolling_median_slip_ratio: float
    role_policy_hashes: dict[str, str]
    passed: bool = False
    promotion_status: str = "REJECTED_DEVELOPMENT"
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    simultaneous_three_body_physics: bool = True
    shared_ball_state: bool = True
    unified_physics_and_render_scene: bool = True
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.three_role_development_evidence.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "result": self.result.to_dict(),
            "claims": {
                "strict_deterministic_replay": self.strict_replay,
                "single_shared_ball": True,
                "goalkeeper_uses_shooter_proprioception_before_contact": True,
                "goalkeeper_ball_contact_achieved": self.result.goalkeeper_ball_contact_observed,
                "candidate_promoted": False,
                "pixels_used_for_promotion": False,
                "real_hardware": False,
            },
        }


def three_role_goal_spec() -> G1TrainingGoalSpec:
    return G1TrainingGoalSpec(
        plane_x_m=_PHYSICAL_TARGET[0],
        width_m=3.0,
        height_m=2.0,
        depth_m=1.2,
        target_y_m=_PHYSICAL_TARGET[1],
        target_z_m=_PHYSICAL_TARGET[2],
        precision_radius_m=0.10,
    )


def three_role_development_kwargs() -> dict[str, Any]:
    values = trained_three_role_skill_simulation_kwargs()
    values.update(
        {
            "shooter_target": _PHYSICAL_TARGET,
            "shooter_policy_target": _POLICY_TARGET,
            "passer_origin": _PASSER_ORIGIN,
            "passer_ball_local_xy": _PASSER_BALL_LOCAL_XY,
            "ball_ground_friction": 0.10,
            "receiver_phase_sync_enabled": False,
            "goal_spec": three_role_goal_spec(),
        }
    )
    return values


def run_three_role_development(
    *, asset_root: Path, output_dir: Path, source_checkout: Path
) -> ThreeRoleDevelopmentEvidence:
    """Run anticipation versus reactive control and retain an auditable candidate."""

    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("three-role development evidence must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    kwargs = three_role_development_kwargs()
    keeper = kwargs.get("goalkeeper_config")
    if not isinstance(keeper, G1GoalkeeperConfig) or not keeper.anticipation_enabled:
        raise RuntimeError("three-role development requires the anticipation policy")
    role_policy_hashes = _role_policy_hashes(kwargs)
    request = {
        "schema_version": "rosclaw_soccer.three_role_development_request.v1",
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "passer_origin_m": list(_PASSER_ORIGIN),
        "passer_ball_local_xy_m": list(_PASSER_BALL_LOCAL_XY),
        "shooter_start_sec": kwargs["shooter_start_sec"],
        "physical_scoring_target_m": list(_PHYSICAL_TARGET),
        "inverse_calibrated_policy_target_m": list(_POLICY_TARGET),
        "goal_spec": asdict(three_role_goal_spec()),
        "goalkeeper_config": asdict(keeper),
        "role_policy_hashes": role_policy_hashes,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "runtime": _runtime_manifest(),
    }
    request["environment_hash"] = hash_json(request["runtime"])
    request_path = output / "request.json"
    _write_json(request_path, request)

    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
    strict_replay = bool(
        result.to_dict() == replay_result.to_dict()
        and trajectory_digest(trajectory) == trajectory_digest(replay_trajectory)
    )
    reactive_kwargs = dict(kwargs)
    reactive_kwargs["goalkeeper_config"] = replace(keeper, anticipation_enabled=False)
    reactive_result, _ = simulate_shared_world(asset_root, **reactive_kwargs)
    metrics = _trajectory_metrics(trajectory, result)
    roll_metrics, _ = _rolling_metrics(trajectory, result, three_role_goal_spec().ball_radius_m)

    trajectory_path = output / "trajectory.npz"
    np.savez_compressed(trajectory_path, **trajectory)  # type: ignore[arg-type]
    reactive_distance = reactive_result.goalkeeper_lateral_displacement_m
    improvement = (
        (result.goalkeeper_lateral_displacement_m - reactive_distance) / reactive_distance
        if reactive_distance > 0.0
        else 0.0
    )
    evidence = ThreeRoleDevelopmentEvidence(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        backend_commit=qualification.backend_commit,
        implementation_hash=_implementation_hash(),
        request_hash=_file_hash(request_path),
        trajectory_hash=_file_hash(trajectory_path),
        trajectory_digest=trajectory_digest(trajectory),
        strict_replay=strict_replay,
        result=result,
        reactive_goalkeeper_lateral_displacement_m=reactive_distance,
        anticipation_displacement_improvement_fraction=improvement,
        pass_distance_m=float(metrics["pass_distance_m"]),
        shot_distance_m=float(metrics["shot_distance_m"]),
        pass_speed_start_mps=float(metrics["pass_speed_start_mps"]),
        pass_speed_end_mps=float(metrics["pass_speed_end_mps"]),
        pass_speed_max_positive_step_mps=float(metrics["pass_speed_max_positive_step_mps"]),
        pass_speed_positive_step_count=int(metrics["pass_speed_positive_step_count"]),
        rolling_authenticity_passed=roll_metrics.passed,
        rolling_median_slip_ratio=roll_metrics.median_slip_ratio,
        role_policy_hashes=role_policy_hashes,
    )
    _write_json(output / "g1-three-player-showcase.json", evidence.to_dict())
    return evidence


def _trajectory_metrics(
    trajectory: dict[str, np.ndarray], result: G1SharedWorldResult
) -> dict[str, float | int]:
    if result.pass_contact_time_sec is None or result.shot_contact_time_sec is None:
        raise ValueError("three-role metrics require both measured contacts")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    pose = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    velocity = np.asarray(trajectory["ball_velocity"], dtype=np.float64)
    pass_index = int(np.searchsorted(time, result.pass_contact_time_sec, side="left"))
    shot_index = int(np.searchsorted(time, result.shot_contact_time_sec, side="left"))
    crossing = np.flatnonzero(pose[:, 0] >= _PHYSICAL_TARGET[0])
    crossing_index = int(crossing[0]) if crossing.size else len(pose) - 1
    rolling_speed = np.linalg.norm(velocity[pass_index:shot_index, :2], axis=1)
    settled_speed = rolling_speed[min(2, len(rolling_speed) - 1) :]
    positive = np.diff(settled_speed)
    positive = positive[positive > 0.01]
    return {
        "pass_distance_m": float(np.linalg.norm(pose[shot_index, :2] - pose[pass_index, :2])),
        "shot_distance_m": float(np.linalg.norm(pose[crossing_index, :2] - pose[shot_index, :2])),
        "pass_speed_start_mps": float(settled_speed[0]),
        "pass_speed_end_mps": float(settled_speed[-1]),
        "pass_speed_max_positive_step_mps": 0.0 if not positive.size else float(np.max(positive)),
        "pass_speed_positive_step_count": int(positive.size),
    }


def _rolling_metrics(
    trajectory: dict[str, np.ndarray], result: G1SharedWorldResult, ball_radius_m: float
) -> tuple[RollingAuthenticityMetrics, np.ndarray]:
    if result.pass_contact_time_sec is None or result.shot_contact_time_sec is None:
        raise ValueError("rolling metrics require both contacts")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    start = int(np.searchsorted(time, result.pass_contact_time_sec + 0.10, side="left"))
    end = int(np.searchsorted(time, result.shot_contact_time_sec - 0.15, side="right"))
    metrics, slip = measure_rolling_authenticity(
        time=time[start:end],
        ball_pose=np.asarray(trajectory["ball_pose"])[start:end],
        ball_velocity=np.asarray(trajectory["ball_velocity"])[start:end],
        ball_radius_m=ball_radius_m,
        ignore_initial_sec=0.0,
    )
    return metrics, np.asarray(slip, dtype=np.float64)


def _role_policy_hashes(kwargs: dict[str, Any]) -> dict[str, str]:
    return {
        "passer": hash_json(kwargs.get("passer_parameter_overrides", {})),
        "shooter": hash_json(
            {
                "start_sec": kwargs["shooter_start_sec"],
                "parameters": kwargs.get("shooter_parameter_overrides", {}),
                "guard": asdict(kwargs["shooter_joint_guard_config"]),
            }
        ),
        "goalkeeper": hash_json(asdict(kwargs["goalkeeper_config"])),
    }


def _runtime_manifest() -> dict[str, str]:
    import mujoco
    import onnxruntime
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "onnxruntime": onnxruntime.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
    }


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("shared_world.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "ThreeRoleDevelopmentEvidence",
    "run_three_role_development",
    "three_role_development_kwargs",
    "three_role_goal_spec",
]
