"""Strict evidence for a MotionDecode-informed shared-world G1 candidate."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.football_motion_prior import load_g1_football_motion_prior
from rosclaw_soccer.physics.rolling_authenticity import measure_rolling_authenticity
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.development_evidence import (
    three_role_development_kwargs,
    three_role_goal_spec,
)
from rosclaw_soccer.skills.team.imitation_learning import (
    G1ImitationCandidate,
    G1ImitationSearchResult,
    G1MotionNaturalnessMetrics,
    measure_g1_motion_naturalness,
    search_g1_imitation_candidate,
)
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world


@dataclass(frozen=True)
class G1ImitationEvidence:
    body_hash: str
    kick_prior_hash: str
    motion_prior_hash: str
    backend_commit: str
    implementation_hash: str
    request_hash: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    search: G1ImitationSearchResult
    result: dict[str, Any]
    candidate_naturalness: G1MotionNaturalnessMetrics
    pass_distance_m: float
    shot_distance_m: float
    pass_speed_start_mps: float
    pass_speed_end_mps: float
    pass_speed_max_positive_step_mps: float
    pass_speed_positive_step_count: int
    rolling_authenticity_passed: bool
    rolling_median_slip_ratio: float
    selected_candidate_hash: str
    passed: bool
    promotion_status: str = "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED"
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    simultaneous_three_body_physics: bool = True
    shared_ball_state: bool = True
    unified_physics_and_render_scene: bool = True
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.g1_imitation_evidence.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "search": self.search.to_dict(),
            "candidate_naturalness": asdict(self.candidate_naturalness),
            "claims": {
                "strict_deterministic_replay": self.strict_replay,
                "motiondecode_whole_body_position_teacher": True,
                "motiondecode_whole_body_velocity_teacher": True,
                "single_shared_ball": True,
                "candidate_promoted": False,
                "pixels_used_for_promotion": False,
                "real_hardware": False,
            },
        }


def run_g1_imitation_development(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    output_dir: Path,
    source_checkout: Path,
    candidates: tuple[G1ImitationCandidate, ...] | None = None,
) -> G1ImitationEvidence:
    """Search, replay and persist one fail-closed imitation candidate."""

    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    prior_path = motion_prior_path.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("imitation evidence must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    motion_prior = load_g1_football_motion_prior(prior_path)
    kwargs = three_role_development_kwargs()
    search = search_g1_imitation_candidate(
        asset_root=asset_root,
        motion_prior_path=prior_path,
        simulation_kwargs=kwargs,
        candidates=candidates,
    )
    if search.selected_candidate is None or search.selected_trial is None or not search.passed:
        raise RuntimeError("imitation search found no safe natural candidate")
    candidate = search.selected_candidate
    candidate_kwargs = {**kwargs, **candidate.simulation_overrides(prior_path)}
    result, trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    strict_replay = bool(
        result.to_dict() == replay_result.to_dict()
        and trajectory_digest(trajectory) == trajectory_digest(replay_trajectory)
    )
    naturalness = measure_g1_motion_naturalness(
        trajectory=trajectory,
        result=result,
        prior=motion_prior,
        contact_policy_frame=candidate.contact_policy_frame,
    )
    metrics = _trajectory_metrics(trajectory, result.to_dict())
    rolling, _ = _rolling_metrics(trajectory, result.to_dict())
    trajectory_path = output / "trajectory.npz"
    np.savez_compressed(trajectory_path, **trajectory)  # type: ignore[arg-type]
    request = {
        "schema_version": "rosclaw_soccer.g1_imitation_request.v1",
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "motion_prior_hash": motion_prior.prior_hash,
        "motion_prior_path_hash": _file_hash(prior_path),
        "selected_candidate": asdict(candidate),
        "selected_candidate_hash": candidate.candidate_hash,
        "passer_origin_m": list(kwargs["passer_origin"]),
        "passer_ball_local_xy_m": list(kwargs["passer_ball_local_xy"]),
        "shooter_start_sec": kwargs["shooter_start_sec"],
        "physical_scoring_target_m": list(kwargs["shooter_target"]),
        "inverse_calibrated_policy_target_m": list(kwargs["shooter_policy_target"]),
        "goal_spec": asdict(three_role_goal_spec()),
        "goalkeeper_config": asdict(kwargs["goalkeeper_config"]),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "runtime": _runtime_manifest(),
        "trajectory_digest_commitment": trajectory_digest(trajectory),
    }
    request["environment_hash"] = hash_json(request["runtime"])
    request_path = output / "request.json"
    _write_json(request_path, request)

    passed = bool(
        strict_replay
        and search.passed
        and result.passed
        and result.shooter_motion_prior_hash == request["motion_prior_hash"]
        and search.selected_trial.eligible
        and result.target_error_m is not None
        and result.target_error_m <= 0.10
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
        and rolling.passed
    )
    evidence = G1ImitationEvidence(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        motion_prior_hash=str(request["motion_prior_hash"]),
        backend_commit=qualification.backend_commit,
        implementation_hash=_implementation_hash(),
        request_hash=_file_hash(request_path),
        trajectory_hash=_file_hash(trajectory_path),
        trajectory_digest=trajectory_digest(trajectory),
        strict_replay=strict_replay,
        search=search,
        result=result.to_dict(),
        candidate_naturalness=naturalness,
        pass_distance_m=float(metrics["pass_distance_m"]),
        shot_distance_m=float(metrics["shot_distance_m"]),
        pass_speed_start_mps=float(metrics["pass_speed_start_mps"]),
        pass_speed_end_mps=float(metrics["pass_speed_end_mps"]),
        pass_speed_max_positive_step_mps=float(metrics["pass_speed_max_positive_step_mps"]),
        pass_speed_positive_step_count=int(metrics["pass_speed_positive_step_count"]),
        rolling_authenticity_passed=rolling.passed,
        rolling_median_slip_ratio=rolling.median_slip_ratio,
        selected_candidate_hash=candidate.candidate_hash,
        passed=passed,
    )
    _write_json(output / "g1-imitation-growth.json", evidence.to_dict())
    return evidence


def _trajectory_metrics(
    trajectory: dict[str, np.ndarray], result: dict[str, Any]
) -> dict[str, float | int]:
    pass_time = float(result["pass_contact_time_sec"])
    shot_time = float(result["shot_contact_time_sec"])
    time = np.asarray(trajectory["time"], dtype=np.float64)
    pose = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    velocity = np.asarray(trajectory["ball_velocity"], dtype=np.float64)
    pass_index = int(np.searchsorted(time, pass_time, side="left"))
    shot_index = int(np.searchsorted(time, shot_time, side="left"))
    crossing = np.flatnonzero(pose[:, 0] >= three_role_goal_spec().plane_x_m)
    crossing_index = int(crossing[0]) if crossing.size else len(pose) - 1
    speed = np.linalg.norm(velocity[pass_index + 2 : shot_index, :2], axis=1)
    positive = np.diff(speed)
    return {
        "pass_distance_m": float(np.linalg.norm(pose[shot_index, :2] - pose[pass_index, :2])),
        "shot_distance_m": float(np.linalg.norm(pose[crossing_index, :2] - pose[shot_index, :2])),
        "pass_speed_start_mps": float(speed[0]),
        "pass_speed_end_mps": float(speed[-1]),
        "pass_speed_max_positive_step_mps": float(0.0 if not positive.size else np.max(positive)),
        "pass_speed_positive_step_count": int(np.sum(positive > 0.01)),
    }


def _rolling_metrics(trajectory: dict[str, np.ndarray], result: dict[str, Any]) -> tuple[Any, Any]:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    start = int(np.searchsorted(time, float(result["pass_contact_time_sec"]) + 0.10))
    end = int(np.searchsorted(time, float(result["shot_contact_time_sec"]) - 0.15))
    return measure_rolling_authenticity(
        time=time[start:end],
        ball_pose=np.asarray(trajectory["ball_pose"])[start:end],
        ball_velocity=np.asarray(trajectory["ball_velocity"])[start:end],
        ball_radius_m=three_role_goal_spec().ball_radius_m,
        ignore_initial_sec=0.0,
    )


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
    for path in (
        Path(__file__),
        Path(__file__).with_name("imitation_learning.py"),
        Path(__file__).with_name("shared_world.py"),
    ):
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


__all__ = ["G1ImitationEvidence", "run_g1_imitation_development"]
