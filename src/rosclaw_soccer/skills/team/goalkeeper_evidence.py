"""Strict evidence loop for a learned, safe goalkeeper block candidate."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
from rosclaw_soccer.skills.team.goalkeeper_learning import (
    GoalkeeperBlockSearchResult,
    goalkeeper_block_parent_config,
    search_goalkeeper_block_candidate,
)
from rosclaw_soccer.skills.team.shared_world import G1GoalkeeperConfig, simulate_shared_world


@dataclass(frozen=True)
class GoalkeeperBlockEvidence:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    implementation_hash: str
    request_hash: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    search: GoalkeeperBlockSearchResult
    result: dict[str, Any]
    baseline_goal_crossed: bool
    baseline_goalkeeper_contact_observed: bool
    pass_distance_m: float
    shot_to_block_distance_m: float
    pass_speed_start_mps: float
    pass_speed_end_mps: float
    pass_speed_positive_step_count: int
    rolling_authenticity_passed: bool
    rolling_median_slip_ratio: float
    selected_policy_hash: str
    passed: bool
    promotion_status: str = "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED"
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    simultaneous_three_body_physics: bool = True
    shared_ball_state: bool = True
    unified_physics_and_render_scene: bool = True
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_block_evidence.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "search": self.search.to_dict(),
            "claims": {
                "strict_deterministic_replay": self.strict_replay,
                "single_shared_ball": True,
                "goalkeeper_uses_shooter_proprioception_before_contact": True,
                "goalkeeper_ball_contact_achieved": bool(
                    self.result["goalkeeper_ball_contact_observed"]
                ),
                "goalkeeper_save_achieved": bool(self.result["goalkeeper_save_observed"]),
                "candidate_promoted": False,
                "pixels_used_for_promotion": False,
                "real_hardware": False,
            },
        }


def run_goalkeeper_block_development(
    *, asset_root: Path, output_dir: Path, source_checkout: Path
) -> GoalkeeperBlockEvidence:
    """Search, replay and persist one goalkeeper save outside the checkout."""

    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("goalkeeper development evidence must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    kwargs = three_role_development_kwargs()
    keeper = kwargs.get("goalkeeper_config")
    if not isinstance(keeper, G1GoalkeeperConfig):
        raise RuntimeError("goalkeeper development requires a keeper parent")

    search = search_goalkeeper_block_candidate(asset_root=asset_root, simulation_kwargs=kwargs)
    if not search.passed or search.selected_config is None or search.selected_trial is None:
        raise RuntimeError("goalkeeper block discovery found no safe save")
    candidate_kwargs = dict(kwargs)
    candidate_kwargs["goalkeeper_config"] = search.selected_config
    request = {
        "schema_version": "rosclaw_soccer.goalkeeper_block_request.v1",
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "passer_origin_m": list(kwargs["passer_origin"]),
        "passer_ball_local_xy_m": list(kwargs["passer_ball_local_xy"]),
        "shooter_start_sec": kwargs["shooter_start_sec"],
        "physical_scoring_target_m": list(kwargs["shooter_target"]),
        "inverse_calibrated_policy_target_m": list(kwargs["shooter_policy_target"]),
        "goal_spec": asdict(three_role_goal_spec()),
        "goalkeeper_config": asdict(search.selected_config),
        "selected_policy_hash": search.selected_trial.policy_hash,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "runtime": _runtime_manifest(),
    }
    request["environment_hash"] = hash_json(request["runtime"])
    request_path = output / "request.json"
    _write_json(request_path, request)

    result, trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    strict_replay = bool(
        result.to_dict() == replay_result.to_dict()
        and trajectory_digest(trajectory) == trajectory_digest(replay_trajectory)
    )
    baseline_kwargs = dict(kwargs)
    baseline_kwargs["goalkeeper_config"] = goalkeeper_block_parent_config(keeper)
    baseline, _ = simulate_shared_world(asset_root, **baseline_kwargs)
    metrics = _metrics(trajectory, result.to_dict(), three_role_goal_spec().ball_radius_m)
    trajectory_path = output / "trajectory.npz"
    np.savez_compressed(trajectory_path, **trajectory)  # type: ignore[arg-type]
    passed = bool(
        strict_replay
        and search.passed
        and baseline.goal_crossed
        and not baseline.goalkeeper_ball_contact_observed
        and result.finite_state
        and result.pass_contact_observed
        and result.shot_contact_observed
        and result.goalkeeper_ball_contact_observed
        and result.goalkeeper_save_observed
        and not result.goal_crossed
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
        and not result.goalkeeper_joint_limit_violation
        and (result.goalkeeper_min_pelvis_height_m or 0.0) >= 0.65
        and bool(metrics["rolling_authenticity_passed"])
    )
    evidence = GoalkeeperBlockEvidence(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        backend_commit=qualification.backend_commit,
        implementation_hash=_implementation_hash(),
        request_hash=_file_hash(request_path),
        trajectory_hash=_file_hash(trajectory_path),
        trajectory_digest=trajectory_digest(trajectory),
        strict_replay=strict_replay,
        search=search,
        result=result.to_dict(),
        baseline_goal_crossed=baseline.goal_crossed,
        baseline_goalkeeper_contact_observed=baseline.goalkeeper_ball_contact_observed,
        pass_distance_m=float(metrics["pass_distance_m"]),
        shot_to_block_distance_m=float(metrics["shot_to_block_distance_m"]),
        pass_speed_start_mps=float(metrics["pass_speed_start_mps"]),
        pass_speed_end_mps=float(metrics["pass_speed_end_mps"]),
        pass_speed_positive_step_count=int(metrics["pass_speed_positive_step_count"]),
        rolling_authenticity_passed=bool(metrics["rolling_authenticity_passed"]),
        rolling_median_slip_ratio=float(metrics["rolling_median_slip_ratio"]),
        selected_policy_hash=search.selected_trial.policy_hash,
        passed=passed,
        promotion_status=(
            "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED" if passed else "REJECTED_DEVELOPMENT"
        ),
    )
    _write_json(output / "g1-goalkeeper-block.json", evidence.to_dict())
    return evidence


def _metrics(
    trajectory: dict[str, np.ndarray], result: dict[str, Any], ball_radius_m: float
) -> dict[str, float | bool]:
    pass_time = float(result["pass_contact_time_sec"])
    shot_time = float(result["shot_contact_time_sec"])
    block_time = float(result["goalkeeper_ball_contact_time_sec"])
    time = np.asarray(trajectory["time"], dtype=np.float64)
    pose = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    pass_index = int(np.searchsorted(time, pass_time))
    shot_index = int(np.searchsorted(time, shot_time))
    block_index = int(np.searchsorted(time, block_time))
    start = int(np.searchsorted(time, pass_time + 0.10, side="left"))
    end = int(np.searchsorted(time, shot_time - 0.15, side="right"))
    rolling, _ = measure_rolling_authenticity(
        time=time[start:end],
        ball_pose=pose[start:end],
        ball_velocity=np.asarray(trajectory["ball_velocity"])[start:end],
        ball_radius_m=ball_radius_m,
        ignore_initial_sec=0.0,
    )
    rolling_speed = np.linalg.norm(
        np.asarray(trajectory["ball_velocity"], dtype=np.float64)[start:end, :2], axis=1
    )
    positive_steps = np.diff(rolling_speed)
    return {
        "pass_distance_m": float(np.linalg.norm(pose[shot_index, :2] - pose[pass_index, :2])),
        "shot_to_block_distance_m": float(
            np.linalg.norm(pose[block_index, :2] - pose[shot_index, :2])
        ),
        "rolling_authenticity_passed": rolling.passed,
        "rolling_median_slip_ratio": rolling.median_slip_ratio,
        "pass_speed_start_mps": float(rolling_speed[0]),
        "pass_speed_end_mps": float(rolling_speed[-1]),
        "pass_speed_positive_step_count": int(np.sum(positive_steps > 0.01)),
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
    for name in (
        "goalkeeper_evidence.py",
        "goalkeeper_learning.py",
        "shared_world.py",
    ):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    field = Path(__file__).resolve().parents[2] / "world/field.py"
    digest.update(field.name.encode())
    digest.update(field.read_bytes())
    return "sha256:" + digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["GoalkeeperBlockEvidence", "run_goalkeeper_block_development"]
