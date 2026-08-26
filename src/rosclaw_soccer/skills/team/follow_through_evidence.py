"""Strict SIM_ONLY evidence for visible, stable G1 follow-through growth."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.football_motion_prior import load_g1_football_motion_prior
from rosclaw_soccer.growth.mosaic_agility_prior import load_g1_mosaic_agility_prior
from rosclaw_soccer.physics.rolling_authenticity import measure_rolling_authenticity
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets, trajectory_digest
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.agility_growth import (
    G1AgilityCandidate,
    G1FollowThroughAgilityMetrics,
    measure_g1_follow_through_agility,
)
from rosclaw_soccer.skills.team.development_evidence import (
    three_role_development_kwargs,
    three_role_goal_spec,
)
from rosclaw_soccer.skills.team.follow_through_growth import (
    G1FollowThroughCandidate,
    G1FollowThroughSearchResult,
    search_g1_follow_through_candidate,
)
from rosclaw_soccer.skills.team.imitation_evidence import _trajectory_metrics
from rosclaw_soccer.skills.team.imitation_learning import (
    G1MotionNaturalnessMetrics,
    measure_g1_motion_naturalness,
)
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world


@dataclass(frozen=True)
class G1FollowThroughEvidence:
    body_hash: str
    kick_prior_hash: str
    motion_prior_hash: str
    contact_prior_hash: str
    mosaic_prior_hash: str
    backend_commit: str
    implementation_hash: str
    request_hash: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    search: G1FollowThroughSearchResult
    result: dict[str, Any]
    candidate_naturalness: G1MotionNaturalnessMetrics
    parent_follow_through: G1FollowThroughAgilityMetrics
    candidate_follow_through: G1FollowThroughAgilityMetrics
    pass_distance_m: float
    shot_distance_m: float
    pass_speed_start_mps: float
    pass_speed_end_mps: float
    pass_speed_max_positive_step_mps: float
    pass_speed_positive_step_count: int
    rolling_authenticity_passed: bool
    rolling_median_slip_ratio: float
    selected_candidate_hash: str
    numerical_thread_contract: dict[str, str]
    passed: bool
    promotion_status: str = "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED"
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    simultaneous_three_body_physics: bool = True
    shared_ball_state: bool = True
    unified_physics_and_render_scene: bool = True
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.g1_follow_through_evidence.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "search": self.search.to_dict(),
            "candidate_naturalness": asdict(self.candidate_naturalness),
            "parent_follow_through": asdict(self.parent_follow_through),
            "candidate_follow_through": asdict(self.candidate_follow_through),
            "claims": {
                "semantic_mosaic_soccer_teacher": True,
                "endpoint_neutral_pose_residual": True,
                "arm_only_plasticity_boundary": True,
                "visible_plasticity_floor_passed": self.passed,
                "counterfactual_parent_retained": True,
                "local_neighborhood_gate_passed": self.search.passed,
                "teacher_direct_torque_output": False,
                "single_shared_ball": True,
                "candidate_promoted": False,
                "pixels_used_for_promotion": False,
                "real_hardware": False,
            },
        }


def run_g1_follow_through_development(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    contact_prior_path: Path,
    mosaic_prior_path: Path,
    output_dir: Path,
    source_checkout: Path,
    candidates: tuple[G1FollowThroughCandidate, ...] | None = None,
) -> G1FollowThroughEvidence:
    """Search, strictly replay, and persist one visible follow-through basin."""

    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    paths = tuple(
        path.expanduser().resolve()
        for path in (motion_prior_path, contact_prior_path, mosaic_prior_path)
    )
    if output == checkout or checkout in output.parents:
        raise ValueError("follow-through evidence must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    motion = load_g1_football_motion_prior(paths[0])
    contact = load_g1_football_motion_prior(paths[1])
    mosaic = load_g1_mosaic_agility_prior(paths[2])
    kwargs = three_role_development_kwargs()
    search = search_g1_follow_through_candidate(
        asset_root=asset_root,
        motion_prior_path=paths[0],
        contact_prior_path=paths[1],
        mosaic_prior_path=paths[2],
        simulation_kwargs=kwargs,
        candidates=candidates,
    )
    candidate = search.selected_candidate
    selected = search.selected_trial
    if candidate is None or selected is None or not search.passed:
        raise RuntimeError("follow-through search found no robust visible candidate")
    base = G1AgilityCandidate(1.14, 1.15)
    candidate_kwargs = {
        **kwargs,
        **base.simulation_overrides(motion_prior_path=paths[0], contact_prior_path=paths[1]),
        **candidate.simulation_overrides(mosaic_prior_path=paths[2]),
    }
    result, trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    strict_replay = bool(
        result.to_dict() == replay_result.to_dict()
        and trajectory_digest(trajectory) == trajectory_digest(replay_trajectory)
    )
    naturalness = measure_g1_motion_naturalness(
        trajectory=trajectory,
        result=result,
        prior=motion,
        contact_policy_frame=253,
    )
    follow_through = measure_g1_follow_through_agility(
        trajectory,
        center_policy_frame=candidate.center_policy_frame,
    )
    parent_follow_through = selected.parent_agility
    metrics = _trajectory_metrics(trajectory, result.to_dict())
    if result.pass_contact_time_sec is None or result.shot_contact_time_sec is None:
        raise RuntimeError("follow-through replay is missing contact time")
    roll_start = int(np.searchsorted(trajectory["time"], result.pass_contact_time_sec + 0.10))
    roll_end = int(np.searchsorted(trajectory["time"], result.shot_contact_time_sec - 0.15))
    rolling, _ = measure_rolling_authenticity(
        time=np.asarray(trajectory["time"])[roll_start:roll_end],
        ball_pose=np.asarray(trajectory["ball_pose"])[roll_start:roll_end],
        ball_velocity=np.asarray(trajectory["ball_velocity"])[roll_start:roll_end],
        ball_radius_m=three_role_goal_spec().ball_radius_m,
        ignore_initial_sec=0.0,
    )
    trajectory_path = output / "trajectory.npz"
    np.savez_compressed(trajectory_path, **trajectory)  # type: ignore[arg-type]
    thread_contract = _numerical_thread_contract()
    request = {
        "schema_version": "rosclaw_soccer.g1_follow_through_request.v1",
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "motion_prior_hash": motion.prior_hash,
        "motion_prior_path_hash": _file_hash(paths[0]),
        "contact_prior_hash": contact.prior_hash,
        "contact_prior_path_hash": _file_hash(paths[1]),
        "mosaic_prior_hash": mosaic.prior_hash,
        "mosaic_prior_path_hash": _file_hash(paths[2]),
        "mosaic_teacher_skill_id": mosaic.teacher_skill_id,
        "selected_candidate": asdict(candidate),
        "selected_candidate_hash": candidate.candidate_hash,
        "neighborhood_candidate_hashes": [trial.candidate_hash for trial in search.trials],
        "neighborhood_eligible_fraction": search.neighborhood_eligible_fraction,
        "passer_origin_m": list(kwargs["passer_origin"]),
        "passer_ball_local_xy_m": list(kwargs["passer_ball_local_xy"]),
        "shooter_start_sec": kwargs["shooter_start_sec"],
        "physical_scoring_target_m": list(kwargs["shooter_target"]),
        "inverse_calibrated_policy_target_m": list(kwargs["shooter_policy_target"]),
        "goal_spec": asdict(three_role_goal_spec()),
        "goalkeeper_config": asdict(kwargs["goalkeeper_config"]),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "numerical_thread_contract": thread_contract,
        "runtime": _runtime_manifest(),
        "trajectory_digest_commitment": trajectory_digest(trajectory),
    }
    request["environment_hash"] = hash_json(
        {"runtime": request["runtime"], "threads": thread_contract}
    )
    request_path = output / "request.json"
    _write_json(request_path, request)
    passed = bool(
        strict_replay
        and result.passed
        and selected.eligible
        and search.passed
        and result.shooter_motion_prior_hash == motion.prior_hash
        and result.shooter_contact_prior_hash == contact.prior_hash
        and result.shooter_agility_prior_hash == mosaic.prior_hash
        and naturalness.target_error_m <= 0.03
        and naturalness.post_contact_support_slip_m <= 0.06
        and naturalness.post_contact_peak_backward_velocity_mps <= 0.01
        and follow_through.arm_excursion_rms_rad
        >= 1.08 * parent_follow_through.arm_excursion_rms_rad
        and follow_through.upper_body_motion_energy
        >= 1.10 * parent_follow_through.upper_body_motion_energy
        and rolling.passed
    )
    evidence = G1FollowThroughEvidence(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        motion_prior_hash=motion.prior_hash,
        contact_prior_hash=contact.prior_hash,
        mosaic_prior_hash=mosaic.prior_hash,
        backend_commit=qualification.backend_commit,
        implementation_hash=_implementation_hash(),
        request_hash=_file_hash(request_path),
        trajectory_hash=_file_hash(trajectory_path),
        trajectory_digest=trajectory_digest(trajectory),
        strict_replay=strict_replay,
        search=search,
        result=result.to_dict(),
        candidate_naturalness=naturalness,
        parent_follow_through=parent_follow_through,
        candidate_follow_through=follow_through,
        pass_distance_m=float(metrics["pass_distance_m"]),
        shot_distance_m=float(metrics["shot_distance_m"]),
        pass_speed_start_mps=float(metrics["pass_speed_start_mps"]),
        pass_speed_end_mps=float(metrics["pass_speed_end_mps"]),
        pass_speed_max_positive_step_mps=float(metrics["pass_speed_max_positive_step_mps"]),
        pass_speed_positive_step_count=int(metrics["pass_speed_positive_step_count"]),
        rolling_authenticity_passed=rolling.passed,
        rolling_median_slip_ratio=rolling.median_slip_ratio,
        selected_candidate_hash=candidate.candidate_hash,
        numerical_thread_contract=thread_contract,
        passed=passed,
    )
    _write_json(output / "g1-follow-through-growth.json", evidence.to_dict())
    return evidence


def _numerical_thread_contract() -> dict[str, str]:
    return {
        name: os.environ.get(name, "UNSET")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
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
    for path in (
        Path(__file__),
        Path(__file__).with_name("follow_through_growth.py"),
        Path(__file__).with_name("agility_growth.py"),
        Path(__file__).with_name("shared_world.py"),
        Path(__file__).parents[2] / "growth" / "mosaic_agility_prior.py",
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


__all__ = ["G1FollowThroughEvidence", "run_g1_follow_through_development"]
