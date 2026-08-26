"""Distil a striker-owned contact actor inside the physical three-G1 duel."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, TypeVar, cast

from rosclaw_soccer.growth.approach_strike_residual import G1ApproachStrikeResidualConfig
from rosclaw_soccer.growth.ballistic_contact_impulse_actor import (
    derive_g1_ballistic_contact_impulse_actor,
)
from rosclaw_soccer.growth.football_motion_prior import load_g1_football_motion_prior
from rosclaw_soccer.providers.g1.learned_runup import G1LearnedRunupConfig
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig
from rosclaw_soccer.skills.shoot.free_kick import (
    G1FreeKickFlowConfig,
    run_g1_free_kick_showcase,
)
from rosclaw_soccer.skills.team.front_duel import G1FrontDuelConfig
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_ConfigT = TypeVar("_ConfigT")
_TUPLE_FIELDS = {
    "joint_gain_scales",
    "strike_gain_scales",
    "follow_through_gain_scales",
    "ballistic_contact_residual_rad",
    "ballistic_contact_torque_residual_nm",
    "ballistic_contact_torque_preload_nm",
    "ballistic_contact_torque_phase_offset_sec",
    "ballistic_counterbalance_torque_residual_nm",
}


@dataclass(frozen=True)
class G1FrontDuelTrainingAssets:
    asset_root: Path
    gait_policy_root: Path
    sonic_model_root: Path
    seed_request: Path
    approach_strike_candidate: Path
    football_motion_prior: Path


@dataclass(frozen=True)
class G1FrontDuelCurriculum:
    target_y_m: float = 1.8508
    target_z_m: float = 0.8649
    ball_mass_kg: float = 0.41
    teacher_force_pairs_n: tuple[tuple[float, float], ...] = (
        (180.0, 250.0),
        (160.0, 250.0),
        (200.0, 250.0),
        (220.0, 250.0),
        (10.0, 10.0),
        (20.0, 10.0),
        (10.0, 20.0),
        (30.0, 30.0),
    )
    worker_count: int = 2
    schema_version: str = "rosclaw_soccer.g1_front_duel_curriculum.v2"

    def __post_init__(self) -> None:
        if not 0.50 <= self.target_y_m <= 3.0:
            raise ValueError("front-duel target y must be a difficult lateral target")
        if not 0.60 <= self.target_z_m <= 1.40:
            raise ValueError("front-duel target z must be above a low ground shot")
        if not 0.41 <= self.ball_mass_kg <= 0.45:
            raise ValueError("front-duel football mass must stay inside the regulation range")
        if len(self.teacher_force_pairs_n) < 8 or len(set(self.teacher_force_pairs_n)) != len(
            self.teacher_force_pairs_n
        ):
            raise ValueError("front-duel actor curriculum requires eight unique probes")
        if not 1 <= self.worker_count <= 4:
            raise ValueError("front-duel worker count must be in [1, 4]")


@dataclass(frozen=True)
class G1FrontDuelTrainingReport:
    actor_path: str
    actor_hash: str
    final_evidence_path: str
    strict_replay: bool
    front_contact_observed: bool
    front_ball_apex_height_m: float
    front_goal_crossing_xyz_m: tuple[float, float, float] | None
    front_goal_plane_target_error_m: float | None
    front_post_kick_fall: bool
    goalkeeper_minimum_pelvis_height_m: float | None
    goalkeeper_peak_tilt_rad: float | None
    goalkeeper_reaction_frames: int
    final_evidence_passed: bool
    development_only: bool = True
    schema_version: str = "rosclaw_soccer.g1_front_duel_training_report.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config_from_json(cls: type[_ConfigT], payload: dict[str, Any]) -> _ConfigT:
    accepted = set(cast(dict[str, Any], cast(Any, cls).__dataclass_fields__))
    values = {
        key: tuple(value) if key in _TUPLE_FIELDS and isinstance(value, list) else value
        for key, value in payload.items()
        if key in accepted and key != "schema_version"
    }
    return cls(**values)


def _run_probe(payload: dict[str, Any]) -> str:
    assets = G1FrontDuelTrainingAssets(
        **{key: Path(value) for key, value in dict(payload["assets"]).items()}
    )
    request = dict(payload["request"])
    flow = _config_from_json(G1FreeKickFlowConfig, dict(request["flow_config"]))
    goal = replace(
        _config_from_json(G1TrainingGoalSpec, dict(request["goal_spec"])),
        target_y_m=float(payload["target_y_m"]),
        target_z_m=float(payload["target_z_m"]),
        ball_mass_kg=float(payload["ball_mass_kg"]),
    )
    lateral_force, vertical_force = cast(tuple[float, float], payload["force_pair_n"])
    flow = replace(
        flow,
        ballistic_contact_impulse_actor_hash=None,
        torque_authority_projection_ratio=0.99,
        contact_task_direction_projection_enabled=False,
        shot_loft_teacher_target_vy_mps=10.0,
        shot_loft_teacher_target_vz_mps=7.0,
        shot_loft_teacher_lateral_gain_n_per_mps=35.0,
        shot_loft_teacher_gain_n_per_mps=50.0,
        shot_loft_teacher_max_lateral_force_n=lateral_force,
        shot_loft_teacher_max_force_n=vertical_force,
        shot_loft_teacher_max_foot_ball_distance_m=0.18,
    )
    output_dir = Path(str(payload["output_dir"]))
    evidence = run_g1_free_kick_showcase(
        asset_root=assets.asset_root,
        gait_policy_root=assets.gait_policy_root,
        output_dir=output_dir,
        source_checkout=Path(str(payload["source_checkout"])),
        runup_config=_config_from_json(G1LearnedRunupConfig, dict(request["runup_config"])),
        flow_config=flow,
        goal_spec=goal,
        sonic_model_root=assets.sonic_model_root,
        sonic_runup_config=_config_from_json(
            G1SonicRunupConfig,
            dict(request["sonic_runup_config"]),
        ),
        approach_strike_candidate_path=assets.approach_strike_candidate,
        approach_strike_residual_config=_config_from_json(
            G1ApproachStrikeResidualConfig,
            dict(request["approach_strike_residual_config"]),
        ),
        football_motion_prior=load_g1_football_motion_prior(assets.football_motion_prior),
        front_duel_config=G1FrontDuelConfig(),
    )
    if not evidence.strict_replay:
        raise RuntimeError(f"front-duel probe failed strict replay: {output_dir}")
    return str(output_dir / "g1-free-kick.json")


def train_g1_front_duel_actor(
    *,
    assets: G1FrontDuelTrainingAssets,
    output_dir: Path,
    source_checkout: Path,
    curriculum: G1FrontDuelCurriculum | None = None,
) -> G1FrontDuelTrainingReport:
    """Run eight strict probes, distil an actor, and replay it teacher-free."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("front-duel training evidence must remain outside the checkout")
    root.mkdir(parents=True, exist_ok=False)
    active = curriculum or G1FrontDuelCurriculum()
    request = json.loads(assets.seed_request.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("front-duel seed request must be a JSON object")
    shared = {
        "assets": {key: str(value) for key, value in asdict(assets).items()},
        "request": request,
        "source_checkout": str(checkout),
        "target_y_m": active.target_y_m,
        "target_z_m": active.target_z_m,
        "ball_mass_kg": active.ball_mass_kg,
    }
    jobs = []
    for index, force_pair in enumerate(active.teacher_force_pairs_n):
        jobs.append(
            {
                **shared,
                "force_pair_n": force_pair,
                "output_dir": str(
                    root / f"probe-{index:02d}-l{force_pair[0]:.0f}-v{force_pair[1]:.0f}"
                ),
            }
        )
    if active.worker_count == 1:
        evidence_paths = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=active.worker_count) as pool:
            evidence_paths = list(pool.map(_run_probe, jobs))

    actor_path = root / "front-duel-contact-actor.json"
    actor = derive_g1_ballistic_contact_impulse_actor(
        evidence_paths=tuple(Path(path) for path in evidence_paths),
        output_path=actor_path,
        source_checkout=checkout,
    )
    base_flow = _config_from_json(G1FreeKickFlowConfig, dict(request["flow_config"]))
    final_flow = replace(
        base_flow,
        ballistic_contact_impulse_actor_hash=actor.actor_hash,
        torque_authority_projection_ratio=0.99,
        contact_task_direction_projection_enabled=False,
        shot_loft_teacher_target_vy_mps=0.0,
        shot_loft_teacher_target_vz_mps=0.0,
        shot_loft_teacher_max_foot_ball_distance_m=0.0,
    )
    goal = replace(
        _config_from_json(G1TrainingGoalSpec, dict(request["goal_spec"])),
        target_y_m=active.target_y_m,
        target_z_m=active.target_z_m,
        ball_mass_kg=active.ball_mass_kg,
    )
    final_dir = root / "final-teacher-free"
    final = run_g1_free_kick_showcase(
        asset_root=assets.asset_root,
        gait_policy_root=assets.gait_policy_root,
        output_dir=final_dir,
        source_checkout=checkout,
        runup_config=_config_from_json(G1LearnedRunupConfig, dict(request["runup_config"])),
        flow_config=final_flow,
        goal_spec=goal,
        sonic_model_root=assets.sonic_model_root,
        sonic_runup_config=_config_from_json(
            G1SonicRunupConfig,
            dict(request["sonic_runup_config"]),
        ),
        approach_strike_candidate_path=assets.approach_strike_candidate,
        approach_strike_residual_config=_config_from_json(
            G1ApproachStrikeResidualConfig,
            dict(request["approach_strike_residual_config"]),
        ),
        football_motion_prior=load_g1_football_motion_prior(assets.football_motion_prior),
        ballistic_contact_impulse_actor=actor,
        front_duel_config=G1FrontDuelConfig(),
    )
    duel = final.front_duel_summary
    report = G1FrontDuelTrainingReport(
        actor_path=str(actor_path),
        actor_hash=actor.actor_hash,
        final_evidence_path=str(final_dir / "g1-free-kick.json"),
        strict_replay=final.strict_replay,
        front_contact_observed=final.result.kick_contact_observed,
        front_ball_apex_height_m=final.result.ball_apex_height_m,
        front_goal_crossing_xyz_m=final.result.goal_crossing_xyz_m,
        front_goal_plane_target_error_m=final.result.goal_plane_target_error_m,
        front_post_kick_fall=final.result.post_kick_fall,
        goalkeeper_minimum_pelvis_height_m=(
            None if duel is None else duel.goalkeeper_minimum_pelvis_height_m
        ),
        goalkeeper_peak_tilt_rad=None if duel is None else duel.goalkeeper_peak_tilt_rad,
        goalkeeper_reaction_frames=0 if duel is None else duel.goalkeeper_reaction_frames,
        final_evidence_passed=final.passed,
    )
    (root / "front-duel-training.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--gait-policy-root", type=Path, required=True)
    parser.add_argument("--sonic-model-root", type=Path, required=True)
    parser.add_argument("--seed-request", type=Path, required=True)
    parser.add_argument("--approach-strike-candidate", type=Path, required=True)
    parser.add_argument("--football-motion-prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    report = train_g1_front_duel_actor(
        assets=G1FrontDuelTrainingAssets(
            asset_root=args.asset_root,
            gait_policy_root=args.gait_policy_root,
            sonic_model_root=args.sonic_model_root,
            seed_request=args.seed_request,
            approach_strike_candidate=args.approach_strike_candidate,
            football_motion_prior=args.football_motion_prior,
        ),
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
        curriculum=G1FrontDuelCurriculum(worker_count=args.workers),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "G1FrontDuelCurriculum",
    "G1FrontDuelTrainingAssets",
    "G1FrontDuelTrainingReport",
    "train_g1_front_duel_actor",
]
