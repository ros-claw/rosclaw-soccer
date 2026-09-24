"""SIM_ONLY probe of the existing learned free-kick with a SONIC full-body run-up.

This is a reference-capability bridge test, not a new trained RSI candidate.
It uses the existing strict physical replay path and disables loft teachers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.skills.shoot.free_kick import (
    G1FreeKickFlowConfig,
    run_g1_free_kick_showcase,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--gait-policy-root", required=True, type=Path)
    parser.add_argument("--sonic-model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--kick-phase-start-frame", type=int, default=150)
    parser.add_argument("--bridge-duration-sec", type=float, default=0.60)
    parser.add_argument("--run-velocity-mps", type=float, default=1.50)
    args = parser.parse_args()
    source_hash = hash_bytes(Path(__file__).read_bytes())
    flow = G1FreeKickFlowConfig(
        approach_provider="sonic_fullbody",
        kick_phase_start_frame=args.kick_phase_start_frame,
        bridge_duration_sec=args.bridge_duration_sec,
    )
    sonic = G1SonicRunupConfig(model_variant="low_latency", run_velocity_mps=args.run_velocity_mps)
    goal = G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    result = run_g1_free_kick_showcase(
        asset_root=args.asset_root,
        gait_policy_root=args.gait_policy_root,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
        flow_config=flow,
        goal_spec=goal,
        sonic_model_root=args.sonic_model_root,
        sonic_runup_config=sonic,
    )
    if hash_bytes(Path(__file__).read_bytes()) != source_hash:
        raise RuntimeError("free-kick bridge probe source changed during execution")
    summary = {
        "schema": "rosclaw_soccer.rsi.sonic_free_kick_reuse_probe.v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "source_hash": source_hash,
        "passed": result.passed,
        "strict_replay": result.strict_replay,
        "kick_contact_observed": result.result.kick_contact_observed,
        "goal_crossed": result.result.goal_crossed,
        "ball_speed_peak_mps": result.result.ball_speed_peak_mps,
        "ball_apex_height_m": result.result.ball_apex_height_m,
        "runup_peak_speed_mps": result.result.runup_peak_speed_mps,
        "handoff_to_contact_sec": result.result.handoff_to_contact_sec,
        "perceptual_continuity_passed": result.result.perceptual_continuity_passed,
        "runup_min_pelvis_height_m": result.result.runup_min_pelvis_height_m,
        "runup_peak_tilt_rad": result.result.runup_peak_tilt_rad,
        "kick_min_pelvis_height_m": result.result.kick_min_pelvis_height_m,
        "kick_peak_tilt_rad": result.result.kick_peak_tilt_rad,
        "goal_plane_target_error_m": result.result.goal_plane_target_error_m,
        "declared_corner_distance_m": result.result.declared_corner_distance_m,
        "loft_teacher_executed": result.result.loft_teacher_executed,
        "trajectory_path": result.trajectory_path,
        "trajectory_hash": result.trajectory_hash,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
