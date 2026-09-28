"""Replay the archived R1 pass/shot parent on the *current* eight-G1 implementation.

This is a consumed development diagnostic, never a Fresh or promotion exam.
"""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
    default_continuous_match_config,
)
from rosclaw_soccer.training.pass_contact_chain import inspect_pass_contact_chain
from rosclaw_soccer.training.phase_conditioned_strike_growth import (
    default_phase_strike_controller,
    default_phase_strike_option,
    default_phase_strike_teacher,
)


class _Recorder:
    def __init__(self) -> None:
        self.agent_id = "red.playmaker"
        self.contract_hash = hash_json(
            {"consumer": "r1_current_parent_replay", "agent_id": self.agent_id}
        )
        self.rows: list[TeamMotorPhysicsObservation] = []

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        self.rows.append(observation)


def _source_hashes(root: Path) -> dict[str, str]:
    names = (
        "scripts/rsi_r1_current_parent_replay.py",
        "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "src/rosclaw_soccer/training/continuous_competitive_match_growth.py",
        "src/rosclaw_soccer/training/phase_conditioned_strike_growth.py",
        "src/rosclaw_soccer/training/pass_contact_chain.py",
    )
    return {name: hash_bytes((root / name).read_bytes()) for name in names}


def run(
    asset_root: Path,
    output_dir: Path,
    *,
    ball_x_m: float = 1.92,
    ball_y_m: float = -0.80,
    stance_lateral_m: float = -0.17,
    seed: int = 207_200,
    pass_speed_mps: float = 0.80,
    pass_stroke_sec: float = 0.0,
    preferred_foot: str = "nearest",
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external evidence directory required")
    torch.set_num_threads(1)
    sources = _source_hashes(root)
    fixture = build_continuous_competitive_fixture(asset_root)
    scenario = IndependentTeamWorldScenario(
        scenario_id="s199.rsi.r1.current-parent.development",
        ball_initial_position_m=(ball_x_m, ball_y_m, 0.115),
        ball_initial_velocity_mps=(0.0, 0.0, 0.0),
        seed=seed,
    )
    world = replace(default_continuous_match_config(), simulation_duration_sec=8.70)
    phase = replace(
        default_phase_strike_controller(),
        strike_aim_lateral_bias_m=0.65,
        target_stance_lateral_m=stance_lateral_m,
    )
    teacher = replace(
        default_phase_strike_teacher(),
        pass_strike_foot_speed_mps=pass_speed_mps,
        pass_stroke_duration_sec=pass_stroke_sec,
        preferred_foot=preferred_foot,
    )
    protocol = {
        "schema": "rosclaw_soccer.rsi.r1_current_parent_replay_protocol.v1",
        "authority": "SIM_ONLY",
        "partition": "CONSUMED_DEV",
        "source_hashes": sources,
        "scenario": asdict(scenario),
        "world_config_hash": world.config_hash,
        "strike_phase_hash": hash_json(asdict(phase)),
        "contact_teacher_hash": teacher.config_hash,
        "asset_root": str(asset_root.resolve()),
        "hypothesis": (
            "Declared R1 configuration gives clean foot-to-foot transfer "
            "and an in-frame shot under current source."
        ),
        "acceptance": (
            "Six bodies safe; clean 500 Hz sender-to-receiver transfer; "
            "physical in-frame goal-plane crossing; no synthetic ball injection."
        ),
        "training_authorized": False,
        "promotion_authorized": False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    recorder = _Recorder()
    with (
        (output_dir / "simulation.log").open("x", encoding="utf-8") as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        result, trace = simulate_independent_team_world(
            asset_root=asset_root,
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=scenario,
            goal=fixture.goal,
            config=world,
            contact_teacher_config=teacher,
            option_bridge_config=default_phase_strike_option(),
            strike_phase_config=phase,
            physics_evidence_consumers={"red.playmaker": recorder},
        )
    if _source_hashes(root) != sources:
        raise RuntimeError("source drift during physical run; result must not be accepted")
    trace_path = output_dir / "current-parent.npz"
    np.savez_compressed(trace_path, **{key: value for key, value in trace.items()})  # type: ignore[arg-type]
    ids = tuple(sorted(agent.agent_id for agent in fixture.roster.agents))
    request_frames = np.flatnonzero(
        (trace["pass_source_agent_code"] > 0) & (trace["pass_target_agent_code"] > 0)
    )
    chain: dict[str, Any] | None = None
    receiver_id: str | None = None
    request_sec: float | None = None
    if len(request_frames) and recorder.rows:
        first = int(request_frames[0])
        receiver_id = ids[int(trace["pass_target_agent_code"][first]) - 1]
        request_sec = float(trace["time"][first])
        chain = asdict(
            inspect_pass_contact_chain(
                tuple(recorder.rows),
                sender_id="red.playmaker",
                receiver_id=receiver_id,
                request_time_sec=request_sec,
            )
        )
    ball = trace["ball_pose"][:, :3]
    speeds = np.linalg.norm(trace["ball_velocity"][:, :3], axis=1)
    crossing: dict[str, Any] | None = None
    for frame in range(len(ball) - 1):
        if ball[frame, 0] < fixture.goal.plane_x_m <= ball[frame + 1, 0]:
            fraction = (fixture.goal.plane_x_m - ball[frame, 0]) / (
                ball[frame + 1, 0] - ball[frame, 0]
            )
            y = float(ball[frame, 1] + fraction * (ball[frame + 1, 1] - ball[frame, 1]))
            z = float(ball[frame, 2] + fraction * (ball[frame + 1, 2] - ball[frame, 2]))
            crossing = {
                "frame": frame,
                "time_sec": float(trace["time"][frame]),
                "y_m": y,
                "z_m": z,
                "inside_geometry": bool(
                    abs(y - fixture.goal.target_y_m)
                    <= fixture.goal.width_m / 2 - fixture.goal.ball_radius_m
                    and fixture.goal.ball_radius_m
                    <= z
                    <= fixture.goal.height_m - fixture.goal.ball_radius_m
                ),
            }
            break
    if result.player_count != 6 or result.red_player_count != 3 or result.blue_player_count != 3:
        raise RuntimeError("R1 parent fixture must be the audited 3v3 roster")
    chain_succeeded = bool(
        result.safe
        and chain is not None
        and chain["clean_transfer_observed"]
        and crossing is not None
        and crossing["inside_geometry"]
        and chain["receiver_contact_sec"] is not None
        and chain["receiver_contact_sec"] < crossing["time_sec"]
        and float(speeds.max()) >= 3.0
    )
    parent_parameters = (
        ball_x_m == 1.92
        and ball_y_m == -0.80
        and stance_lateral_m == -0.17
        and seed == 207_200
        and pass_speed_mps == 0.80
        and pass_stroke_sec == 0.0
        and preferred_foot == "nearest"
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_current_parent_replay.v1",
        "protocol_hash": hash_bytes((output_dir / "protocol.json").read_bytes()),
        "trace_hash": hash_bytes(trace_path.read_bytes()),
        "source_stable_during_run": True,
        "result": result.to_dict(),
        "request_time_sec": request_sec,
        "receiver_id": receiver_id,
        "chain": chain,
        "max_ball_speed_mps": float(speeds.max()),
        "goal_plane_crossing": crossing,
        "goal_geometry": asdict(fixture.goal),
        "pass_shot_chain_succeeded": chain_succeeded,
        "archived_parent_reproduced": bool(parent_parameters and chain_succeeded),
        "training_authorized": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ball-x", type=float, default=1.92)
    parser.add_argument("--ball-y", type=float, default=-0.80)
    parser.add_argument("--stance", type=float, default=-0.17)
    parser.add_argument("--seed", type=int, default=207_200)
    parser.add_argument("--pass-speed", type=float, default=0.80)
    parser.add_argument("--pass-stroke", type=float, default=0.0)
    parser.add_argument("--preferred-foot", choices=("nearest", "left", "right"), default="nearest")
    args = parser.parse_args()
    report = run(
        args.asset_root,
        args.output_dir,
        ball_x_m=args.ball_x,
        ball_y_m=args.ball_y,
        stance_lateral_m=args.stance,
        seed=args.seed,
        pass_speed_mps=args.pass_speed,
        pass_stroke_sec=args.pass_stroke,
        preferred_foot=args.preferred_foot,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "pass_shot_chain_succeeded",
                    "max_ball_speed_mps",
                    "goal_plane_crossing",
                    "report_hash",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
