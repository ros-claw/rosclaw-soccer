"""Official dm_control continuous-soccer environment contract probe.

This is deliberately an environment smoke test, not a football-skill exam.
It injects a ball into a goal detector, verifies that the same MuJoCo episode
continues after an in-match restart, and advances physics to the time limit.
No result from this module is eligible to promote a G1 policy.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_REQUIRED_GATES = frozenset(
    {
        "physics_advanced",
        "four_agents_present",
        "goal_reward_observed",
        "goal_step_did_not_terminate",
        "goal_detected_before_restart",
        "restart_stayed_in_episode",
        "goal_detector_cleared",
        "ball_reinitialized_after_goal",
        "time_limit_terminated_match",
    }
)


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_continuous_dm_control_probe(
    *,
    output_dir: Path,
    source_checkout: Path,
    dm_control_checkout: Path,
    seed: int = 118,
    duration_sec: float = 60.0,
) -> dict[str, Any]:
    """Advance an official 2v2 BoxHead match across a forced goal."""

    destination = output_dir.expanduser().resolve()
    source = source_checkout.expanduser().resolve()
    upstream = dm_control_checkout.expanduser().resolve()
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError("continuous-soccer probe output must be new and outside the checkout")
    if not source.is_dir() or not upstream.is_dir():
        raise ValueError("continuous-soccer source checkouts are unavailable")
    if not 30.0 <= duration_sec <= 180.0 or not 0 <= seed <= 2**32 - 1:
        raise ValueError("continuous-soccer probe config is invalid")

    try:
        from dm_control.locomotion import soccer  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised on optional hosts
        raise RuntimeError("official dm_control is required for this optional probe") from exc

    env = soccer.load(
        team_size=2,
        time_limit=duration_sec,
        random_state=seed,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=soccer.WalkerType.BOXHEAD,
    )
    timestep = env.reset()
    actions = [np.zeros(spec.shape, dtype=spec.dtype) for spec in env.action_spec()]
    # dm_control exposes MuJoCo-backed views here; copy every evidentiary
    # snapshot so later physics steps cannot rewrite historical telemetry.
    start_ball = np.asarray(env.task.ball.get_pose(env.physics)[0], dtype=np.float64).copy()

    injected_goal = np.asarray(env.task.arena.away_goal.mid, dtype=np.float64).copy()
    injected_goal[2] = float(start_ball[2])
    env.task.ball.set_pose(env.physics, injected_goal)
    env.task.ball.set_velocity(
        env.physics,
        velocity=np.zeros(3, dtype=np.float64),
        angular_velocity=np.zeros(3, dtype=np.float64),
    )
    goal_timestep = env.step(actions)
    goal_reward = tuple(float(value) for value in goal_timestep.reward)
    post_goal_ball = np.asarray(env.task.ball.get_pose(env.physics)[0], dtype=np.float64).copy()
    goal_detected = env.task.arena.detected_goal()

    restart_timestep = env.step(actions)
    restart_ball = np.asarray(env.task.ball.get_pose(env.physics)[0], dtype=np.float64).copy()
    restart_goal_detected = env.task.arena.detected_goal()
    steps = 2
    timestep = restart_timestep
    while not timestep.last():
        timestep = env.step(actions)
        steps += 1

    gates = {
        "physics_advanced": bool(env.physics.data.time >= duration_sec),
        "four_agents_present": len(env.task.players) == 4,
        "goal_reward_observed": goal_reward == (1.0, 1.0, -1.0, -1.0),
        "goal_step_did_not_terminate": not goal_timestep.last(),
        "goal_detected_before_restart": goal_detected is not None,
        "restart_stayed_in_episode": not restart_timestep.last(),
        "goal_detector_cleared": restart_goal_detected is None,
        "ball_reinitialized_after_goal": bool(
            np.linalg.norm(post_goal_ball - injected_goal) > 1.0
            and np.linalg.norm(restart_ball - injected_goal) > 1.0
        ),
        "time_limit_terminated_match": timestep.last(),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_dm_control_probe.v1",
        "status": "PASS_ENVIRONMENT_CONTRACT" if all(gates.values()) else "FAIL",
        "claim": "OFFICIAL_DM_CONTROL_CONTINUOUS_MATCH_ENVIRONMENT_SMOKE",
        "config": {
            "team_size": 2,
            "duration_sec": duration_sec,
            "seed": seed,
            "terminate_on_goal": False,
            "walker_type": "BOXHEAD",
            "zero_action_policy": True,
            "goal_state_injected": True,
        },
        "gates": gates,
        "telemetry": {
            "physics_time_sec": float(env.physics.data.time),
            "physics_control_steps": steps,
            "goal_step_time_sec": float(goal_timestep.observation[0].get("time", [0.0])[0])
            if "time" in goal_timestep.observation[0]
            else None,
            "goal_reward": list(goal_reward),
            "start_ball_position_m": start_ball.tolist(),
            "injected_goal_position_m": injected_goal.tolist(),
            "post_goal_ball_position_m": post_goal_ball.tolist(),
            "restart_ball_position_m": restart_ball.tolist(),
        },
        "provenance": {
            "source_commit": _git_head(source),
            "dm_control_commit": _git_head(upstream),
            "dm_control_checkout": str(upstream),
            "dm_control_license": "Apache-2.0",
            "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("dm-control", "dm-env", "mujoco", "numpy")
            },
        },
        "evidence_ceiling": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "environment_contract_only": True,
            "g1_policy_executed": False,
            "agent_skill_claimed": False,
            "promotion_eligible": False,
            "statement": (
                "This forced-goal zero-action probe validates only continuous MuJoCo match "
                "semantics. It is not evidence that any G1 can play football."
            ),
        },
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    destination.mkdir(parents=True)
    _atomic_json(destination / "probe-report.json", report)
    return report


def validate_continuous_dm_control_probe(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("continuous-soccer probe report must be a JSON object")
    report_hash = payload.pop("report_hash", None)
    try:
        provenance = payload.get("provenance")
        config = payload.get("config")
        gates = payload.get("gates")
        telemetry = payload.get("telemetry")
        evidence_ceiling = payload.get("evidence_ceiling")
        implementation_hash = (
            provenance.get("implementation_hash") if isinstance(provenance, dict) else None
        )
        duration_sec = config.get("duration_sec") if isinstance(config, dict) else None
        physics_time_sec = (
            telemetry.get("physics_time_sec") if isinstance(telemetry, dict) else None
        )
        physics_control_steps = (
            telemetry.get("physics_control_steps") if isinstance(telemetry, dict) else None
        )
        if (
            payload.get("schema_version") != "rosclaw_soccer.continuous_dm_control_probe.v1"
            or payload.get("status") != "PASS_ENVIRONMENT_CONTRACT"
            or payload.get("claim") != "OFFICIAL_DM_CONTROL_CONTINUOUS_MATCH_ENVIRONMENT_SMOKE"
            or payload.get("hardware_command_sent") is not False
            or not isinstance(config, dict)
            or config.get("team_size") != 2
            or config.get("terminate_on_goal") is not False
            or config.get("zero_action_policy") is not True
            or config.get("goal_state_injected") is not True
            or config.get("walker_type") != "BOXHEAD"
            or isinstance(duration_sec, bool)
            or not isinstance(duration_sec, (int, float))
            or not 30.0 <= float(duration_sec) <= 180.0
            or not isinstance(gates, dict)
            or set(gates) != _REQUIRED_GATES
            or not all(value is True for value in gates.values())
            or not isinstance(telemetry, dict)
            or isinstance(physics_time_sec, bool)
            or not isinstance(physics_time_sec, (int, float))
            or float(physics_time_sec) < float(duration_sec)
            or isinstance(physics_control_steps, bool)
            or not isinstance(physics_control_steps, int)
            or physics_control_steps <= 0
            or not isinstance(evidence_ceiling, dict)
            or evidence_ceiling.get("physics_authority") != "CPU_MUJOCO"
            or evidence_ceiling.get("activation_ceiling") != "SIM_ONLY"
            or evidence_ceiling.get("environment_contract_only") is not True
            or evidence_ceiling.get("g1_policy_executed") is not False
            or evidence_ceiling.get("agent_skill_claimed") is not False
            or evidence_ceiling.get("promotion_eligible") is not False
            or not isinstance(provenance, dict)
            or provenance.get("dm_control_license") != "Apache-2.0"
            or implementation_hash != hash_bytes(Path(__file__).read_bytes())
            or report_hash != hash_json(payload)
        ):
            raise ValueError("continuous-soccer probe evidence is invalid")
    finally:
        if report_hash is not None:
            payload["report_hash"] = report_hash
    return cast(dict[str, Any], payload)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--dm-control-checkout", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=118)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    args = parser.parse_args()
    report = run_continuous_dm_control_probe(
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
        dm_control_checkout=args.dm_control_checkout,
        seed=args.seed,
        duration_sec=args.duration_sec,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = ["run_continuous_dm_control_probe", "validate_continuous_dm_control_probe"]
