from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.training.continuous_dm_control_probe import (
    validate_continuous_dm_control_probe,
)


def test_continuous_probe_validator_is_content_and_implementation_bound(tmp_path: Path) -> None:
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
    from rosclaw_soccer.training import continuous_dm_control_probe as module

    report = {
        "schema_version": "rosclaw_soccer.continuous_dm_control_probe.v1",
        "status": "PASS_ENVIRONMENT_CONTRACT",
        "claim": "OFFICIAL_DM_CONTROL_CONTINUOUS_MATCH_ENVIRONMENT_SMOKE",
        "config": {
            "team_size": 2,
            "duration_sec": 60.0,
            "terminate_on_goal": False,
            "zero_action_policy": True,
            "goal_state_injected": True,
            "walker_type": "BOXHEAD",
        },
        "gates": {
            "physics_advanced": True,
            "four_agents_present": True,
            "goal_reward_observed": True,
            "goal_step_did_not_terminate": True,
            "goal_detected_before_restart": True,
            "restart_stayed_in_episode": True,
            "goal_detector_cleared": True,
            "ball_reinitialized_after_goal": True,
            "time_limit_terminated_match": True,
        },
        "telemetry": {"physics_time_sec": 60.0, "physics_control_steps": 2400},
        "provenance": {
            "implementation_hash": hash_bytes(Path(module.__file__).read_bytes()),
            "dm_control_license": "Apache-2.0",
        },
        "evidence_ceiling": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "environment_contract_only": True,
            "g1_policy_executed": False,
            "agent_skill_claimed": False,
            "promotion_eligible": False,
        },
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_continuous_dm_control_probe(path)["status"] == "PASS_ENVIRONMENT_CONTRACT"

    report["gates"]["physics_advanced"] = False
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence is invalid"):
        validate_continuous_dm_control_probe(path)


def test_continuous_probe_validator_rejects_a_missing_gate(tmp_path: Path) -> None:
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
    from rosclaw_soccer.training import continuous_dm_control_probe as module

    report = {
        "schema_version": "rosclaw_soccer.continuous_dm_control_probe.v1",
        "status": "PASS_ENVIRONMENT_CONTRACT",
        "claim": "OFFICIAL_DM_CONTROL_CONTINUOUS_MATCH_ENVIRONMENT_SMOKE",
        "config": {
            "team_size": 2,
            "duration_sec": 60.0,
            "terminate_on_goal": False,
            "zero_action_policy": True,
            "goal_state_injected": True,
            "walker_type": "BOXHEAD",
        },
        "gates": {"physics_advanced": True},
        "telemetry": {"physics_time_sec": 60.0, "physics_control_steps": 2400},
        "provenance": {
            "implementation_hash": hash_bytes(Path(module.__file__).read_bytes()),
            "dm_control_license": "Apache-2.0",
        },
        "evidence_ceiling": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "environment_contract_only": True,
            "g1_policy_executed": False,
            "agent_skill_claimed": False,
            "promotion_eligible": False,
        },
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "missing-gate.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence is invalid"):
        validate_continuous_dm_control_probe(path)
