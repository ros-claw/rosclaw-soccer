"""Causal, content-bound measured-foot navigation stays SIM_ONLY."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.team_foot_velocity_chooser import TeamFootVelocityChooser
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationObservation, NavigationSlot


def _observation() -> NavigationObservation:
    return NavigationObservation(
        agent_id="red.playmaker",
        frame=30,
        time_sec=0.6,
        role="playmaker",
        intent="pass",
        body_pose=(0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0),
        body_velocity=(0.0, 0.0, 0.0),
        ball_position=(1.0, 0.0, 0.115),
        ball_velocity=(-0.4, 0.0, 0.0),
        task_target=(2.0, 0.0, 0.0),
        steering_target=(2.0, 0.0),
        baseline_command=(0.0, 0.0, 0.0),
        previous_command=(0.0, 0.0, 0.0),
        neighbors=(),
        effector_positions=(
            ("left_foot", 0.1, 0.1, 0.1),
            ("right_foot", 0.1, -0.1, 0.1),
        ),
        effector_velocities=(
            ("left_foot", 0.0, 0.0, 0.0),
            ("right_foot", 0.0, 0.0, 0.0),
        ),
    )


def _model(path: Path) -> None:
    names = [
        "adaptive_c19",
        "fixed_left_negative",
        "phase_front04_lat12",
        "phase_front04_negative",
        "phase_front04_positive",
        "phase_stay_lat12",
    ]
    layers = [
        {"weight": [[0.0] * 12 for _ in range(24)], "bias": [0.0] * 24},
        {"weight": [[0.0] * 24 for _ in range(24)], "bias": [0.0] * 24},
        {
            "weight": [[0.0] * 24 for _ in range(18)],
            "bias": [3.0, 0.0, 3.0] + [-3.0, 0.0, 0.0] * 5,
        },
    ]
    model = {
        "schema": "rsi_team_foot_velocity_chooser_model_v33",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "feature_count": 12,
        "safety_threshold": 0.65,
        "safety_quantile": 0.2,
        "arm_names": names,
        "arm_parameters": {name: {"foot_selection": "adaptive"} for name in names},
        "adaptive_model_parameters": {
            "forward_gain": 0.8,
            "lateral_gain": -0.8,
            "target_gap_m": 0.5,
            "activation_max_gap_m": 2.0,
            "speed_cap_mps": 0.22,
        },
        "mean": [0.0] * 12,
        "scale": [1.0] * 12,
        "networks": [{"layers": layers} for _ in range(3)],
    }
    model["model_hash"] = hash_json(model)
    path.write_text(json.dumps(model), encoding="utf-8")


def _chooser(path: Path) -> TeamFootVelocityChooser:
    return TeamFootVelocityChooser(
        agent_id="red.playmaker",
        foundation_hash="sha256:" + "a" * 64,
        foundation_config_hash="sha256:" + "b" * 64,
        model_path=path,
    )


def test_chooser_uses_measured_foot_velocity_and_bounded_navigation(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    _model(path)
    chooser = _chooser(path)
    slot = NavigationSlot(chooser)
    slot._frame = 29
    slot._time = 0.58
    delta = slot.propose(_observation())
    assert not slot.faulted
    assert chooser.selected_arm == "adaptive_c19"
    assert abs(delta[0]) <= 0.25 and abs(delta[1]) <= 0.25


def test_missing_velocity_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    _model(path)
    chooser = _chooser(path)
    slot = NavigationSlot(chooser)
    slot._frame = 29
    slot._time = 0.58
    assert slot.propose(replace(_observation(), effector_velocities=())) == (0.0, 0.0, 0.0)
    assert slot.faulted
    assert chooser.selected_arm is None


def test_tampered_model_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    _model(path)
    model = json.loads(path.read_text())
    model["mean"][0] = 99.0
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed"):
        _chooser(path)


def test_velocity_ids_must_match_measured_position_ids() -> None:
    with pytest.raises(ValueError, match="end-effectors"):
        replace(_observation(), effector_velocities=(("left_foot", 0.0, 0.0, 0.0),))
