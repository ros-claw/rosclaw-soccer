"""Execute the runner's context expression with only selected sensor fields.

This supplements native full-episode validation; it does not certify football
quality or dynamics. In particular, the current branch has no ball_vel local.
"""

import ast
from pathlib import Path

import numpy as np
import pytest


def runner_tree():
    source = Path(__file__).resolve().parents[2] / "scripts/rsi_mujoco_motor_transfer.py"
    return ast.parse(source.read_text())


@pytest.mark.parametrize("snapshot", ["cached", "current-kinematic"])
def test_context_consumes_selected_ball_position_and_linear_velocity(snapshot):
    calls = [
        node
        for node in ast.walk(runner_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "current_context"
    ]
    assert len(calls) == 1
    positions = np.arange(9, dtype=float).reshape(3, 3)
    velocity = np.array([0.2, 0.4, 0.6])
    namespace = {
        "current_context": lambda *values: values,
        "root_pose": np.zeros(7),
        "root_vel": np.zeros(6),
        "positions": positions,
        "ball": 2,
        "ball_linear_vel": velocity,
    }
    if snapshot == "cached":
        # Poison cached-only state: the selected fields are the sole inputs.
        namespace["ball_vel"] = np.full(6, np.nan)
    values = eval(compile(ast.Expression(calls[0]), "runner-context", "eval"), namespace)
    np.testing.assert_array_equal(values[2], positions[2][None])
    np.testing.assert_array_equal(values[3], velocity[None])


def test_swing_geometry_consumes_selected_snapshot_not_live_cached_positions():
    calls = [
        node
        for node in ast.walk(runner_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"choose_swing_side", "swing_joint_delta"}
    ]
    assert len(calls) == 2
    for call in calls:
        for argument in call.args[:2]:
            assert isinstance(argument, ast.Subscript)
            assert isinstance(argument.value, ast.Name)
            assert argument.value.id == "positions"
