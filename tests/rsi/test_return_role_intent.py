import subprocess
import sys

import numpy as np
import pytest

from rosclaw_soccer.rsi.return_role_intent import return_receive_command, return_strike_seen


def test_signal_requires_simultaneous_foot_force_and_return_velocity():
    forces = np.zeros((3, 2, 3))
    velocity = np.zeros((3, 6))
    forces[0, 1, 0] = 11
    velocity[1, 2] = 2
    direction = np.array([1.0, 0.0])
    assert not return_strike_seen(forces, velocity, ball_v=2, return_direction=direction)
    velocity[0, 2] = 1
    assert return_strike_seen(forces, velocity, ball_v=2, return_direction=direction)
    velocity[0, 2] = -2
    assert not return_strike_seen(forces, velocity, ball_v=2, return_direction=direction)


def test_command_is_causal_bounded_and_no_return_means_zero():
    q, v = np.zeros(43), np.zeros(41)
    q[3] = 1
    q[36] = -1
    v[35] = 2
    direction = np.array([1.0, 0.0])
    assert np.array_equal(
        return_receive_command(q, v, return_direction=direction, launch_seen=False), np.zeros(3)
    )
    command = return_receive_command(q, v, return_direction=direction, launch_seen=True)
    assert command.shape == (3,) and command.dtype == np.float32
    assert np.max(np.abs(command)) <= 1
    assert np.any(command != 0)


def test_invalid_signal_is_rejected():
    with pytest.raises(ValueError):
        return_strike_seen(
            np.zeros((2, 2, 3)), np.zeros((1, 6)), ball_v=2, return_direction=np.array([1.0, 0.0])
        )


def test_return_pair_rejects_missing_coordinated_protocol(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--frontier-root",
            str(tmp_path),
            "--frontier-head",
            "x",
            "--frontier-hash",
            "x",
            "--expected-head",
            "x",
            "--output",
            str(tmp_path / "output"),
            "--new-context-block",
            "0",
            "--passer-return-pair",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "return pair requires the isolated coordinated receiver protocol" in result.stderr
    assert not (tmp_path / "output").exists()
