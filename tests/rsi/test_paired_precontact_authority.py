import numpy as np
import pytest

from rosclaw_soccer.rsi.paired_precontact_authority import paired_precontact_authority


def sample():
    command = np.zeros((5, 3))
    baseline_command = command.copy()
    command[1:, 0] = 0.2
    qpos = np.zeros((5, 10, 8))
    baseline_qpos = qpos.copy()
    qpos[2, -1, 3] = 0.3
    qpos[2, -1, :2] = [0.3, 0.4]
    forces = np.zeros((5, 10, 2, 3))
    forces[3, 1, 1, 0] = 11
    return [command, baseline_command, qpos, baseline_qpos, forces]


def evaluate(arrays):
    return paired_precontact_authority(*arrays, joint_indices=np.array([3, 4]), root_index=0)


def test_precontact_is_time_aligned_and_not_fixed_state_causal():
    result = evaluate(sample())
    assert result["first_command_change_control"] == 1
    assert result["first_receiver_contact_substep"] == 31
    assert result["prior_control"] == 2
    assert result["precontact_max_joint_delta_rad"] == pytest.approx(0.3)
    assert result["precontact_root_xy_delta_m"] == pytest.approx(0.5)
    assert result["fixed_state_causal_effect"] is False


def test_short_baseline_never_compares_against_its_terminal_state():
    arrays = sample()
    arrays[3] = arrays[3][:2]
    result = evaluate(arrays)
    assert result["prior_control_time_aligned"] is False
    assert result["precontact_max_joint_delta_rad"] is None


def test_no_contact_and_nonfoot_contact_are_distinct():
    arrays = sample()
    arrays[4][:] = 0
    assert evaluate(arrays)["first_receiver_contact_substep"] is None
    arrays[4][2, 4, 1, 2] = 1.01
    assert evaluate(arrays)["first_receiver_contact_substep"] == 24


def test_nonfinite_physics_rejected():
    arrays = sample()
    arrays[2][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate(arrays)
