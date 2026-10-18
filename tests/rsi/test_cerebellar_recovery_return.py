import numpy as np
import pytest

from rosclaw_soccer.rsi.cerebellar_recovery_return import reward_contract, training_return
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch


def trace(frame=85):
    force = np.zeros((300, 1, 6))
    ball = np.zeros((300, 1, 3))
    if frame is not None:
        force[frame, 0, 1] = 2
        ball[frame + 1 :, 0, 0] = np.arange(1, 300 - frame) / 50
    return dict(
        force_n=force,
        ball_position_after_step_m=ball,
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        cerebellar_canonical_qvel_before_substep=np.zeros((300, 1, 10, 41)),
        applied_cerebellar_torque_offset_nm=np.zeros((300, 1, 10, 29)),
        cerebellar_before_integration_substep_index=np.arange(3000).reshape(300, 1, 10, 1),
    )


def test_secondary_learning_return_does_not_change_formal_label():
    raw = trace()
    outcome = label_first_touch(raw).to_dict()
    result = training_return(raw)
    assert result["formal_outcome"] == outcome
    assert result["formal_training_reward"] == pytest.approx(3.2)
    assert result["training_return"] == pytest.approx(8.2)
    raw["cerebellar_canonical_qvel_before_substep"][86:146, 0, :, 3] = 2
    changed = training_return(raw)
    assert changed["formal_outcome"] == outcome
    assert changed["applied_recovery_penalty"] == 1
    assert changed["training_return"] == pytest.approx(7.2)


def test_correct_60_whole_frame_window_not_future_outside_task():
    raw = trace(239)
    raw["cerebellar_canonical_qvel_before_substep"][:240, 0, :, 3] = 100
    result = training_return(raw)
    assert result["applied_recovery_penalty"] == 0
    assert result["recovery_cost_diagnostics"]["actual_substep_samples"] == 600


@pytest.mark.parametrize("first", [None, 240, 260, 299])
def test_missing_recovery_diagnostics_are_unknown_not_zero_stability(first):
    raw = trace(first)
    result = training_return(raw)
    assert result["recovery_cost_diagnostics"] is None
    assert result["pass_bonus"] == 0
    assert result["training_return"] == result["formal_training_reward"]
    assert not result["formal_outcome"]["task_pass"]


def test_recovery_cost_is_bounded_and_does_not_override_safety_failure():
    raw = trace()
    raw["cerebellar_canonical_qvel_before_substep"][:, :, :, 3:35] = 100
    assert training_return(raw)["applied_recovery_penalty"] == 2
    raw["pelvis_z_per_substep_m"][299, 0, 9] = 0.1
    result = training_return(raw)
    assert result["formal_training_reward"] == -10
    assert result["training_return"] == -12
    assert result["pass_bonus"] == 0
    assert not result["formal_outcome"]["safe"]


def test_bad_capture_or_clock_is_not_a_legal_missing_window():
    raw = trace(260)
    raw["cerebellar_before_integration_substep_index"][5, 0, 0, 0] = 0
    with pytest.raises(ValueError):
        training_return(raw)
    raw = trace()
    raw["cerebellar_canonical_qvel_before_substep"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        training_return(raw)


def test_reward_contract_is_fresh_and_cannot_rewrite_formal_m0():
    contract = reward_contract()
    contract["pass_bonus"] = 999
    assert reward_contract()["pass_bonus"] == 5
    assert not reward_contract()["formal_task_or_safety_label_changed"]
