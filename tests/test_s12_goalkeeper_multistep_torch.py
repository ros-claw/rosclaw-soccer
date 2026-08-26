from __future__ import annotations

import pytest

from rosclaw_soccer.training.goalkeeper_multistep import (
    GoalkeeperEpisodePhase,
    GoalkeeperMultiStepConfig,
)
from rosclaw_soccer.training.goalkeeper_multistep_torch import (
    TorchGoalkeeperMultiStepAccumulator,
)


def _step(
    torch,
    *,
    time: float,
    shot: int,
    contact: bool = False,
    hand_contact: bool = False,
    save: bool = False,
    pelvis: float = 0.80,
    upright: float = 1.0,
):
    zeros3 = torch.zeros((1, 3), dtype=torch.float32)
    return {
        "time_sec": torch.tensor([time]),
        "ball_position_m": torch.tensor([[0.0, 0.0, 1.0]]),
        "ball_velocity_mps": torch.tensor([[4.0, 0.0, 0.0]]),
        "intercept_target_m": torch.tensor([[0.0, 0.0, 1.0]]),
        "left_hand_position_m": torch.tensor([[0.0, 0.05, 1.0]]),
        "right_hand_position_m": torch.tensor([[0.0, -0.05, 1.0]]),
        "pelvis_height_m": torch.tensor([pelvis]),
        "root_linear_velocity_mps": zeros3.clone(),
        "root_angular_velocity_rad_s": zeros3.clone(),
        "upright_projection": torch.tensor([upright]),
        "action": torch.zeros((1, 29)),
        "previous_action": torch.zeros((1, 29)),
        "joint_acceleration_rad_s2": torch.zeros((1, 29)),
        "applied_torque_nm": torch.zeros((1, 29)),
        "ball_contact": torch.tensor([contact], dtype=torch.bool),
        "hand_contact": torch.tensor([hand_contact], dtype=torch.bool),
        "true_save": torch.tensor([save], dtype=torch.bool),
        "shot_index": torch.tensor([shot], dtype=torch.long),
    }


def test_torch_multistep_runs_without_host_roundtrip() -> None:
    torch = pytest.importorskip("torch")
    config = GoalkeeperMultiStepConfig(recovery_hold_sec=0.04)
    task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"), config=config)

    first = task.step(_step(torch, time=0.5, shot=1, contact=True, save=True))
    assert first["phase"].device.type == "cpu"
    task.step(_step(torch, time=0.52, shot=0))
    recovered = task.step(_step(torch, time=0.54, shot=0))
    assert bool(recovered["recovered_after_first"][0])
    second = task.step(_step(torch, time=2.8, shot=2, contact=True, save=True))
    assert bool(second["second_attempt_save"][0])
    assert bool(second["second_save"][0])
    assert task.summary()["second_save_rate"] == 1.0


def test_torch_multistep_exposes_complete_reward_ledger() -> None:
    torch = pytest.importorskip("torch")
    result = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu")).step(
        _step(torch, time=0.5, shot=1)
    )
    reconstructed = (
        result["reach"]
        + result["bimanual_reach"]
        + result["task_motion"]
        + result["upright"]
        + result["recovery_progress"]
        + result["event_bonus"]
        - result["smoothness_penalty"]
        - result["effort_penalty"]
        - result["safety_penalty"]
    )

    assert torch.allclose(result["total"], reconstructed)
    assert torch.allclose(
        result["smoothness_penalty"],
        result["action_rate_penalty"]
        + result["joint_acceleration_penalty"]
        + result["root_linear_speed_penalty"]
        + result["root_angular_speed_penalty"]
        + result["root_angular_excess_penalty"]
        + result["action_magnitude_penalty"],
    )


def test_torch_multistep_charges_extra_debt_when_a_save_ends_unsafe() -> None:
    torch = pytest.importorskip("torch")
    baseline = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"))
    debt = TorchGoalkeeperMultiStepAccumulator(
        1,
        device=torch.device("cpu"),
        config=GoalkeeperMultiStepConfig(save_then_unsafe_penalty=125.0),
    )
    for task in (baseline, debt):
        task.step(_step(torch, time=0.50, shot=1, contact=True, save=True))

    baseline_failure = baseline.step(_step(torch, time=0.52, shot=0, pelvis=0.30, upright=0.2))
    debt_failure = debt.step(_step(torch, time=0.52, shot=0, pelvis=0.30, upright=0.2))

    assert float(debt_failure["safety_penalty"] - baseline_failure["safety_penalty"]) == 125.0
    assert float(debt_failure["total"] - baseline_failure["total"]) == -125.0


def test_torch_multistep_rejects_nonfinite_input() -> None:
    torch = pytest.importorskip("torch")
    task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"))
    sample = _step(torch, time=0.5, shot=1)
    sample["pelvis_height_m"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        task.step(sample)


def test_torch_multistep_tail_penalty_matches_declared_excess() -> None:
    torch = pytest.importorskip("torch")
    sample = _step(torch, time=0.5, shot=1)
    sample["root_angular_velocity_rad_s"][0, 1] = 3.0
    baseline = TorchGoalkeeperMultiStepAccumulator(
        1,
        device=torch.device("cpu"),
        config=GoalkeeperMultiStepConfig(root_angular_speed_soft_limit_rad_s=2.0),
    ).step(sample)
    tail = TorchGoalkeeperMultiStepAccumulator(
        1,
        device=torch.device("cpu"),
        config=GoalkeeperMultiStepConfig(
            root_angular_speed_soft_limit_rad_s=2.0,
            root_angular_speed_excess_penalty_scale=10.0,
        ),
    ).step(sample)

    assert float(tail["smoothness_penalty"][0] - baseline["smoothness_penalty"][0]) == (
        pytest.approx(10.0)
    )
    phase_aware_config = GoalkeeperMultiStepConfig(
        root_angular_speed_soft_limit_rad_s=2.0,
        root_angular_speed_excess_penalty_scale=10.0,
        flight_root_angular_penalty_scale=0.1,
    )
    sample["posture_exception_granted"] = torch.tensor([True])
    flight_scaled = TorchGoalkeeperMultiStepAccumulator(
        1,
        device=torch.device("cpu"),
        config=phase_aware_config,
    ).step(sample)
    recovery_sample = {key: value.clone() for key, value in sample.items()}
    recovery_sample["shot_index"][0] = 0
    recovery_sample["posture_exception_granted"][0] = False
    recovery_full = TorchGoalkeeperMultiStepAccumulator(
        1,
        device=torch.device("cpu"),
        config=phase_aware_config,
    ).step(recovery_sample)

    assert float(flight_scaled["root_angular_excess_penalty"][0]) == pytest.approx(1.0)
    assert float(recovery_full["root_angular_excess_penalty"][0]) == pytest.approx(10.0)


def test_torch_multistep_recovery_potential_rewards_braking_after_save() -> None:
    torch = pytest.importorskip("torch")
    config = GoalkeeperMultiStepConfig(recovery_progress_reward_scale=20.0)
    braking = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"), config=config)
    worsening = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"), config=config)
    impact = _step(torch, time=0.50, shot=1, contact=True, save=True)
    impact["root_angular_velocity_rad_s"][0, 1] = 2.0
    braking.step(impact)
    worsening.step({key: value.clone() for key, value in impact.items()})
    improved_sample = _step(torch, time=0.52, shot=0)
    improved_sample["root_angular_velocity_rad_s"][0, 1] = 1.0
    degraded_sample = _step(torch, time=0.52, shot=0)
    degraded_sample["root_angular_velocity_rad_s"][0, 1] = 3.0

    improved = braking.step(improved_sample)
    degraded = worsening.step(degraded_sample)

    assert float(improved["recovery_progress"][0]) > 0.0
    assert float(degraded["recovery_progress"][0]) < 0.0
    assert float(improved["total"][0]) > float(degraded["total"][0])


def test_torch_multistep_densifies_second_shot_reach() -> None:
    torch = pytest.importorskip("torch")
    config = GoalkeeperMultiStepConfig(second_shot_reach_reward_multiplier=1.6)
    first = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"), config=config).step(
        _step(torch, time=0.5, shot=1)
    )
    second = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"), config=config).step(
        _step(torch, time=2.8, shot=2)
    )

    assert float(second["reach"][0]) == pytest.approx(1.6 * float(first["reach"][0]))


def test_torch_multistep_adds_long_range_signal_only_for_hard_height() -> None:
    torch = pytest.importorskip("torch")
    config = GoalkeeperMultiStepConfig(hard_height_reach_reward_scale=2.0)
    sample = _step(torch, time=0.5, shot=1)
    sample["intercept_target_m"][0, 2] = 1.30
    shaped_task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"), config=config)
    baseline_task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"))
    shaped_task.step(sample)
    baseline_task.step(sample)
    closer = {key: value.clone() for key, value in sample.items()}
    closer["left_hand_position_m"][0, 2] = 1.20
    closer["right_hand_position_m"][0, 2] = 1.20
    shaped = shaped_task.step(closer)
    baseline = baseline_task.step(closer)

    assert float(shaped["reach"][0]) > float(baseline["reach"][0])


def test_torch_potential_reach_cannot_be_farmed_by_holding() -> None:
    torch = pytest.importorskip("torch")
    config = GoalkeeperMultiStepConfig(reach_reward_semantics="POTENTIAL_PROGRESS_ONLY")
    task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"), config=config)
    far = _step(torch, time=0.5, shot=1)
    far["left_hand_position_m"][0, 1] = 0.80
    far["right_hand_position_m"][0, 1] = -0.80
    assert float(task.step(far)["reach"][0]) == pytest.approx(0.0)
    assert float(task.step(far)["reach"][0]) == pytest.approx(0.0)

    closer = {key: value.clone() for key, value in far.items()}
    closer["left_hand_position_m"][0, 1] = 0.20
    closer["right_hand_position_m"][0, 1] = -0.20
    assert float(task.step(closer)["reach"][0]) > 0.0
    assert float(task.step(far)["reach"][0]) < 0.0


def test_torch_task_motion_rewards_body_progress_not_holding() -> None:
    torch = pytest.importorskip("torch")
    config = GoalkeeperMultiStepConfig(task_motion_reward_scale=8.0)
    task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"), config=config)
    far = _step(torch, time=0.5, shot=1)
    far["intercept_target_m"] = torch.tensor([[4.49, 0.90, 0.20]])
    far["pelvis_position_m"] = torch.tensor([[4.52, 0.00, 0.793]])
    assert float(task.step(far)["task_motion"][0]) == pytest.approx(0.0)

    closer = {key: value.clone() for key, value in far.items()}
    closer["pelvis_height_m"][0] = 0.60
    closer["pelvis_position_m"] = torch.tensor([[4.52, 0.45, 0.60]])
    closer["posture_exception_granted"] = torch.tensor([True])
    assert float(task.step(closer)["task_motion"][0]) > 0.0
    assert float(task.step(closer)["task_motion"][0]) == pytest.approx(0.0)
    assert float(task.step(far)["task_motion"][0]) < 0.0


def test_torch_task_motion_fails_closed_without_pelvis_position() -> None:
    torch = pytest.importorskip("torch")
    task = TorchGoalkeeperMultiStepAccumulator(
        1,
        device=torch.device("cpu"),
        config=GoalkeeperMultiStepConfig(task_motion_reward_scale=1.0),
    )
    with pytest.raises(ValueError, match="requires pelvis position"):
        task.step(_step(torch, time=0.5, shot=1))


def test_torch_task_motion_arrival_bonus_is_one_shot_and_hand_coupled() -> None:
    torch = pytest.importorskip("torch")
    config = GoalkeeperMultiStepConfig(task_motion_reward_scale=8.0)
    task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"), config=config)
    prepared = _step(torch, time=0.5, shot=1)
    prepared["ball_position_m"] = torch.tensor([[3.00, 0.90, 0.20]])
    prepared["intercept_target_m"] = torch.tensor([[4.49, 0.90, 0.20]])
    prepared["pelvis_height_m"] = torch.tensor([0.60])
    prepared["pelvis_position_m"] = torch.tensor([[4.52, 0.45, 0.60]])
    prepared["posture_exception_granted"] = torch.tensor([True])
    task.step(prepared)

    aligned = {key: value.clone() for key, value in prepared.items()}
    aligned["ball_position_m"] = torch.tensor([[4.05, 0.90, 0.20]])
    aligned["left_hand_position_m"] = torch.tensor([[4.49, 0.88, 0.20]])
    aligned["right_hand_position_m"] = torch.tensor([[4.49, 0.70, 0.20]])
    arrival_reward = float(task.step(aligned)["task_motion"][0])

    assert arrival_reward > 0.0
    assert float(task.step(aligned)["task_motion"][0]) == pytest.approx(0.0)


def test_torch_multistep_fails_closed_on_fall() -> None:
    torch = pytest.importorskip("torch")
    task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"))
    sample = _step(torch, time=0.5, shot=1)
    sample["pelvis_height_m"][0] = 0.2
    result = task.step(sample)
    assert result["phase"][0] == int(GoalkeeperEpisodePhase.FAILED)
    assert bool(result["terminated"][0])


def test_torch_failed_phase_cannot_be_laundered_by_safe_timeout() -> None:
    torch = pytest.importorskip("torch")
    task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"))
    failed_sample = _step(torch, time=0.3, shot=1)
    failed_sample["pelvis_height_m"][0] = 0.2
    failed = task.step(failed_sample)
    assert failed["phase"][0] == int(GoalkeeperEpisodePhase.FAILED)

    after_quarantine = task.step(_step(torch, time=task.config.episode_duration_sec, shot=0))

    assert after_quarantine["phase"][0] == int(GoalkeeperEpisodePhase.FAILED)
    assert bool(after_quarantine["terminated"][0])
    assert task.summary()["failed_rate"] == 1.0


def test_torch_failed_save_cannot_reenter_recovery() -> None:
    torch = pytest.importorskip("torch")
    task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"))
    task.step(_step(torch, time=0.5, shot=1, contact=True, hand_contact=True, save=True))
    failed_sample = _step(torch, time=0.6, shot=0)
    failed_sample["pelvis_height_m"][0] = 0.2
    failed = task.step(failed_sample)
    assert failed["phase"][0] == int(GoalkeeperEpisodePhase.FAILED)

    after_quarantine = task.step(_step(torch, time=0.7, shot=0))

    assert after_quarantine["phase"][0] == int(GoalkeeperEpisodePhase.FAILED)
    assert not bool(after_quarantine["recovered_after_first"][0])


def test_torch_contact_without_save_cannot_earn_recovery() -> None:
    torch = pytest.importorskip("torch")
    config = GoalkeeperMultiStepConfig(recovery_hold_sec=0.04)
    task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"), config=config)
    task.step(_step(torch, time=0.5, shot=1, contact=True, save=False))
    task.step(_step(torch, time=0.52, shot=0))
    result = task.step(_step(torch, time=0.54, shot=0))

    assert not bool(result["recovered_after_first"][0])


def test_torch_hand_save_is_explicit_and_more_valuable() -> None:
    torch = pytest.importorskip("torch")
    body_task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"))
    hand_task = TorchGoalkeeperMultiStepAccumulator(1, device=torch.device("cpu"))
    body = body_task.step(_step(torch, time=0.5, shot=1, contact=True, save=True))
    hand = hand_task.step(
        _step(torch, time=0.5, shot=1, contact=True, hand_contact=True, save=True)
    )
    assert float(hand["total"][0]) > float(body["total"][0])
    assert bool(hand["first_hand_save"][0])
