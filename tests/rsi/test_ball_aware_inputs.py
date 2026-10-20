import numpy as np
import pytest

from rosclaw_soccer.rsi.ball_aware_inputs import (
    BALL_AWARE_ABI,
    ball_goal9,
    require_ball_aware_qualification,
)


def fixture():
    return dict(
        pelvis_rotation=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        pelvis_position=np.array([10.0, 20.0, 1.0]),
        pelvis_linear_velocity=np.array([1.0, 0.0, 0.0]),
        ball_position=np.array([11.0, 22.0, 0.11]),
        ball_linear_velocity=np.array([0.0, 2.0, 0.0]),
        target_position=np.array([13.0, 24.0, 2.0]),
    )


def test_measured_ball_and_relative_world_velocity_use_current_pelvis_basis():
    values = fixture()
    result = ball_goal9(**values)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [2.0, -1.0, -0.89, 2.0, 1.0, 0.0, 4.0, -3.0, 1.0], atol=1e-7)
    for name in ("pelvis_position", "ball_position", "target_position"):
        values[name] += np.array([40.0, -25.0, 0.0])
    np.testing.assert_allclose(ball_goal9(**values), result, atol=1e-7)
    values["ball_position"][1] += 1
    assert ball_goal9(**values)[0] == 3


@pytest.mark.parametrize("kind", ["nan", "improper", "shape", "dtype"])
def test_invalid_perception_rejected_without_repair(kind):
    values = fixture()
    if kind == "nan":
        values["ball_linear_velocity"][0] = np.nan
    elif kind == "improper":
        values["pelvis_rotation"][2, 2] = -1
    elif kind == "shape":
        values["target_position"] = np.zeros(2)
    else:
        values["ball_position"] = np.zeros(3, dtype=int)
    with pytest.raises(ValueError):
        ball_goal9(**values)


def qualification():
    expected = [[i % 2 == 0, i % 2 == 1] if i < 6 else [False, False] for i in range(8)]
    return dict(
        status="BALL_AWARE_ABI_SENSOR_QUALIFICATION_NOT_LEARNED_SKILL",
        ball_aware_abi=BALL_AWARE_ABI,
        actor_observation_dim=177,
        critic_observation_dim=304,
        checkpoint_file="ball-aware-initialization.pt",
        new_optimizer_steps=0,
        optimizer_state_migrated=False,
        weights_only_checkpoint_reload=True,
        goal_complete=False,
        football_skill_proven=False,
        hardware_authorized=False,
        promotion_authorized=False,
        perception_max_error=1e-6,
        actor_migration_max_error=0.0,
        episode_gpu_world_steps=23120,
        actual_neural_rows=4624,
        perception_native_fk_calls=2312,
        fixture_gpu_world_steps=80,
        fixture_native_steps=80,
        full_reference_episodes_completed=[True] * 8,
        failed=[False] * 8,
        right_censored=[False] * 8,
        sensor_calibration=dict(
            expected_labels=expected,
            gpu_labels=expected,
            native_labels=expected,
            native_sensor_vs_collision_max_error_n=0.0,
            max_qpos_error_mixed_units=1e-7,
            max_qvel_error_mixed_units=0.001,
        ),
    )


def test_qualification_checks_physics_and_new_abi_not_just_status():
    require_ball_aware_qualification(qualification())


@pytest.mark.parametrize(
    "key,value",
    [
        ("new_optimizer_steps", 1),
        ("actor_observation_dim", 168),
        ("actual_neural_rows", 2312),
        ("perception_max_error", float("nan")),
        ("actor_migration_max_error", 1.0),
        ("fixture_native_steps", 0),
        ("full_reference_episodes_completed", [True] * 7 + [False]),
        ("promotion_authorized", True),
        ("sensor_calibration", {}),
        ("checkpoint_file", "../parent.pt"),
    ],
)
def test_invalid_physics_qualification_cannot_unlock_training(key, value):
    report = qualification()
    report[key] = value
    with pytest.raises(ValueError):
        require_ball_aware_qualification(report)
