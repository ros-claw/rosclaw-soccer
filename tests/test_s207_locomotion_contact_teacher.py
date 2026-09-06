from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import (
    G1LocomotionContactTeacherConfig,
    G1RollingOptionBridgeConfig,
    locomotion_contact_teacher_effect,
)


class _Data:
    def __init__(self) -> None:
        self.xpos = np.asarray(((0.0, 0.0, 0.10),), dtype=np.float64)
        self.qvel = np.zeros(32, dtype=np.float64)


class _Model:
    nv = 32


def test_contact_teacher_is_sim_only_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    mujoco = pytest.importorskip("mujoco")

    def fake_jac(
        model: object,
        data: object,
        jacobian: np.ndarray,
        rotation: np.ndarray,
        point: np.ndarray,
        body_id: int,
    ) -> None:
        del model, data, rotation, point, body_id
        jacobian[:, 3:6] = np.eye(3)

    monkeypatch.setattr(mujoco, "mj_jac", fake_jac)
    config = G1LocomotionContactTeacherConfig(maximum_joint_residual_nm=4.0)
    effect = locomotion_contact_teacher_effect(
        model=_Model(),
        data=_Data(),
        ankle_body_id=0,
        actuated_dof_indices=np.arange(3, 32, dtype=np.int64),
        ball_position_m=np.asarray((0.22, 0.0, 0.115)),
        ball_velocity_mps=np.zeros(3, dtype=np.float64),
        desired_ball_direction_xy=np.asarray((1.0, 0.0)),
        contact_mode="strike",
        local_lateral_sign=1.0,
        contact_recent=False,
        config=config,
    )

    assert effect.active
    assert np.max(np.abs(effect.torque_nm)) > 0.0
    assert np.max(np.abs(effect.torque_nm)) <= 4.0
    assert effect.ankle_target_m[0] < 0.22
    assert config.training_only
    assert not config.hardware_authorized
    assert effect.ankle_target_m[1] > 0.0
    with pytest.raises(ValueError, match="SIM-only"):
        replace(config, hardware_authorized=True)


def test_contact_teacher_stays_inactive_outside_contact_envelope() -> None:
    effect = locomotion_contact_teacher_effect(
        model=_Model(),
        data=_Data(),
        ankle_body_id=0,
        actuated_dof_indices=np.arange(3, 32, dtype=np.int64),
        ball_position_m=np.asarray((2.0, 0.0, 0.115)),
        ball_velocity_mps=np.zeros(3, dtype=np.float64),
        desired_ball_direction_xy=np.asarray((1.0, 0.0)),
        contact_mode="strike",
        local_lateral_sign=-1.0,
        contact_recent=False,
        config=G1LocomotionContactTeacherConfig(),
    )

    assert not effect.active
    assert np.count_nonzero(effect.torque_nm) == 0


def test_contact_teacher_rejects_a_foot_that_has_overrun_the_ball() -> None:
    data = _Data()
    data.xpos[0, 0] = 0.32
    effect = locomotion_contact_teacher_effect(
        model=_Model(),
        data=data,
        ankle_body_id=0,
        actuated_dof_indices=np.arange(3, 32, dtype=np.int64),
        ball_position_m=np.asarray((0.22, 0.0, 0.115)),
        ball_velocity_mps=np.zeros(3, dtype=np.float64),
        desired_ball_direction_xy=np.asarray((1.0, 0.0)),
        contact_mode="strike",
        local_lateral_sign=1.0,
        contact_recent=False,
        config=G1LocomotionContactTeacherConfig(),
    )

    assert effect.longitudinal_foot_offset_m > 0.0
    assert not effect.active
    assert np.count_nonzero(effect.torque_nm) == 0


def test_receive_teacher_cushions_forward_instead_of_reflecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mujoco = pytest.importorskip("mujoco")

    def fake_jac(
        model: object,
        data: object,
        jacobian: np.ndarray,
        rotation: np.ndarray,
        point: np.ndarray,
        body_id: int,
    ) -> None:
        del model, data, rotation, point, body_id
        jacobian[:, 3:6] = np.eye(3)

    monkeypatch.setattr(mujoco, "mj_jac", fake_jac)
    data = _Data()
    data.xpos[0, 0] = 0.30
    effect = locomotion_contact_teacher_effect(
        model=_Model(),
        data=data,
        ankle_body_id=0,
        actuated_dof_indices=np.arange(3, 32, dtype=np.int64),
        ball_position_m=np.asarray((0.22, 0.0, 0.115)),
        ball_velocity_mps=np.asarray((0.8, 0.0, 0.0)),
        desired_ball_direction_xy=np.asarray((1.0, 0.0)),
        contact_mode="receive",
        local_lateral_sign=1.0,
        contact_recent=False,
        config=G1LocomotionContactTeacherConfig(),
    )

    assert effect.active
    assert effect.ankle_target_m[0] > data.xpos[0, 0]
    assert effect.task_force_n[0] > 0.0


def test_rolling_option_bridge_is_training_only() -> None:
    config = G1RollingOptionBridgeConfig()

    assert config.activation_ceiling == "SIM_ONLY"
    assert config.training_only
    assert not config.hardware_authorized
    with pytest.raises(ValueError, match="SIM-only"):
        replace(config, activation_ceiling="REAL")
