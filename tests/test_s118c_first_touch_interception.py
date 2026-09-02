from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.first_touch_interception import (
    FirstTouchInterceptionConfig,
    first_touch_interception_effect,
)


def test_interception_reflex_decodes_task_error_to_bounded_joint_torque(monkeypatch) -> None:
    def fake_jac(_model, _data, jacobian, _rotation, _point, _body_id) -> None:
        jacobian[1, 6] = 0.5
        jacobian[2, 7] = 0.4

    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mj_jac=fake_jac))
    model = SimpleNamespace(nv=35)
    data = SimpleNamespace(
        xpos=np.asarray(((0.0, 0.0, 0.0),)),
        qvel=np.zeros(35, dtype=np.float64),
    )
    config = FirstTouchInterceptionConfig()

    effect = first_touch_interception_effect(
        model=model,
        data=data,
        striking_ankle_body_id=0,
        actuated_dof_indices=np.arange(6, 35, dtype=np.int64),
        ball_position_m=np.asarray((0.10, 0.0, 0.0)),
        ball_velocity_mps=np.zeros(3),
        policy_frame=245,
        contact_observed=False,
        kick_foot="right",
        config=config,
    )

    assert effect.active
    np.testing.assert_allclose(effect.position_error_m, (0.0, 0.21, -0.07))
    np.testing.assert_allclose(effect.torque[:2], (6.0, -1.68))
    assert np.max(np.abs(effect.torque)) <= config.maximum_joint_residual_nm

    left = first_touch_interception_effect(
        model=model,
        data=data,
        striking_ankle_body_id=0,
        actuated_dof_indices=np.arange(6, 35, dtype=np.int64),
        ball_position_m=np.asarray((0.10, 0.0, 0.0)),
        ball_velocity_mps=np.zeros(3),
        policy_frame=245,
        contact_observed=False,
        kick_foot="left",
        config=config,
    )
    assert left.torque[0] == pytest.approx(-6.0)


def test_interception_reflex_stops_after_contact_and_rejects_unsafe_authority() -> None:
    model = SimpleNamespace(nv=35)
    data = SimpleNamespace(
        xpos=np.asarray(((0.0, 0.0, 0.0),)),
        qvel=np.zeros(35, dtype=np.float64),
    )
    effect = first_touch_interception_effect(
        model=model,
        data=data,
        striking_ankle_body_id=0,
        actuated_dof_indices=np.arange(6, 35, dtype=np.int64),
        ball_position_m=np.asarray((0.10, 0.0, 0.0)),
        ball_velocity_mps=np.zeros(3),
        policy_frame=245,
        contact_observed=True,
        kick_foot="right",
        config=FirstTouchInterceptionConfig(),
    )
    assert not effect.active
    np.testing.assert_array_equal(effect.torque, np.zeros(29))
    with pytest.raises(ValueError, match="must remain SIM_ONLY"):
        replace(FirstTouchInterceptionConfig(), hardware_authorized=True)


def test_interception_reflex_rejects_legacy_or_duplicated_dof_slices() -> None:
    with pytest.raises(ValueError, match="29 unique"):
        first_touch_interception_effect(
            model=SimpleNamespace(nv=35),
            data=SimpleNamespace(
                xpos=np.asarray(((0.0, 0.0, 0.0),)),
                qvel=np.zeros(35, dtype=np.float64),
            ),
            striking_ankle_body_id=0,
            actuated_dof_indices=np.full(29, 6, dtype=np.int64),
            ball_position_m=np.asarray((0.10, 0.0, 0.0)),
            ball_velocity_mps=np.zeros(3),
            policy_frame=245,
            contact_observed=False,
            kick_foot="right",
            config=FirstTouchInterceptionConfig(),
        )
