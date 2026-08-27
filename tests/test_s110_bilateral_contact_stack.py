from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip(
    "rosclaw.feedback.contracts",
    reason="requires the stacked ROSClaw Growth Core contracts",
)

from rosclaw_soccer.growth.ballistic_contact_impulse_actor import (
    G1BallisticContactImpulseActor,
    g1_ballistic_contact_impulse_effect,
)
from rosclaw_soccer.growth.ballistic_contact_residual import (
    G1BallisticContactResidualConfig,
    blend_g1_ballistic_contact_target,
)
from rosclaw_soccer.growth.ballistic_contact_torque_residual import (
    G1BallisticContactTorqueResidualConfig,
    g1_ballistic_contact_torque_residual,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import mirror_g1_joint_positions
from rosclaw_soccer.skills.team.shared_world import G1PhysicalSecondStrikerConfig


def _hash(label: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _actor() -> G1BallisticContactImpulseActor:
    sources = tuple(_hash(f"bilateral-{index}") for index in range(8))
    return G1BallisticContactImpulseActor(
        body_hash=_hash("body"),
        implementation_hash=_hash("implementation"),
        experiment_context_hash=_hash("context"),
        source_evidence_hashes=sources,
        selected_evidence_hash=sources[0],
        selected_goal_plane_target_error_m=0.10,
        precision_success_count=2,
        rejected_probe_count=6,
        task_space_actor_weight_matrix=((400.0, -40.0, 0.0), (350.0, 0.0, -50.0)),
        maximum_lateral_force_n=250.0,
        maximum_vertical_force_n=250.0,
        maximum_foot_ball_distance_m=0.18,
        start_policy_frame=230,
        end_policy_frame=335,
        foot_strike_point_offset_m=(0.13, 0.0, -0.025),
        qualified_error_max_m=0.16,
    )


def test_contact_target_and_torque_memories_are_exact_sagittal_mirrors() -> None:
    target_config = G1BallisticContactResidualConfig(
        right_leg_residual_rad=(0.01, -0.02, 0.03, -0.04, 0.05, -0.06)
    )
    torque_config = G1BallisticContactTorqueResidualConfig(
        right_leg_residual_nm=(1.0, -2.0, 3.0, -4.0, 5.0, -6.0),
        right_leg_preload_nm=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        counterbalance_residual_nm=(0.5, -0.5, 0.25, -0.25, 0.2, -0.2),
    )
    target = np.zeros(29, dtype=np.float64)
    _, right_delta, right_active = blend_g1_ballistic_contact_target(
        target=target,
        policy_frame=target_config.contact_policy_frame,
        control_dt_sec=0.02,
        config=target_config,
        kick_foot="right",
    )
    _, left_delta, left_active = blend_g1_ballistic_contact_target(
        target=target,
        policy_frame=target_config.contact_policy_frame,
        control_dt_sec=0.02,
        config=target_config,
        kick_foot="left",
    )
    right_torque, right_torque_active = g1_ballistic_contact_torque_residual(
        policy_frame=torque_config.contact_policy_frame,
        control_dt_sec=0.02,
        config=torque_config,
        kick_foot="right",
    )
    left_torque, left_torque_active = g1_ballistic_contact_torque_residual(
        policy_frame=torque_config.contact_policy_frame,
        control_dt_sec=0.02,
        config=torque_config,
        kick_foot="left",
    )

    assert right_active and left_active and right_torque_active and left_torque_active
    np.testing.assert_allclose(left_delta, mirror_g1_joint_positions(right_delta))
    np.testing.assert_allclose(left_torque, mirror_g1_joint_positions(right_torque))
    assert np.count_nonzero(left_delta[:6]) == 6
    assert np.count_nonzero(left_delta[6:12]) == 0


def test_contact_actor_decodes_the_selected_left_ankle_in_canonical_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[int] = []

    def fake_jac(
        _model: Any,
        _data: Any,
        jacobian: np.ndarray[Any, Any],
        _rotation: Any,
        _point: Any,
        body_id: int,
    ) -> None:
        called.append(int(body_id))
        jacobian[1, 35] = 0.5
        jacobian[2, 36] = 0.4

    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mj_jac=fake_jac))
    model = SimpleNamespace(nv=64)
    data = SimpleNamespace(
        xmat=np.asarray((np.eye(3), np.eye(3))),
        # The legacy right ankle is deliberately far from the ball.  The
        # selected anatomical left ankle owns both the point and Jacobian.
        xpos=np.asarray(((4.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
        qvel=np.zeros(64, dtype=np.float64),
    )
    effect = g1_ballistic_contact_impulse_effect(
        model=model,
        data=data,
        right_ankle_body_id=0,
        striking_ankle_body_id=1,
        lateral_mirror_sign=-1.0,
        actor=_actor(),
        policy_frame=255,
        contact_observed=False,
        ball_position=np.asarray((0.13, 0.0, -0.025)),
        actuated_dof_indices=np.arange(35, 64, dtype=np.int64),
    )

    assert called == [1]
    assert effect.active
    assert effect.lateral_force_n == -250.0
    assert effect.vertical_force_n == 250.0
    np.testing.assert_allclose(effect.torque[:2], (-125.0, 100.0))
    assert np.count_nonzero(effect.torque) == 2


def test_bilateral_contracts_fail_closed_on_unknown_foot() -> None:
    with pytest.raises(ValueError, match="kick foot"):
        G1PhysicalSecondStrikerConfig(kick_foot="tail")
    with pytest.raises(ValueError, match="ballistic target"):
        G1PhysicalSecondStrikerConfig(ballistic_target_z_m=0.115)
    with pytest.raises(ValueError, match="kick foot"):
        blend_g1_ballistic_contact_target(
            target=np.zeros(29),
            policy_frame=256,
            control_dt_sec=0.02,
            config=G1BallisticContactResidualConfig(),
            kick_foot="tail",
        )
    with pytest.raises(ValueError, match="mirror sign"):
        g1_ballistic_contact_impulse_effect(
            model=SimpleNamespace(nv=35),
            data=SimpleNamespace(xmat=np.asarray((np.eye(3),)), xpos=np.zeros((1, 3))),
            right_ankle_body_id=0,
            lateral_mirror_sign=0.0,
            actor=_actor(),
            policy_frame=255,
            contact_observed=False,
            ball_position=np.asarray((0.13, 0.0, -0.025)),
        )
