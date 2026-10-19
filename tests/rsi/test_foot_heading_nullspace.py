"""Kinematic proposal arithmetic only, not native motion or training."""

import inspect

import numpy as np
import pytest

from rosclaw_soccer.rsi import native_first_touch_episode
from rosclaw_soccer.rsi.foot_heading_nullspace import (
    check_recorded_pose_proposal,
    propose_foot_heading,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_initial_contact_heading import kinematic_fixture


def inputs(angle=np.pi / 6):
    linear, angular = np.eye(3, 6), np.zeros((3, 6))
    angular[2, 5] = 1.0
    return [
        linear,
        angular,
        np.array([np.cos(angle), np.sin(angle), 0.0]),
        np.zeros(6),
        np.zeros(6),
        np.tile([-2.0, 2.0], (6, 1)),
    ]


def test_bounded_heading_request_preserves_linear_position_and_has_no_authority():
    args = inputs()
    total, r = propose_foot_heading(*args, enabled=True)
    np.testing.assert_array_equal(total[:5], np.zeros(5))
    assert total[5] == pytest.approx(-0.1)
    assert r["requested_heading_change_rad"] == pytest.approx(-np.pi / 18)
    assert r["predicted_additional_heading_change_rad"] == pytest.approx(-0.1)
    np.testing.assert_array_equal(r["predicted_position_increment_m"], np.zeros(3))
    assert r["position_jacobian_rank"] == 3
    assert r["linearization_only"]
    assert r["new_physical_steps"] == r["optimizer_updates"] == 0
    for key in (
        "trained_policy",
        "physics_qualified",
        "runtime_execution_authorized",
        "hardware_authorized",
        "promotion_authorized",
    ):
        assert r[key] is False
    assert r["report_hash"] == hash_json({k: v for k, v in r.items() if k != "report_hash"})


def test_off_preserves_original_object_and_negative_zero_bits():
    args = inputs()
    args[4][0] = -0.0
    old = args[4].tobytes()
    total, r = propose_foot_heading(*args, enabled=False)
    assert total is args[4] and total.tobytes() == old
    assert r["status"] == "OFF_UNCHANGED"
    assert not np.asarray(r["additional_joint_applied_rad"]).any()


@pytest.mark.parametrize(
    "mode", ["no_yaw", "no_limit_room", "baseline_outside", "already_aligned", "tilted", "backward"]
)
def test_unreachable_or_out_of_domain_requests_leave_original_reference_unchanged(mode):
    args = inputs()
    if mode == "no_yaw":
        args[1][:] = 0
    elif mode == "no_limit_room":
        args[3][5] = -2
    elif mode == "baseline_outside":
        args[3][5] = -2.1
    elif mode == "already_aligned":
        args = inputs(0.0)
    elif mode == "tilted":
        args[2][:] = [0.1, 0, np.sqrt(0.99)]
    else:
        args = inputs(np.pi)
    total, r = propose_foot_heading(*args, enabled=True)
    assert total is args[4]
    assert r["status"].endswith("UNCHANGED")
    assert not np.asarray(r["additional_joint_applied_rad"]).any()


@pytest.mark.parametrize("bound", ["total", "joint"])
def test_uniform_scaling_preserves_nullspace_at_actual_limits(bound):
    args = inputs()
    if bound == "total":
        args[4][5] = -0.34
        expected = -0.35
    else:
        args[3][5] = -1.95
        expected = -0.05
    total, r = propose_foot_heading(*args, enabled=True)
    assert total[5] == pytest.approx(expected)
    assert 0 < r["uniform_limit_scale"] < 1
    np.testing.assert_allclose(args[0] @ (total - args[4]), 0, atol=1e-12)


def test_exact_heading_derivative_includes_foot_tilt():
    args = inputs()
    f = np.array([0.8, 0.3, 0.4])
    f /= np.linalg.norm(f)
    args[2] = f
    omega = np.array([0.1, 0.2, 0.3])
    args[1][:, 5] = omega
    total, r = propose_foot_heading(*args, enabled=True)
    eps = 1e-6
    plus, minus = f + eps * np.cross(omega, f), f - eps * np.cross(omega, f)
    derivative = (np.arctan2(plus[1], plus[0]) - np.arctan2(minus[1], minus[0])) / (2 * eps)
    assert r["predicted_additional_heading_change_rad"] == pytest.approx(
        derivative * total[5], abs=1e-10
    )


def test_random_linearizations_preserve_limits_and_inputs_but_do_not_claim_realized_motion():
    rng = np.random.default_rng(2026101903)
    for _ in range(64):
        args = inputs(float(rng.uniform(-np.pi / 4, np.pi / 4)))
        args[0] = rng.normal(size=(3, 6))
        args[1] = rng.normal(size=(3, 6))
        args[3] = rng.uniform(-0.3, 0.3, 6)
        args[4] = rng.uniform(-0.3, 0.3, 6)
        before = [a.copy() for a in args]
        total, r = propose_foot_heading(*args, enabled=True)
        additional = total - args[4]
        assert np.max(np.abs(additional)) <= 0.1 + 1e-12
        assert np.max(np.abs(total)) <= 0.35 + 1e-12
        np.testing.assert_allclose(args[0] @ additional, 0, atol=1e-8)
        assert (
            r["predicted_additional_heading_change_rad"] * r["measured_foot_heading_rad"] <= 1e-12
        )
        for a, b in zip(args, before, strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "mutation", ["shape", "nan", "dtype", "old_cap", "axis", "limits", "list", "truthy"]
)
def test_invalid_inputs_do_not_make_a_motion_proposal(mutation):
    args = inputs()
    enabled = True
    if mutation == "shape":
        args[0] = np.zeros((3, 5))
    elif mutation == "nan":
        args[0][0, 0] = np.nan
    elif mutation == "dtype":
        args[0] = args[0].astype(np.float32)
    elif mutation == "old_cap":
        args[4][0] = 0.36
    elif mutation == "axis":
        args[2][:] = [1.1, 0, 0]
    elif mutation == "limits":
        args[5][:, 1] = -3
    elif mutation == "list":
        args[0] = args[0].tolist()
    else:
        enabled = 1
    with pytest.raises(ValueError):
        propose_foot_heading(*args, enabled=enabled)


def test_native_heading_extension_is_explicit_and_default_off():
    signature = inspect.signature(native_first_touch_episode.run_native_first_touch_episode)
    assert signature.parameters["foot_heading_control"].default is None


@pytest.mark.parametrize("enabled", [False, True])
def test_recorded_pose_check_uses_actual_jacobian_and_fk_without_simulating_or_qualifying(
    enabled, monkeypatch
):
    import mujoco

    model, _, arrays = kinematic_fixture()
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("no physical integration"))
    monkeypatch.setattr(
        mujoco, "mj_forward", lambda *a: pytest.fail("minimal kinematic pipeline only")
    )
    q = arrays["canonical_qpos_before_substep"][1200]
    before = q.copy()
    r = check_recorded_pose_proposal(model, q, "left_ankle_roll_link", enabled=enabled)
    assert r["measured_original_heading_rad"] == pytest.approx(np.pi / 6)
    if enabled:
        assert abs(r["forward_kinematic_proposed_heading_rad"]) < np.pi / 6
    else:
        assert r["forward_kinematic_proposed_heading_rad"] == r["measured_original_heading_rad"]
    assert r["body_origin_displacement_norm_m"] < 1e-12
    assert r["preserves_body_origin_not_contact_point"]
    assert not r["balance_or_ball_outcome_verified"]
    assert not r["physical_integration_performed"]
    assert r["retrospective_pose_analysis_not_actor_observation"]
    np.testing.assert_array_equal(q, before)
