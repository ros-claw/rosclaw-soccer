from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.growth.three_axis_contact_actor import (
    G1ThreeAxisContactActor,
    fit_g1_three_axis_contact_actor,
    load_g1_three_axis_contact_actor,
    project_g1_three_axis_contact_actor,
    save_g1_three_axis_contact_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.runtime_causal_strike_exam import (
    default_runtime_causal_strike_holdouts,
)
from rosclaw_soccer.training.three_axis_contact_discovery import ThreeAxisContactProbe
from rosclaw_soccer.training.three_axis_contact_growth import (
    train_three_axis_contact_actor,
)


def _sha(value: str) -> str:
    return "sha256:" + value.encode().hex().ljust(64, "0")[:64]


def _actor() -> G1ThreeAxisContactActor:
    return G1ThreeAxisContactActor(
        body_hash=_sha("body"),
        implementation_hash=_sha("implementation"),
        source_evidence_hashes=tuple(_sha(f"evidence-{index}") for index in range(4)),
        training_snapshot_hash=_sha("snapshot"),
        task_space_actor_weight_matrix=(
            (80.0, -20.0, 0.0, 0.0),
            (5.0, 0.0, -10.0, 0.0),
            (10.0, 0.0, 0.0, -5.0),
        ),
        minimum_force_xyz_n=(0.0, -40.0, -30.0),
        maximum_force_xyz_n=(80.0, 40.0, 30.0),
        maximum_foot_ball_distance_m=0.50,
        start_policy_frame=220,
        end_policy_frame=320,
        foot_strike_point_offset_m=(0.13, 0.0, -0.025),
        training_sample_count=128,
        training_trajectory_count=6,
        failed_trajectory_count=2,
        distillation_rmse_n=0.25,
    )


def test_three_axis_actor_projects_all_axes_through_role_specific_jacobian() -> None:
    actor = _actor()
    jacobian = np.zeros((3, 128), dtype=np.float64)
    jacobian[0, 90] = 0.5
    jacobian[1, 91] = 0.25
    jacobian[2, 92] = -0.5
    velocity = np.zeros(128, dtype=np.float64)
    velocity[90] = 2.0
    velocity[91] = 4.0
    velocity[92] = -2.0

    effect = project_g1_three_axis_contact_actor(
        jacobian_position=jacobian,
        generalized_velocity=velocity,
        actor=actor,
        actuated_dof_indices=np.arange(90, 119, dtype=np.int64),
    )

    np.testing.assert_allclose(effect.foot_velocity_xyz_mps, (1.0, 1.0, 1.0))
    np.testing.assert_allclose(effect.force_xyz_n, (60.0, -5.0, 5.0))
    np.testing.assert_allclose(effect.torque[:3], (30.0, -1.25, -2.5))
    assert effect.active


def test_three_axis_actor_is_fitted_from_measured_teacher_samples() -> None:
    rng = np.random.default_rng(128)
    velocity = rng.uniform(-2.0, 2.0, size=(96, 3))
    expected_weights = np.asarray(
        (
            (70.0, -12.0, 0.0, 0.0),
            (-5.0, 0.0, -8.0, 0.0),
            (10.0, 0.0, 0.0, -4.0),
        )
    )
    design = np.column_stack((np.ones(len(velocity)), velocity))
    force = design @ expected_weights.T

    actor = fit_g1_three_axis_contact_actor(
        foot_velocity_xyz_mps=velocity,
        teacher_force_xyz_n=force,
        body_hash=_sha("body"),
        implementation_hash=_sha("implementation"),
        source_evidence_hashes=tuple(_sha(f"fit-evidence-{index}") for index in range(4)),
        training_trajectory_count=6,
        failed_trajectory_count=2,
        maximum_foot_ball_distance_m=0.50,
        start_policy_frame=230,
        end_policy_frame=335,
    )

    np.testing.assert_allclose(actor.task_space_actor_weight_matrix, expected_weights, atol=1e-4)
    assert actor.training_sample_count == 96
    assert actor.distillation_rmse_n < 1e-4


def test_three_axis_actor_mirrors_lateral_policy_but_not_other_axes() -> None:
    actor = _actor()
    jacobian = np.zeros((3, 35), dtype=np.float64)
    jacobian[:, 6:9] = np.eye(3)
    velocity = np.zeros(35, dtype=np.float64)
    velocity[7] = 1.0

    right = project_g1_three_axis_contact_actor(
        jacobian_position=jacobian,
        generalized_velocity=velocity,
        actor=actor,
    )
    left = project_g1_three_axis_contact_actor(
        jacobian_position=jacobian,
        generalized_velocity=velocity,
        actor=actor,
        lateral_mirror_sign=-1.0,
    )

    assert right.force_xyz_n[0] == left.force_xyz_n[0]
    assert right.force_xyz_n[2] == left.force_xyz_n[2]
    assert right.force_xyz_n[1] == pytest.approx(-5.0)
    assert left.force_xyz_n[1] == pytest.approx(-15.0)
    with pytest.raises(ValueError, match="mirror sign"):
        project_g1_three_axis_contact_actor(
            jacobian_position=jacobian,
            generalized_velocity=velocity,
            actor=actor,
            lateral_mirror_sign=0.0,
        )


def test_three_axis_actor_artifact_is_content_bound_and_sim_only(tmp_path) -> None:
    actor = _actor()
    path = tmp_path / "actor.json"
    save_g1_three_axis_contact_actor(actor, path)

    loaded = load_g1_three_axis_contact_actor(path)
    assert loaded == actor
    assert loaded.actor_hash == actor.actor_hash
    assert loaded.to_dict()["teacher_required_at_runtime"] is False
    assert loaded.to_dict()["direct_joint_torque_output"] is True

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["maximum_force_xyz_n"][0] = 79.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_three_axis_contact_actor(path)

    with pytest.raises(ValueError, match="SIM-only"):
        replace(actor, hardware_authorized=True)


def test_three_axis_actor_rejects_bad_evidence_and_nonfinite_state() -> None:
    actor = _actor()
    with pytest.raises(ValueError, match="four unique"):
        replace(actor, source_evidence_hashes=actor.source_evidence_hashes[:3])
    with pytest.raises(ValueError, match="SIM-only contract"):
        replace(actor, failed_trajectory_count=0)

    jacobian = np.zeros((3, 35), dtype=np.float64)
    velocity = np.zeros(35, dtype=np.float64)
    velocity[0] = np.nan
    with pytest.raises(FloatingPointError, match="finite"):
        project_g1_three_axis_contact_actor(
            jacobian_position=jacobian,
            generalized_velocity=velocity,
            actor=actor,
        )


def test_three_axis_growth_binds_failures_and_exact_teacher_channels(tmp_path) -> None:
    discovery = tmp_path / "discovery"
    discovery.mkdir()
    rows = []
    for index in range(4):
        velocity = np.column_stack(
            (
                np.linspace(-1.0, 1.0, 16) + index * 0.02,
                np.linspace(0.5, -0.5, 16),
                np.linspace(-0.2, 0.4, 16),
            )
        )
        design = np.column_stack((np.ones(16), velocity))
        weights = np.asarray(
            ((70.0, -10.0, 0.0, 0.0), (0.0, 0.0, -5.0, 0.0), (5.0, 0.0, 0.0, -2.0))
        )
        force = design @ weights.T
        trajectory_path = discovery / f"probe-{index:02d}.npz"
        np.savez_compressed(
            trajectory_path,
            shooter_loft_teacher_active=np.ones(16, dtype=np.bool_),
            shooter_loft_teacher_force_xyz_n=force,
            shooter_loft_teacher_foot_velocity_xyz_mps=velocity,
        )
        rows.append(
            {
                "quality": {"safe": index != 3, "chain_passed": index < 2},
                "trajectory": {
                    "file": trajectory_path.name,
                    "file_hash": hash_bytes(trajectory_path.read_bytes()),
                },
            }
        )
    report = {
        "schema_version": "rosclaw.growth.three_axis_contact_discovery.v1",
        "status": "PASS_THREE_AXIS_CONTACT_DISCOVERY",
        "body_hash": _sha("body"),
        "implementation_hash": _sha("implementation"),
        "teacher_config": {
            "maximum_foot_ball_distance_m": 0.50,
            "start_policy_frame": 230,
            "end_policy_frame": 335,
            "foot_strike_point_offset_m": [0.13, 0.0, -0.025],
        },
        "rows": rows,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = discovery / "discovery-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    actor, training = train_three_axis_contact_actor(
        discovery_report_path=report_path,
        output_dir=tmp_path / "training",
    )

    assert actor.training_trajectory_count == 4
    assert actor.failed_trajectory_count == 2
    assert actor.training_sample_count == 48
    assert training["teacher_enabled_at_runtime"] is False
    assert training["actor_hash"] == actor.actor_hash


def test_three_axis_probe_binds_context_stance_and_contact_clock() -> None:
    probe = ThreeAxisContactProbe(
        context=default_runtime_causal_strike_holdouts()[0],
        maximum_arrival_advance_frames=0,
        stance_offset_x_m=0.12,
    )

    assert probe.probe_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="SIM-only envelope"):
        replace(probe, stance_offset_x_m=0.121)
    with pytest.raises(ValueError, match="SIM-only envelope"):
        replace(probe, contact_policy_frame=300)
