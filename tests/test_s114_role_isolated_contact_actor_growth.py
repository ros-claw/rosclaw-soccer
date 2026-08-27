from __future__ import annotations

import hashlib

import numpy as np
import pytest

from rosclaw_soccer.growth.ballistic_contact_impulse_actor import (
    G1BallisticContactImpulseActor,
)
from rosclaw_soccer.skills.team.shared_world import G1PhysicalSecondStrikerConfig
from rosclaw_soccer.training.role_isolated_contact_actor_growth import (
    RoleIsolatedContactActorGrowthConfig,
    RoleIsolatedContactTeacherProbe,
    _fit_candidate,
    _inside_convex_cloud,
    _probe_row,
)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _parent() -> G1BallisticContactImpulseActor:
    hashes = tuple(_sha(f"parent-{index}") for index in range(8))
    return G1BallisticContactImpulseActor(
        body_hash=_sha("body"),
        implementation_hash=_sha("implementation"),
        experiment_context_hash=_sha("parent-context"),
        source_evidence_hashes=hashes,
        selected_evidence_hash=hashes[0],
        selected_goal_plane_target_error_m=0.10,
        precision_success_count=2,
        rejected_probe_count=6,
        task_space_actor_weight_matrix=((400.0, -40.0, 0.0), (350.0, 0.0, -50.0)),
        maximum_lateral_force_n=250.0,
        maximum_vertical_force_n=250.0,
        maximum_foot_ball_distance_m=0.25,
        start_policy_frame=230,
        end_policy_frame=335,
        foot_strike_point_offset_m=(0.13, 0.0, -0.025),
        qualified_error_max_m=0.20,
    )


def _rows() -> list[dict[str, object]]:
    values = (
        (32.366, 0.0, 7.750, 0.138, 5.198, False),
        (32.366, -138.900, 7.203, -0.072, 4.917, False),
        (62.366, -138.900, 7.130, -0.135, 4.932, False),
        (122.366, -138.900, 6.951, -0.246, 4.899, False),
        (182.366, 180.0, 8.389, -0.090, 5.182, True),
        (250.0, 180.0, 8.248, -0.212, 5.011, True),
        (250.0, 120.0, 7.971, -0.289, 5.177, True),
        (212.366, 120.0, 8.051, -0.200, 5.207, True),
        (212.366, 150.0, 8.213, -0.144, 5.105, False),
    )
    return [
        {
            "label": f"probe-{index}",
            "hard_safe": True,
            "teacher_success": success,
            "teacher_peak_force_yz_n": [force_y, force_z],
            "teacher_peak_foot_velocity_yz_mps": [0.92, 0.63],
            "launch_velocity_xyz_mps": [velocity_x, velocity_y, velocity_z],
            "trajectory_hash": _sha(f"trajectory-{index}"),
            "probe": {
                "target_lateral_speed_mps": 7.0,
                "target_vertical_speed_mps": 7.0,
            },
        }
        for index, (force_y, force_z, velocity_x, velocity_y, velocity_z, success) in enumerate(
            values
        )
    ]


def test_target_must_be_inside_measured_convex_launch_support() -> None:
    cloud = np.asarray(((-0.20, 5.20), (-0.09, 5.18), (-0.14, 5.10)))
    assert _inside_convex_cloud(cloud, np.asarray((-0.15, 5.16)))
    assert not _inside_convex_cloud(cloud, np.asarray((0.20, 5.40)))


def test_local_distillation_stays_bound_to_safe_teacher_rehearsal() -> None:
    actor, diagnostics = _fit_candidate(
        rows=_rows(),
        parent_actor=_parent(),
        striker=G1PhysicalSecondStrikerConfig(),
        goal_x_m=7.5,
        context_hash=_sha("context"),
        config=RoleIsolatedContactActorGrowthConfig(),
    )

    desired_y, desired_z = diagnostics["desired_launch_velocity_yz_mps"]
    assert actor.target_conditioned
    assert actor.safe_probe_count == 5
    assert actor.rejected_probe_count >= 2
    assert actor.minimum_supported_lateral_launch_speed_mps < desired_y
    assert actor.maximum_supported_lateral_launch_speed_mps > desired_y
    assert actor.minimum_supported_vertical_launch_speed_mps < desired_z
    assert actor.maximum_supported_vertical_launch_speed_mps > desired_z
    assert diagnostics["target_inside_measured_convex_support"] is True
    assert actor.task_space_actor_weight_matrix[0][3] == -6.0
    assert actor.task_space_actor_weight_matrix[1][4] == -6.0


def test_contact_growth_contract_rejects_hardware_and_weak_local_support() -> None:
    with pytest.raises(ValueError, match="SIM_ONLY"):
        RoleIsolatedContactActorGrowthConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="local probe count"):
        RoleIsolatedContactActorGrowthConfig(local_probe_count=3)
    with pytest.raises(ValueError, match="two exact"):
        RoleIsolatedContactActorGrowthConfig(probe_replay_count=1)


def test_teacher_success_excludes_only_deployment_specific_gates() -> None:
    trajectory = {
        "time": np.asarray((0.0, 0.02)),
        "second_ball_velocity": np.asarray(((0.0, 0.0, 0.0), (8.0, -0.1, 5.1))),
        "second_striker_loft_teacher_active": np.asarray((True, False)),
        "second_striker_loft_teacher_force_yz_n": np.asarray(((200.0, 150.0), (0.0, 0.0))),
        "second_striker_loft_teacher_foot_velocity_yz_mps": np.asarray(((0.9, 0.6), (0.0, 0.0))),
        "second_striker_ballistic_actor_active": np.asarray((False, False)),
    }
    gates = {
        "qualified_first_airborne_save": True,
        "collision_faithful_high_glove_contact": True,
        "outward_physical_save": True,
        "whole_world_safety": True,
        "learned_multi_role_contact_stack_active": False,
        "final_goalkeeper_ready": False,
    }
    row = _probe_row(
        probe=RoleIsolatedContactTeacherProbe("probe", 8.0, 6.0, 250.0, 150.0),
        trajectory=trajectory,
        result={
            "second_striker_contact_time_sec": 0.02,
            "second_striker_contact_observed": True,
            "finite_state": True,
        },
        evaluation={"first_takeoff_exam": {"passed": True}, "gates": gates},
    )

    assert row["teacher_success"] is True
    assert "learned_multi_role_contact_stack_active" not in row["teacher_success_gates"]
    assert "final_goalkeeper_ready" not in row["teacher_success_gates"]
