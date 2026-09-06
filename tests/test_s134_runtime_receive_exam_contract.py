from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.shared_world import _simulate_shared_world
from rosclaw_soccer.training.multitarget_neural_contact_canary import (
    validate_multitarget_neural_contact_canary,
)
from rosclaw_soccer.training.neural_contact_growth import _bound_teacher_report
from rosclaw_soccer.training.runtime_receive_contact_teacher import (
    default_runtime_receive_contact_probes,
    runtime_receive_contact_countersteer_probes,
)
from rosclaw_soccer.training.runtime_receive_discovery import (
    runtime_receive_direction_actions,
    runtime_receive_timing_actions,
)
from rosclaw_soccer.training.runtime_receive_exam import (
    default_runtime_receive_holdouts,
    default_runtime_receive_v2_holdouts,
    default_runtime_receive_v3_holdouts,
    validate_runtime_receive_exam,
)
from rosclaw_soccer.training.runtime_receive_growth import _bound_discovery


def test_runtime_receive_holdouts_are_fresh_unique_and_role_partitioned() -> None:
    holdouts = default_runtime_receive_holdouts()

    assert len(holdouts) == 6
    assert len({context.context_hash for context, _ in holdouts}) == 6
    assert all(context.case_id.startswith("s134.holdout.") for context, _ in holdouts)
    assert all(action.body_yaw_correction_rad == 0.04 for _, action in holdouts)
    assert sum(action.stance_correction_x_m == -0.02 for _, action in holdouts) == 3

    v2 = default_runtime_receive_v2_holdouts()
    assert len(v2) == 6
    assert len({context.context_hash for context, _ in v2}) == 6
    assert {context.context_hash for context, _ in holdouts}.isdisjoint(
        context.context_hash for context, _ in v2
    )
    assert all(context.case_id.startswith("s135.holdout.v2.") for context, _ in v2)

    v3 = default_runtime_receive_v3_holdouts()
    assert len(v3) == 6
    assert len({context.context_hash for context, _ in v3}) == 6
    prior = {context.context_hash for context, _ in (*holdouts, *v2)}
    assert prior.isdisjoint(context.context_hash for context, _ in v3)
    assert all(context.case_id.startswith("s136.holdout.v3.") for context, _ in v3)


def test_shared_world_exposes_content_bound_runtime_receive_actor_only() -> None:
    parameters = inspect.signature(_simulate_shared_world).parameters

    assert "shooter_runtime_receive_actor_path" in parameters
    assert "shooter_runtime_receive_probe_action" in parameters
    assert "shooter_runtime_receive_action" not in parameters
    assert tuple(inspect.signature(validate_runtime_receive_exam).parameters) == ("path",)


def test_runtime_receive_timing_search_does_not_change_ready_stance() -> None:
    actions = runtime_receive_timing_actions()

    assert len(actions) == 8
    assert len({action.action_hash for action in actions}) == 8
    assert {action.stance_offset_x_m for action in actions} == {-0.08}
    assert {action.stance_offset_y_m for action in actions} == {-0.06}
    assert {action.arrival_alignment_tolerance_sec for action in actions} == {0.02}


def test_runtime_receive_direction_search_is_bounded_and_unique() -> None:
    actions = runtime_receive_direction_actions()

    assert len(actions) == 8
    assert len({action.action_hash for action in actions}) == 8
    assert {action.contact_policy_frame for action in actions} == {258}
    assert all(-0.04 <= action.foot_yaw_offset_rad <= 0.10 for action in actions)


def test_pre_action_leakage_discovery_is_not_trainable(tmp_path: Path) -> None:
    payload = {
        "schema_version": "rosclaw_soccer.runtime_receive_discovery.v1",
        "implementation_hash": "sha256:" + "a" * 64,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["report_hash"] = hash_json(payload)
    path = tmp_path / "discovery-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="authority is invalid"):
        _bound_discovery(path)


def test_low_level_contact_curriculum_keeps_upper_layer_fixed() -> None:
    cases = default_runtime_receive_holdouts()[:2]
    probes = default_runtime_receive_contact_probes(cases)

    assert len(probes) == 16
    assert len({probe.probe_hash for probe in probes}) == 16
    assert {probe.action.contact_policy_frame for probe in probes} == {258}
    assert {probe.action.stance_offset_y_m for probe in probes} == {-0.06}
    assert {probe.action.target_foot_velocity_xyz_mps[1] for probe in probes} == {
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
    }

    countersteer = runtime_receive_contact_countersteer_probes(cases)
    assert len(countersteer) == 16
    assert {probe.action.target_foot_velocity_xyz_mps[1] for probe in countersteer} == {
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    }
    assert tuple(inspect.signature(validate_multitarget_neural_contact_canary).parameters) == (
        "path",
    )


def test_rejected_low_level_teacher_cannot_train_neural_actor(tmp_path: Path) -> None:
    payload = {
        "schema_version": "rosclaw_soccer.runtime_receive_contact_teacher.v1",
        "status": "REJECTED_RUNTIME_RECEIVE_CONTACT_TEACHER",
        "teacher_role": "SIM_ONLY_LOW_LEVEL_CONTACT_DATA_GENERATOR",
        "rows": [{}] * 16,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["report_hash"] = hash_json(payload)
    path = tmp_path / "teacher-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="requires intact teacher responses"):
        _bound_teacher_report(path)
