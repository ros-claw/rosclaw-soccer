from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rosclaw_soccer.training.age04_regulation import (
    Age04RegulationCurriculum,
    _config_from_json,
    assess_age04_regulation,
)


def test_regulation_curriculum_binds_eight_unique_probes() -> None:
    curriculum = Age04RegulationCurriculum()

    assert len(curriculum.teacher_force_pairs_n) == 8
    assert curriculum.teacher_force_pairs_n[:2] == ((250.0, 250.0), (245.0, 245.0))
    assert curriculum.precision_radius_m == 0.10
    assert (curriculum.target_y_m, curriculum.target_z_m) == (1.32, 1.04)
    assert curriculum.torque_authority_projection_ratio == 0.98
    assert curriculum.torque_authority_projection_max_fraction == 0.01
    assert (curriculum.aim_bias_y_m, curriculum.aim_bias_z_m) == (1.13, 0.20)
    assert curriculum.sonic_planner_seed == 0


def test_regulation_curriculum_rejects_duplicate_probe() -> None:
    with pytest.raises(ValueError, match="unique"):
        Age04RegulationCurriculum(teacher_force_pairs_n=((10.0, 10.0),) * 8)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("torque_authority_projection_ratio", 0.89, "authority"),
        ("torque_authority_projection_max_fraction", 0.051, "authority"),
        ("sonic_planner_seed", -1, "planner seed"),
    ],
)
def test_regulation_curriculum_rejects_invalid_contract(
    field: str, value: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Age04RegulationCurriculum(**{field: value})


def test_seed_config_discards_old_schema_and_restores_tuple_fields() -> None:
    @dataclass(frozen=True)
    class Example:
        joint_gain_scales: tuple[float, ...]
        schema_version: str = "new"

    result = _config_from_json(
        Example,
        {"joint_gain_scales": [0.5, 1.0], "schema_version": "old", "removed": True},
    )

    assert result == Example(joint_gain_scales=(0.5, 1.0))


def _evidence(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "finite_state": True,
        "learned_runup_executed": True,
        "ballistic_contact_impulse_actor_executed": True,
        "loft_teacher_executed": False,
        "continuous_single_world": True,
        "state_reset_after_start": False,
        "kick_contact_observed": True,
        "goal_crossed": True,
        "goal_mouth_hit": True,
        "goal_plane_target_error_m": 0.0081,
        "ball_retained_in_goal": True,
        "perceptual_continuity_passed": True,
        "initial_ball_distance_m": 4.4,
        "runup_distance_m": 3.37,
        "runup_peak_speed_mps": 1.84,
        "runup_min_pelvis_height_m": 0.659,
        "runup_peak_tilt_rad": 0.496,
        "runup_terminal_speed_mps": 0.396,
        "kick_min_pelvis_height_m": 0.69,
        "kick_peak_tilt_rad": 0.28,
        "post_kick_fall": False,
        "joint_limit_violation": False,
        "final_pelvis_height_m": 0.78,
        "final_speed_mps": 0.001,
        "post_contact_backward_displacement_m": 0.0,
        "post_contact_forward_velocity_reversals": 4,
        "post_contact_settling_time_sec": 3.34,
        "post_contact_final_joint_velocity_rms_rad_s": 0.001,
        "post_contact_mean_pelvis_speed_mps": 0.035,
        "post_contact_mean_joint_velocity_rms_rad_s": 0.106,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "actuator_saturation_steps": 0,
        "torque_authority_projection_qualified": True,
        "torque_authority_projection_fraction": 0.0058,
        "contact_task_authority_scale_min": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(
        strict_replay=True,
        learned_gait_qualification=SimpleNamespace(eligible=True),
        sonic_qualification=SimpleNamespace(eligible=True),
        activation_ceiling="SIM_ONLY",
        hardware_command_sent=False,
        result=SimpleNamespace(**values),
    )


def test_regulation_assessment_preserves_stability_development_boundary() -> None:
    assessment = assess_age04_regulation(_evidence(), Age04RegulationCurriculum())

    assert not assessment.passed
    assert assessment.verdict == "DEVELOPMENT"
    assert assessment.development_breakthrough
    assert assessment.failure_codes == ("DYNAMIC_STABILITY_GATE",)
    assert assessment.precision_passed
    assert assessment.torque_authority_passed


def test_regulation_assessment_passes_all_independent_axes() -> None:
    evidence = _evidence(runup_min_pelvis_height_m=0.72, runup_peak_tilt_rad=0.25)

    assessment = assess_age04_regulation(evidence, Age04RegulationCurriculum())

    assert assessment.passed
    assert assessment.verdict == "PASS"
    assert assessment.failure_codes == ()


@pytest.mark.parametrize(
    "overrides,failure_code",
    [
        ({"goal_plane_target_error_m": float("nan")}, "PRECISION_GATE"),
        ({"torque_authority_projection_fraction": 0.02}, "TORQUE_AUTHORITY_GATE"),
    ],
)
def test_regulation_assessment_fails_closed(
    overrides: dict[str, object], failure_code: str
) -> None:
    assessment = assess_age04_regulation(_evidence(**overrides), Age04RegulationCurriculum())

    assert not assessment.passed
    assert assessment.verdict == "REJECTED"
    assert failure_code in assessment.failure_codes
