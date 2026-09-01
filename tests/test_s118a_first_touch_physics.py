from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from rosclaw_soccer.growth.first_touch import FirstTouchFailure
from rosclaw_soccer.providers.g1.asset_qualification import G1AssetQualification
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult
from rosclaw_soccer.training.first_touch_physics import (
    FirstTouchCandidate,
    FirstTouchPhysicsScenario,
    measure_first_touch_trajectory,
)


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _qualification() -> G1AssetQualification:
    return G1AssetQualification(
        eligible=True,
        asset_root=Path("/external/g1"),
        body_hash=_hash("a"),
        kick_prior_hash=_hash("b"),
        motion_hash=_hash("c"),
        backend_commit="1" * 40,
        actuator_count=29,
        joint_names=tuple(f"joint_{index}" for index in range(29)),
        policy_input_size=547,
        policy_output_size=29,
        errors=(),
    )


def _result(**overrides: object) -> G1SharedWorldResult:
    values: dict[str, object] = {
        "finite_state": True,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "shooter_post_kick_fall": False,
        "physics_steps": 4_000,
    }
    values.update(overrides)
    return cast(G1SharedWorldResult, SimpleNamespace(**values))


def _trace(*, speed: float = 1.2, pelvis_height: float = 0.72) -> dict[str, np.ndarray]:
    time = np.arange(41, dtype=np.float64) * 0.02
    contact_index = 10
    ball_pose = np.zeros((time.size, 7), dtype=np.float64)
    ball_pose[:, 3] = 1.0
    ball_velocity = np.zeros((time.size, 6), dtype=np.float64)
    ball_velocity[:contact_index, 0] = -1.0
    ball_velocity[contact_index:, 0] = speed
    ball_pose[: contact_index + 1, 0] = 1.2 - time[: contact_index + 1]
    ball_pose[contact_index:, 0] = ball_pose[contact_index, 0] + speed * (
        time[contact_index:] - time[contact_index]
    )
    ball_pose[:, 2] = 0.115
    contact_role = np.zeros(time.size, dtype=np.int64)
    contact_role[contact_index : contact_index + 2] = 2
    contact_foot = np.zeros(time.size, dtype=np.int64)
    contact_foot[contact_index : contact_index + 2] = -1
    pelvis = np.zeros((time.size, 7), dtype=np.float64)
    pelvis[:, 2] = pelvis_height
    pelvis[:, 3] = 1.0
    torso = np.zeros((time.size, 4), dtype=np.float64)
    torso[:, 0] = 1.0
    return {
        "time": time,
        "ball_pose": ball_pose,
        "ball_velocity": ball_velocity,
        "ball_contact_role": contact_role,
        "shooter_ball_contact_foot": contact_foot,
        "shooter_pelvis_pose": pelvis,
        "shooter_torso_quaternion": torso,
    }


def test_physics_measurement_accepts_a_controlled_anatomical_touch() -> None:
    scenario = FirstTouchPhysicsScenario(
        scenario_id="s118a.synthetic.pass",
        incoming_speed_mps=1.0,
    )
    candidate = FirstTouchCandidate(candidate_id="candidate.left", kick_foot="left")

    measurement, evaluation = measure_first_touch_trajectory(
        trace=_trace(),
        result=_result(),
        scenario=scenario,
        candidate=candidate,
        qualification=_qualification(),
    )

    assert measurement.contact_detected
    assert measurement.selected_foot == "left"
    assert measurement.incoming_speed_mps == pytest.approx(1.0)
    assert measurement.outgoing_speed_mps == pytest.approx(1.2)
    assert measurement.target_error_m == pytest.approx(0.0)
    assert measurement.next_action_latency_sec == pytest.approx(0.04)
    assert evaluation.passed


def test_physics_measurement_attributes_wrong_foot_hard_touch_and_balance() -> None:
    trace = _trace(speed=5.0, pelvis_height=0.50)
    trace["shooter_ball_contact_foot"][10:12] = 1
    scenario = FirstTouchPhysicsScenario(
        scenario_id="s118a.synthetic.failure",
        incoming_speed_mps=1.0,
    )
    candidate = FirstTouchCandidate(candidate_id="candidate.left", kick_foot="left")

    _, evaluation = measure_first_touch_trajectory(
        trace=trace,
        result=_result(),
        scenario=scenario,
        candidate=candidate,
        qualification=_qualification(),
    )

    assert evaluation.primary_failure is FirstTouchFailure.WRONG_FOOT
    assert FirstTouchFailure.LOST_BALANCE in evaluation.all_failures
    assert FirstTouchFailure.TOUCH_TOO_HARD in evaluation.all_failures
    assert FirstTouchFailure.TOO_SLOW_TO_NEXT_ACTION in evaluation.all_failures


def test_physics_measurement_forces_unsafe_runtime_into_balance_failure() -> None:
    scenario = FirstTouchPhysicsScenario(
        scenario_id="s118a.synthetic.runtime-failure",
        incoming_speed_mps=1.0,
    )
    candidate = FirstTouchCandidate(candidate_id="candidate.left", kick_foot="left")

    measurement, evaluation = measure_first_touch_trajectory(
        trace=_trace(),
        result=_result(torque_limit_violation=True),
        scenario=scenario,
        candidate=candidate,
        qualification=_qualification(),
    )

    assert measurement.minimum_pelvis_height_m < 0.62
    assert FirstTouchFailure.LOST_BALANCE in evaluation.all_failures


def test_late_foot_contact_cannot_erase_an_earlier_body_contact() -> None:
    trace = _trace()
    trace["ball_contact_role"][8] = 2
    trace["shooter_ball_contact_foot"][8] = 0
    scenario = FirstTouchPhysicsScenario(
        scenario_id="s118a.synthetic.body-first",
        incoming_speed_mps=1.0,
    )
    candidate = FirstTouchCandidate(candidate_id="candidate.left", kick_foot="left")

    measurement, evaluation = measure_first_touch_trajectory(
        trace=trace,
        result=_result(),
        scenario=scenario,
        candidate=candidate,
        qualification=_qualification(),
    )

    assert not measurement.contact_detected
    assert evaluation.primary_failure is FirstTouchFailure.TOUCH_TOO_SOFT


def test_first_touch_scenario_and_candidate_reject_out_of_scope_values() -> None:
    scenario = FirstTouchPhysicsScenario(
        scenario_id="s118a.valid",
        incoming_speed_mps=0.9,
    )
    assert scenario.launcher_velocity_mps == (-0.9, 0.0, 0.0)
    with pytest.raises(ValueError, match="identity"):
        replace(scenario, scenario_id="BAD ID")
    with pytest.raises(ValueError, match="acquisition speed"):
        replace(scenario, incoming_speed_mps=3.0)
    with pytest.raises(ValueError, match="swing_amplitude"):
        FirstTouchCandidate(candidate_id="candidate.bad", swing_amplitude=0.2)
    with pytest.raises(ValueError, match="start delay"):
        FirstTouchCandidate(candidate_id="candidate.bad", receiver_start_delay_sec=1.1)
