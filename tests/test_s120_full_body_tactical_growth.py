from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from rosclaw_soccer.growth.tactical_2v1 import TacticalAction
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.full_body_tactical_2v1 import (
    FullBodyTwoVsOneConfig,
    FullBodyTwoVsOneScenario,
    simulate_full_body_two_vs_one,
)
from rosclaw_soccer.training.full_body_tactical_growth import (
    FullBodyTwoVsOneRetentionManifest,
    default_full_body_acquisition_scenarios,
    default_full_body_retention_manifest,
)
from rosclaw_soccer.training.tactical_2v1_physics import FrozenTacticalSkillBundle


def _hash(label: str) -> str:
    return str(hash_json({"s120_fixture": label}))


def _bundle() -> FrozenTacticalSkillBundle:
    return FrozenTacticalSkillBundle(
        body_hash=_hash("body"),
        athlete_foundation_hash=_hash("foundation"),
        first_touch_actor_hash=_hash("first-touch"),
        pass_skill_hash=_hash("pass"),
        shoot_skill_hash=_hash("shoot"),
    )


def _fake_shared_world(
    *_: object, **kwargs: object
) -> tuple[SimpleNamespace, dict[str, np.ndarray]]:
    policy_target = cast(tuple[float, float, float], kwargs["shooter_policy_target"])
    is_pass = float(policy_target[2]) < 0.20
    time = np.asarray((0.0, 5.44, 6.04, 6.50), dtype=np.float64)
    ball_pose: NDArray[np.float64] = np.zeros((4, 7), dtype=np.float64)
    ball_pose[:, 0] = (1.25, 1.25, 5.50, 6.20) if is_pass else (1.25, 2.0, 7.60, 8.0)
    ball_pose[:, 1] = (0.0, 0.0, -0.40, -0.40) if is_pass else (0.0, -0.2, -0.8, -0.8)
    ball_pose[:, 2] = 0.115
    left_foot: NDArray[np.float64] = np.zeros((4, 3), dtype=np.float64)
    left_foot[2] = ball_pose[2, :3] + np.asarray((0.10, 0.0, 0.0))
    focal_present = bool(kwargs["passer_collision_enabled"])
    result = SimpleNamespace(
        shot_contact_observed=True,
        shot_contact_time_sec=5.44,
        pass_contact_observed=bool(is_pass and focal_present),
        pass_contact_time_sec=6.04 if is_pass and focal_present else None,
        goalkeeper_ball_contact_observed=False,
        goalkeeper_ball_contact_time_sec=None,
        goal_crossed=not is_pass,
        goal_crossing_y_m=-0.80 if not is_pass else None,
        goal_crossing_z_m=0.24 if not is_pass else None,
        finite_state=True,
        shooter_min_pelvis_height_m=0.68,
        shooter_roll_peak_rad=0.20,
        shooter_pitch_peak_rad=0.20,
        shooter_joint_limit_violation=False,
        goalkeeper_min_pelvis_height_m=0.74,
        goalkeeper_joint_limit_violation=False,
        torque_limit_violation=False,
        actuator_saturation=False,
        robot_robot_contact_count=40,
        passer_min_pelvis_height_m=0.74,
        passer_joint_limit_violation=False,
    )
    return result, {
        "time": time,
        "ball_pose": ball_pose,
        "passer_left_foot_position": left_foot,
        "passer_right_foot_position": left_foot + 0.20,
    }


def test_full_body_contracts_are_sealed_sim_only_and_disjoint() -> None:
    with pytest.raises(ValueError, match="safety envelope"):
        FullBodyTwoVsOneConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="curriculum"):
        FullBodyTwoVsOneScenario("s120.bad", 1, (5.5, -0.40, 0.0), (7.0, 0.4, 0.0))
    with pytest.raises(ValueError, match="sealed"):
        FullBodyTwoVsOneRetentionManifest(scenarios=default_full_body_acquisition_scenarios()[:4])
    acquisition = default_full_body_acquisition_scenarios()
    retention = default_full_body_retention_manifest()
    assert len(acquisition) == len(retention.scenarios) == 8
    assert {row.scenario_hash for row in acquisition}.isdisjoint(
        row.scenario_hash for row in retention.scenarios
    )
    assert retention.training_access_allowed is False


def test_full_body_wrapper_scores_real_contact_not_phase_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rosclaw_soccer.training.full_body_tactical_2v1.simulate_shared_world",
        _fake_shared_world,
    )
    scenario = default_full_body_acquisition_scenarios()[0]
    passed, pass_trace = simulate_full_body_two_vs_one(
        asset_root=Path("/unused"),
        scenario=scenario,
        action=TacticalAction.PASS,
        skill_bundle=_bundle(),
    )
    shot, _ = simulate_full_body_two_vs_one(
        asset_root=Path("/unused"),
        scenario=scenario,
        action=TacticalAction.SHOOT,
        skill_bundle=_bundle(),
    )
    ablated, ablated_trace = simulate_full_body_two_vs_one(
        asset_root=Path("/unused"),
        scenario=scenario,
        action=TacticalAction.PASS,
        skill_bundle=_bundle(),
        focal_teammate_present=False,
    )
    assert passed.pass_completed and passed.safe and passed.task_succeeded
    assert passed.teammate_foot_reception_distance_m == pytest.approx(0.10)
    assert shot.goal_scored and shot.safe and shot.task_succeeded
    assert not ablated.pass_completed and not ablated.task_succeeded
    assert np.all(pass_trace["focal_teammate_present"])
    assert not np.any(ablated_trace["focal_teammate_present"])
    assert passed.trajectory_hash != ablated.trajectory_hash


def test_full_body_state_uses_geometry_and_frozen_content() -> None:
    config = FullBodyTwoVsOneConfig()
    pass_state = default_full_body_acquisition_scenarios()[0].state(
        skill_bundle=_bundle(), config=config
    )
    shoot_state = default_full_body_acquisition_scenarios()[-1].state(
        skill_bundle=_bundle(), config=config
    )
    assert pass_state.carrier_pressure > shoot_state.carrier_pressure
    assert pass_state.world_state_hash != shoot_state.world_state_hash
    assert pass_state.frozen_skill_bundle_hash == _bundle().bundle_hash
    assert all(
        0.0 <= value <= 1.0
        for value in (
            pass_state.carrier_pressure,
            pass_state.teammate_lane_openness,
            pass_state.shot_lane_openness,
            pass_state.goal_progress,
            pass_state.teammate_progress,
        )
    )
