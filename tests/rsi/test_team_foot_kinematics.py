"""Read-only shared-world foot geometry is identity-bound and correctly sliced."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.skills.team.foot_kinematics import (
    TeamFootKinematics,
    measure_team_foot_kinematics,
)
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture


def _sample(agent_id: str = "red.finisher", frame: int = 30) -> TeamFootKinematics:
    return TeamFootKinematics(
        agent_id=agent_id,
        frame=frame,
        foot_position_world_m=((1.0, 0.1, 0.2), (1.0, -0.1, 0.1)),
        foot_linear_jacobian_world=(
            ((0.0,) * 6, (0.0,) * 6, (0.0,) * 6),
            ((0.0,) * 6, (0.0,) * 6, (0.0,) * 6),
        ),
        leg_joint_limits_rad=(
            ((-1.0, 1.0),) * 6,
            ((-1.0, 1.0),) * 6,
        ),
    )


def test_motor_observation_rejects_stale_or_foreign_foot_geometry() -> None:
    kwargs = dict(
        agent_id="red.finisher",
        frame=30,
        time_sec=0.6,
        intent="other",
        prospective_owner=False,
        qpos=(0.0,) * 43,
        qvel=(0.0,) * 41,
        target_position_m=(0.0, 0.0, 0.0),
    )
    assert TeamMotorObservation(**kwargs, foot_kinematics=_sample()).foot_kinematics is not None
    with pytest.raises(ValueError, match="another player or frame"):
        TeamMotorObservation(**kwargs, foot_kinematics=_sample("blue.finisher"))
    with pytest.raises(ValueError, match="another player or frame"):
        TeamMotorObservation(**kwargs, foot_kinematics=_sample(frame=29))


def test_foot_geometry_rejects_nonfinite_or_inverted_limits() -> None:
    original = _sample()
    with pytest.raises(ValueError, match="immutable finite"):
        TeamFootKinematics(
            agent_id=original.agent_id,
            frame=original.frame,
            foot_position_world_m=((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0)),
            foot_linear_jacobian_world=original.foot_linear_jacobian_world,
            leg_joint_limits_rad=original.leg_joint_limits_rad,
        )
    with pytest.raises(ValueError, match="immutable finite"):
        TeamFootKinematics(
            agent_id=original.agent_id,
            frame=original.frame,
            foot_position_world_m=original.foot_position_world_m,
            foot_linear_jacobian_world=original.foot_linear_jacobian_world,
            leg_joint_limits_rad=original.leg_joint_limits_rad,
            foot_linear_velocity_world_mps=((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
    with pytest.raises(ValueError, match="finite player-bound"):
        TeamFootKinematics(
            agent_id=original.agent_id,
            frame=original.frame,
            foot_position_world_m=original.foot_position_world_m,
            foot_linear_jacobian_world=original.foot_linear_jacobian_world,
            leg_joint_limits_rad=(((1.0, -1.0),) * 6, ((-1.0, 1.0),) * 6),
        )
    with pytest.raises(ValueError, match="immutable finite"):
        TeamFootKinematics(
            agent_id=original.agent_id,
            frame=original.frame,
            foot_position_world_m=[[1.0, 0.1, 0.2], [1.0, -0.1, 0.1]],  # type: ignore[arg-type]
            foot_linear_jacobian_world=original.foot_linear_jacobian_world,
            leg_joint_limits_rad=original.leg_joint_limits_rad,
        )


def test_mujoco_jacobian_is_projected_to_own_six_leg_dofs(monkeypatch: pytest.MonkeyPatch) -> None:
    import mujoco

    model = SimpleNamespace(nbody=2, nv=20)
    data = SimpleNamespace(
        xpos=np.asarray(((1.0, 0.2, 0.3), (1.0, -0.2, 0.4))),
        qvel=np.full(20, 0.1),
    )

    def fake_jac(_model, _data, linear, angular, body):
        linear[:] = np.arange(60).reshape(3, 20) + body * 100
        angular[:] = 0

    monkeypatch.setattr(mujoco, "mj_jacBody", fake_jac)
    measured = measure_team_foot_kinematics(
        model=model,
        data=data,
        agent_id="red.finisher",
        frame=30,
        ankle_body_ids=(0, 1),
        leg_dof_ids=((6, 7, 8, 9, 10, 11), (12, 13, 14, 15, 16, 17)),
        leg_joint_ranges=(((-1.0, 1.0),) * 6, ((-1.0, 1.0),) * 6),
    )
    assert measured.foot_linear_jacobian_world[0][0] == (6, 7, 8, 9, 10, 11)
    assert measured.foot_linear_jacobian_world[1][0] == (112, 113, 114, 115, 116, 117)
    assert measured.foot_position_world_m[1] == (1.0, -0.2, 0.4)
    assert measured.foot_linear_velocity_world_mps is not None
    assert measured.foot_linear_velocity_world_mps[0][0] == pytest.approx(
        np.dot(np.arange(20), np.full(20, 0.1))
    )


@pytest.mark.integration
def test_read_only_kinematics_request_preserves_shared_world_physics() -> None:
    root_text = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if root_text is None:
        pytest.skip("qualified external RoboNaldo asset root required")
    root = Path(root_text)
    fixture = build_independent_three_vs_three_fixture(root)
    agent = fixture.players[0].agent_id

    class Observer:
        contract_hash = "sha256:" + "1" * 64

        def __init__(self, needs: bool) -> None:
            self.needs_foot_kinematics = needs
            self.snapshots = []

        def propose(self, observation: TeamMotorObservation):
            self.snapshots.append(observation.foot_kinematics)
            return None

    traces = []
    observers = []
    for needs in (False, True):
        observer = Observer(needs)
        _, trace = simulate_independent_team_world(
            asset_root=root,
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=IndependentTeamWorldScenario(
                "s199.readonly-foot-geometry",
                (2.0, 0.0, fixture.goal.ball_radius_m),
                (0.2, 0.0, 0.0),
                920601,
            ),
            goal=fixture.goal,
            config=IndependentTeamWorldConfig(simulation_duration_sec=5.0),
            motor_options={agent: observer},
        )
        observers.append(observer)
        traces.append(trace)
    assert observers[0].snapshots and all(value is None for value in observers[0].snapshots)
    assert observers[1].snapshots and all(
        value is not None
        and value.agent_id == agent
        and value.frame == frame
        and value.foot_linear_velocity_world_mps is not None
        for frame, value in enumerate(observers[1].snapshots)
    )
    for key in traces[0]:
        np.testing.assert_array_equal(traces[0][key], traces[1][key])
