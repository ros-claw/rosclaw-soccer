from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    simulate_independent_team_world,
)
from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorObservation,
    TeamMotorPhysicsObservation,
    TeamMotorTarget,
    motor_blocks_residual,
)


def observation():
    return TeamMotorObservation(
        "red.finisher", 0, 0.0, "shoot", True, (0.0,) * 43, (0.0,) * 41, (7.5, 0.0, 0.0)
    )


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize(
    "proposed,faulted", [(False, False), (True, False), (False, True), (True, True)]
)
def test_motor_residual_ownership_is_phase_exclusive_and_fault_latched(allow, proposed, faulted):
    assert motor_blocks_residual(
        registered=True, proposed=proposed, faulted=faulted, allow_idle_fallback=allow
    ) is (not allow or proposed or faulted)
    assert not motor_blocks_residual(
        registered=False, proposed=False, faulted=False, allow_idle_fallback=allow
    )


def test_motor_residual_fallback_is_explicit_content_bound_and_requires_both_backends():
    default = IndependentTeamWorldConfig()
    enabled = replace(default, motor_idle_residual_fallback=True)
    assert default.config_hash != enabled.config_hash
    with pytest.raises(ValueError):
        replace(default, motor_idle_residual_fallback=1)
    with pytest.raises(ValueError, match="per-player motor"):
        simulate_independent_team_world(
            asset_root=Path("must-not-be-opened"),
            roster=SimpleNamespace(agents=[]),
            cells=(),
            players=(),
            scenario=None,
            goal=None,
            config=enabled,
        )
    with pytest.raises(ValueError):
        motor_blocks_residual(
            registered=False, proposed=True, faulted=False, allow_idle_fallback=True
        )
    with pytest.raises(ValueError):
        motor_blocks_residual(registered=True, proposed=False, faulted=False, allow_idle_fallback=1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("qpos", [0.0] * 43),
        ("qvel", (0.0,) * 40),
        ("frame", True),
        ("time_sec", float("nan")),
        ("prospective_owner", 1),
        ("intent", "real"),
    ],
)
def test_motor_observation_is_bounded_immutable(key, value):
    with pytest.raises(ValueError):
        replace(observation(), **{key: value})


@pytest.mark.parametrize(
    "key,value",
    [
        ("target_rad", (float("inf"),) * 29),
        ("kp", (301.0,) * 29),
        ("kd", (-1.0,) * 29),
        ("kp", np.zeros(29)),
        ("target_rad", (0.0,) * 28),
    ],
)
def test_motor_proposal_cannot_bypass_gain_or_numeric_envelope(key, value):
    valid = TeamMotorTarget((0.0,) * 29, (50.0,) * 29, (1.0,) * 29)
    with pytest.raises(ValueError):
        replace(valid, **{key: value})


def test_shared_player_history_rejected_before_assets_or_physics():
    motor = SimpleNamespace(contract_hash="sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="per-player motor"):
        simulate_independent_team_world(
            asset_root=Path("must-not-be-opened"),
            roster=SimpleNamespace(
                agents=[SimpleNamespace(agent_id=x) for x in ("red.a", "blue.a")]
            ),
            cells=(),
            players=(),
            scenario=None,
            goal=None,
            motor_options={"red.a": motor, "blue.a": motor},
        )


@pytest.mark.parametrize("value", [True, float("nan"), 0.19, 0.51, "0.35"])
def test_moving_entry_standoff_is_explicit_and_bounded(value):
    with pytest.raises(ValueError):
        IndependentTeamWorldConfig(motor_approach_standoff_m=value)
    assert IndependentTeamWorldConfig().motor_approach_standoff_m is None


def test_physics_observation_rejects_mutable_or_nonfinite_evidence():
    good = TeamMotorPhysicsObservation(0.002, (0.0,) * 43, (0.0,) * 41, True, 2.0, 0.0)
    for key, value in [
        ("qpos", [0.0] * 43),
        ("world_bodies_safe", 1),
        ("foot_normal_force_n", float("nan")),
        ("time_sec", -1.0),
    ]:
        with pytest.raises(ValueError):
            replace(good, **{key: value})


@pytest.mark.parametrize("failure", ["none", "observer", "force"])
def test_physics_observer_is_read_only_and_fault_removes_override(monkeypatch, failure):
    import mujoco

    from rosclaw_soccer.skills.team.independent_team_world import _observe_team_motor_physics

    class Observer:
        def __init__(self):
            self.seen = []

        def observe_physics(self, value):
            self.seen.append(value)
            if failure == "observer":
                raise ValueError("evidence consumer failed")

    observer = Observer()
    q = np.zeros(43)
    q[2] = 0.75
    q[3] = q[39] = 1
    d = SimpleNamespace(
        qpos=q, qvel=np.zeros(41), time=0.002, ncon=1, contact=[SimpleNamespace(geom1=9, geom2=5)]
    )
    model = SimpleNamespace(
        jnt_range=np.tile([-10.0, 10.0], (29, 1)),
        jnt_limited=np.ones(29),
        geom=lambda _: SimpleNamespace(name="foot"),
    )
    player = SimpleNamespace(
        cell=SimpleNamespace(agent_id="red.a"),
        qpos_base=0,
        qvel_base=0,
        joint_qpos=np.arange(7, 36),
        joint_qvel=np.arange(6, 35),
        joint_ids=np.arange(29),
        left_foot_geoms={5},
        right_foot_geoms={6},
    )

    def force(model, data, index, wrench):
        wrench[0] = float("nan") if failure == "force" else 2.0

    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    targets = {"red.a": TeamMotorTarget((0.0,) * 29, (50.0,) * 29, (1.0,) * 29)}
    faults = set()
    saved = q.copy()
    _observe_team_motor_physics(
        model, d, (player,), {"red.a": observer}, targets, faults, 9, 36, 35
    )
    np.testing.assert_array_equal(q, saved)
    if failure == "none":
        assert targets and not faults and len(observer.seen) == 1
        evidence = observer.seen[0]
        assert evidence.world_bodies_safe and evidence.foot_normal_force_n == 2.0
        assert evidence.other_non_ground_normal_force_n == 0.0
        with pytest.raises(TypeError):
            evidence.qpos[0] = 9
    else:
        assert not targets and faults == {"red.a"}
