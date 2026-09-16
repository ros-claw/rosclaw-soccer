"""Low-ball admission is opt-in; success and collision geometry are unchanged."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.shared_keeper_reach import SharedKeeperReachConfig
from rosclaw_soccer.sim.contracts import hash_json


@pytest.mark.parametrize(
    "height",
    [True, False, np.bool_(True), None, "0.3", float("nan"), float("inf"), -0.1, 0.19, 0.651],
)
def test_invalid_height_is_rejected(height):
    with pytest.raises(ValueError, match="interception height"):
        SharedKeeperReachConfig(minimum_intercept_height_m=height)


@pytest.mark.parametrize("height", [0.20, 0.30, 0.50, 0.65])
def test_height_is_explicitly_bound_to_config(height):
    config = SharedKeeperReachConfig(minimum_intercept_height_m=height)
    assert asdict(config)["minimum_intercept_height_m"] == height
    assert config.minimum_intercept_height_m == height
    if height != 0.65:
        assert hash_json(asdict(config)) != hash_json(asdict(SharedKeeperReachConfig()))


def test_legacy_height_remains_default():
    assert SharedKeeperReachConfig().minimum_intercept_height_m == 0.65
    assert SharedKeeperReachConfig().ballistic_airborne_velocity is False
    assert SharedKeeperReachConfig().minimum_reach_height_m == 0.72


@pytest.mark.parametrize(
    "height", [True, False, None, "0.3", float("nan"), float("inf"), 0.19, 0.721]
)
def test_invalid_reach_floor_is_rejected(height):
    with pytest.raises(ValueError, match="reach height"):
        SharedKeeperReachConfig(minimum_reach_height_m=height)


def test_reach_floor_changes_task_target_not_geometry_or_leg_authority(monkeypatch):
    import mujoco

    from rosclaw_soccer.skills.team.shared_world import (
        _apply_goalkeeper_bimanual_operational_space_reach,
    )

    model = SimpleNamespace(nv=29, jnt_range=np.tile([-3.0, 3.0], (29, 1)), jnt_limited=np.ones(29))
    data = SimpleNamespace(
        qpos=np.zeros(43),
        xmat=np.tile(np.eye(3).ravel(), (2, 1)),
        xpos=np.array([[0.0, -0.08, 0.9], [0.0, 0.08, 0.9]]),
    )
    original_solve = np.linalg.solve
    errors = []

    def jacobian(model, data, position, rotation, point, body):
        start = 15 if body == 0 else 22
        position[:, start : start + 3] = np.eye(3)

    def solve(matrix, error):
        errors.append(error.copy())
        return original_solve(matrix, error)

    monkeypatch.setattr(mujoco, "mj_jac", jacobian)
    monkeypatch.setattr(np.linalg, "solve", solve)
    targets = []
    before = (data.qpos.copy(), data.xmat.copy(), data.xpos.copy())
    for floor in (0.72, 0.30):
        robot = SimpleNamespace(
            goalkeeper_reach_memory=np.zeros(29),
            origin=np.zeros(3),
            world_from_local_quat=np.array([1.0, 0, 0, 0]),
            qpos_base=0,
            left_hand_body=0,
            right_hand_body=1,
            joint_qvel=np.arange(29),
            joint_ids=np.arange(29),
            last_target=np.zeros(29),
            goalkeeper_reach_memory_peak_rad=0.0,
        )
        artifact = SimpleNamespace(
            operational_space_memory_decay=0.96,
            operational_space_reach_damping=0.08,
            operational_space_reach_gain=0.25,
            operational_space_reach_maximum_step_rad=0.15,
            operational_space_reach_ramp_sec=0.18,
        )
        active = _apply_goalkeeper_bimanual_operational_space_reach(
            robot,
            model=model,
            data=data,
            observation=SimpleNamespace(
                intercept_confidence=1.0, estimated_intercept=(0.2, 0.0, 0.45)
            ),
            artifact=artifact,
            target_local_x_m=0.24,
            half_span_m=0.08,
            height_offset_m=0.0,
            reach_fraction=1.0,
            gain_scale=1.5,
            memory_decay=0.96,
            memory_maximum_rad=0.8,
            elapsed_sec=0.3,
            minimum_target_height_m=floor,
        )
        assert active
        np.testing.assert_array_equal(robot.last_target[:15], np.zeros(15))
        targets.append(robot.last_target.copy())
    assert len(errors) == 4
    for hand in range(2):
        assert errors[hand + 2][2] - errors[hand][2] == pytest.approx(-0.27)
    assert not np.array_equal(*targets)
    for initial, current in zip(before, (data.qpos, data.xmat, data.xpos), strict=True):
        np.testing.assert_array_equal(initial, current)
