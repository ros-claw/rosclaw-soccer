import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.keeper_frame import KeeperFrame
from rosclaw_soccer.providers.g1.shared_keeper_reach import (
    SharedKeeperReach,
    SharedKeeperReachConfig,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    _run_locomotion,
)


def test_configuration_keeps_disabled_hash_and_rejects_invalid_authority():
    config = IndependentTeamWorldConfig()
    previous = asdict(config)
    previous.pop("keeper_reach")
    previous.pop("glove_material")
    assert config.config_hash == hash_json(previous)
    with pytest.raises(ValueError):
        SharedKeeperReachConfig(gain_scale=float("nan"))
    with pytest.raises(ValueError):
        SharedKeeperReachConfig(gmt_model_path="model.onnx")
    with pytest.raises(ValueError):
        IndependentTeamWorldConfig(keeper_reach=True)


def test_foundation_observation_ablation_preserves_sensor_state_even_on_error():
    q, dq = np.arange(29, dtype=float), np.ones(29)
    state = SimpleNamespace(q=q.copy(), dq=dq.copy(), gravity_ori=np.array((0.1, 0.2, -0.97)))

    def run():
        np.testing.assert_array_equal(state.q[:15], q[:15])
        np.testing.assert_array_equal(state.q[15:], np.zeros(14))
        np.testing.assert_array_equal(state.dq[15:], np.zeros(14))
        np.testing.assert_array_equal(state.gravity_ori, (0.1, 0.2, -0.97))
        raise RuntimeError("test inference failure")

    keeper = SimpleNamespace(
        config=SharedKeeperReachConfig(neutralize_foundation_arm_observation=True)
    )
    controller = SimpleNamespace(
        state=state,
        keeper_reach=keeper,
        policy=SimpleNamespace(default_angles_reorder=np.zeros(29), run=run),
    )
    with pytest.raises(RuntimeError, match="test inference failure"):
        _run_locomotion(controller, mirror=False)
    np.testing.assert_array_equal(state.q, q)
    np.testing.assert_array_equal(state.dq, dq)
    assert controller.keeper_reach is keeper


def test_physics_contact_latch_is_once_per_epoch_and_frame_bounded():
    keeper = SimpleNamespace(
        last_time=1.0, active=True, contact_time=None, contact_target=None, last_output=np.ones(29)
    )
    SharedKeeperReach.notify_glove_contact(keeper, 1.002)
    assert keeper.contact_time == 1.002
    keeper.last_output[:] = 9
    SharedKeeperReach.notify_glove_contact(keeper, 1.012)
    assert keeper.contact_time == 1.002
    np.testing.assert_array_equal(keeper.contact_target, np.ones(29))
    for invalid in (float("nan"), 0.98, 1.022):
        with pytest.raises(ValueError):
            SharedKeeperReach.notify_glove_contact(keeper, invalid)


def test_bilateral_adapter_is_arm_only_private_causal_and_recovers():
    assets = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not assets:
        pytest.skip("requires actual G1 MuJoCo assets")
    import mujoco

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
    from rosclaw_soccer.training.goalkeeper_targeted_dive_exam import _ready_control
    from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

    root = Path(assets)
    fixture = build_four_vs_four_fixture(root)
    model = build_g1_multi_player_stadium_model(root, players=fixture.players, spec=fixture.goal)
    ready, _, _, _ = _ready_control()
    outputs = []
    for player in fixture.players:
        if not player.goalkeeper_gloves:
            continue
        data = mujoco.MjData(model)
        frame = KeeperFrame(player.origin_m[:2], player.yaw_rad)
        controller = SharedKeeperReach(
            asset_root=root, goal=fixture.goal, frame=frame, prefix=player.body_prefix
        )
        q = int(model.jnt_qposadr[model.joint(player.body_prefix + "floating_base_joint").id])
        data.qpos[q : q + 7] = (
            *player.origin_m[:2],
            0.793,
            np.cos(player.yaw_rad / 2),
            0,
            0,
            np.sin(player.yaw_rad / 2),
        )
        for name, value in zip(G1_DDS_JOINT_NAMES, ready, strict=True):
            j = model.joint(player.body_prefix + name).id
            data.qpos[model.jnt_qposadr[j]] = value
        bq = int(model.jnt_qposadr[model.joint("ball_free").id])
        values = []
        for tick in range(100):
            data.time = tick * 0.02
            canonical = (
                np.array((2.0 + tick * 0.08, 0.1, 1.35)) if tick < 30 else np.array((0, 0, 0.115))
            )
            data.qpos[bq : bq + 3] = np.array((*frame.origin_xy, 0)) + frame.world_vector(
                canonical - (4.52, 0, 0)
            )
            before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
            target = controller.step(model, data, ready)
            np.testing.assert_array_equal(target[:15], ready[:15])
            for prior, after in zip(before, (data.qpos, data.qvel, data.ctrl), strict=True):
                np.testing.assert_array_equal(prior, after)
            values.append(target)
        np.testing.assert_allclose(values[-1], ready, atol=1e-12)
        assert np.max(np.abs(np.asarray(values) - ready)) > 0.01
        assert not controller.active
        assert np.max(np.abs(np.diff(values, axis=0))) <= 0.100001
        with pytest.raises(ValueError, match="50 Hz"):
            controller.step(model, data, ready)
        outputs.append(values)
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-10)
