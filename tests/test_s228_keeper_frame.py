import math
import os
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.keeper_frame import KeeperFrame, KeeperSnapshot


def test_frame_half_turn_is_rotation_not_joint_reflection():
    frame = KeeperFrame((-0.95, 0), 0)
    np.testing.assert_allclose(frame.point(np.asarray((-0.95, 0, 0.8))), (4.52, 0, 0.8))
    v = np.asarray((2.0, 0.3, -0.4))
    np.testing.assert_allclose(frame.world_vector(frame.rotation @ v), v, atol=1e-14)
    assert np.linalg.det(frame.rotation) == pytest.approx(1)
    with pytest.raises(ValueError):
        KeeperFrame((float("nan"), 0), 0)
    with pytest.raises(ValueError):
        frame._pose(np.zeros(7))


def test_snapshot_is_finite_detached_and_immutable():
    qpos = np.zeros(43)
    snapshot = KeeperSnapshot(qpos, np.zeros(41), 0)
    qpos[0] = 7
    assert snapshot.qpos[0] == 0
    with pytest.raises(ValueError):
        snapshot.qpos.flags.writeable = True
    with pytest.raises(ValueError):
        KeeperSnapshot(np.zeros(42), np.zeros(41), 0)


def test_actual_eight_g1_projection_preserves_fk_velocity_and_free_dynamics(record_property):
    assets = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not assets:
        pytest.skip("set ROSCLAW_G1_ASSET_ROOT for actual MuJoCo projection exam")
    import mujoco
    import torch

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
    from rosclaw_soccer.training.goalkeeper_targeted_dive_actor_exam import (
        _actor_observation,
        _load_actor,
    )
    from rosclaw_soccer.training.goalkeeper_targeted_dive_exam import _ready_control
    from rosclaw_soccer.world.field import build_g1_stadium_model
    from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

    root = Path(assets)
    fixture = build_four_vs_four_fixture(root)
    model = build_g1_multi_player_stadium_model(root, players=fixture.players, spec=fixture.goal)
    data = mujoco.MjData(model)
    canonical_model = build_g1_stadium_model(root, fixture.goal)
    ready, _, _, _ = _ready_control()
    actor = None
    actor_path = os.environ.get("ROSCLAW_G1_KEEPER_ACTOR")
    if actor_path:
        from rosclaw_soccer.sim.contracts import hash_bytes

        actor, _ = _load_actor(path=Path(actor_path), device=torch.device("cpu"))
        record_property("read_only_actor_hash", hash_bytes(Path(actor_path).read_bytes()))
    record_property("actor_forward_enabled", actor is not None)
    for player in fixture.players:
        joint = model.joint(player.body_prefix + "floating_base_joint").id
        q, v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        data.qpos[q : q + 7] = (
            *player.origin_m[:2],
            1.5,
            math.cos(player.yaw_rad / 2),
            0,
            0,
            math.sin(player.yaw_rad / 2),
        )
        data.qvel[v : v + 6] = (0.02, 0.01, 0.0, 0.03, -0.02, 0.01)
        for name, value in zip(G1_DDS_JOINT_NAMES, ready, strict=True):
            index = model.joint(player.body_prefix + name).id
            data.qpos[model.jnt_qposadr[index]] = value
    bq = int(model.jnt_qposadr[model.joint("ball_free").id])
    data.qpos[bq : bq + 7] = (3, -3, 1.5, 1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    before_q, before_v = data.qpos.copy(), data.qvel.copy()
    projections = []
    for player in fixture.players:
        if not player.agent_id.endswith("goalkeeper"):
            continue
        frame = KeeperFrame(player.origin_m[:2], player.yaw_rad)
        snapshot = frame.project(model, data, prefix=player.body_prefix)
        standalone = mujoco.MjData(canonical_model)
        standalone.qpos[:] = snapshot.qpos
        standalone.qvel[:] = snapshot.qvel
        mujoco.mj_forward(canonical_model, standalone)
        observation_arguments = dict(
            torch=torch,
            target=np.asarray((4.44, 0, 0.82)),
            cue=np.zeros(3),
            cue_visible=False,
            shot_active=False,
            previous_action=torch.zeros(30),
            previous_target=ready,
            step=0,
            first_end_step=50,
            episode_duration_sec=12.0,
            control_dt_sec=0.02,
        )
        projected_observation = _actor_observation(data=snapshot, **observation_arguments)
        native_observation = _actor_observation(data=standalone, **observation_arguments)
        assert projected_observation.shape == (1, 89)
        torch.testing.assert_close(projected_observation, native_observation, rtol=0, atol=1e-7)
        if actor is not None:
            with torch.inference_mode():
                projected_action = actor(projected_observation)[0]
                native_action = actor(native_observation)[0]
            assert projected_action.shape == (1, 30) and torch.isfinite(projected_action).all()
            torch.testing.assert_close(projected_action, native_action, rtol=0, atol=1e-7)
        for name in (
            "pelvis",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ):
            a = model.body(player.body_prefix + name).id
            b = canonical_model.body(name).id
            np.testing.assert_allclose(frame.point(data.xpos[a]), standalone.xpos[b], atol=1e-10)
            av, bv = np.zeros(6), np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, a, av, 0)
            mujoco.mj_objectVelocity(
                canonical_model, standalone, mujoco.mjtObj.mjOBJ_BODY, b, bv, 0
            )
            np.testing.assert_allclose(
                np.r_[frame.rotation @ av[:3], frame.rotation @ av[3:]], bv, atol=1e-10
            )
        projections.append((player, frame, standalone))
    np.testing.assert_array_equal(data.qpos, before_q)
    np.testing.assert_array_equal(data.qvel, before_v)
    assert data.ncon == 0  # Isolate rigid-frame dynamics, not differing goal collisions.
    mujoco.mj_step(model, data)
    for player, frame, standalone in projections:
        assert standalone.ncon == 0
        mujoco.mj_step(canonical_model, standalone)
        snapshot = frame.project(model, data, prefix=player.body_prefix)
        np.testing.assert_allclose(snapshot.qpos, standalone.qpos, atol=1e-8)
        np.testing.assert_allclose(snapshot.qvel, standalone.qvel, atol=1e-8)
