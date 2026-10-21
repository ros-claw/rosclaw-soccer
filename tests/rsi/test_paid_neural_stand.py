from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask


def fixture(monkeypatch, memory="carry"):
    """Flow-only motor/physics fixture, not learned or physical evidence."""
    import mujoco

    monkeypatch.setattr(mujoco, "mj_forward", lambda *_: None)
    q, v = np.zeros(43), np.zeros(41)
    q[2:4] = [0.78, 1]
    q[38:40] = [0.11, 1]
    d = SimpleNamespace(
        qpos=q,
        qvel=v,
        ctrl=np.zeros(29),
        qacc_warmstart=v.copy(),
        mocap_pos=np.array([[6.0, 0, 0.11]]),
        mocap_quat=np.array([[1.0, 0, 0, 0]]),
        qfrc_applied=v.copy(),
        xfrc_applied=np.zeros((33, 6)),
        act=np.zeros(0),
        time=0.0,
    )
    ep = SimpleNamespace(
        done=True,
        termination_reason="RECOVERY_ADMITTED",
        frame=20,
        clip=SimpleNamespace(length=20),
        ref_pos=np.ones(29, np.float32) * 0.1,
        last_action=np.zeros(29, np.float32),
        goal=d.mocap_pos[0].copy(),
        env=SimpleNamespace(model=None, data=d, goal_mocap_id=0, ball_pos=q[36:39]),
        offset=np.zeros(29),
        scale=np.ones(29),
        order=np.arange(29),
        layout=SimpleNamespace(canonical=lambda data: (data.qpos, data.qvel)),
    )

    def observation(env, motion, time, last, pose, velocity, angular, ball, goal, offset, obs_dim):
        assert motion is None and time == 0 and not velocity.any() and not angular.any()
        assert obs_dim is None
        obs = np.zeros(160, np.float32)
        obs[:29] = pose
        obs[125:154] = last
        return obs

    def integrate(neural):
        result = dict(
            native_qpos_before=q.copy(),
            native_qvel_before=v.copy(),
            target=ep.target.copy(),
            neural_control=neural,
            time_before=d.time,
            qpos=np.tile(q, (10, 1)),
            qvel=np.tile(v, (10, 1)),
            forces=np.zeros((10, 3)),
            torque=np.zeros((10, 29)),
            body_failed=False,
        )
        d.time += 0.02
        return result

    ep.build_observation = observation
    ep._integrate_target = integrate

    class Actor:
        def __call__(self, obs, state):
            return torch.zeros(1, 1, 29), tuple(t + 1 for t in state)

    calls = []
    task = NeuralStandingTask(
        ep,
        Actor(),
        state=(torch.ones(2, 1, 128), torch.ones(2, 1, 128)),
        memory_mode=memory,
        count_inference=lambda: calls.append(1),
    )
    return ep, task, calls


@pytest.mark.parametrize("memory,initial", [("carry", 1.0), ("reset", 0.0)])
def test_static_task_keeps_body_and_private_memory_live_without_motion_padding(
    monkeypatch, memory, initial
):
    ep, task, calls = fixture(monkeypatch, memory)
    raw, row = task.collect(100)
    assert raw["NN_h_before"][0, 0, 0, 0] == initial
    assert raw["NN_h_after"][-1, 0, 0, 0] == initial + 100
    assert len(calls) == 100 and row["neural_rows"] == 100 and row["healthy"]
    assert ep.frame == 20 and ep.clip.length == 20
    assert ep.termination_reason == "REFERENCE_COMPLETE"
    assert raw["neural_control"].all()
    assert np.array_equal(raw["NN_obs"][:, 125:154], np.zeros((100, 29), np.float32))


def test_own_context_continues_during_bounded_settling_instead_of_holding_target(monkeypatch):
    ep, task, calls = fixture(monkeypatch)
    first, _ = task.collect(100)
    second, _ = task.collect(50, until_ball_settled=True)
    assert len(calls) == 101 and second["STAND_control"].tolist() == [100]
    assert np.array_equal(second["NN_h_before"][0], first["NN_h_after"][-1])
    with pytest.raises(ValueError):
        task.collect(50)


@pytest.mark.parametrize("fault", ["clock", "frame", "action", "failure", "goal"])
def test_stale_standing_owner_cannot_replay_a_private_motor_context(monkeypatch, fault):
    ep, task, calls = fixture(monkeypatch)
    if fault == "clock":
        ep.env.data.time = 0.02
    if fault == "frame":
        ep.frame = 0
    if fault == "action":
        ep.last_action[0] = 1
    if fault == "failure":
        ep.termination_reason = "BODY_FAILURE"
    if fault == "goal":
        ep.goal[0] = 5
    with pytest.raises(ValueError, match="expired"):
        task.step()
    assert not calls
