from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorFoundation,
    TeamMotorObservation,
    TeamMotorTarget,
)
from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic

POLICY = "sha256:" + "a" * 64
CONFIG = "sha256:" + "b" * 64


def observation(frame=415, time=8.3):
    q = np.zeros(43)
    q[2] = 0.78
    q[3] = q[39] = 1
    q[36:39] = [0.8, 0.1, 0.115]
    return TeamMotorObservation(
        agent_id="blue.playmaker",
        frame=frame,
        time_sec=time,
        intent="other",
        prospective_owner=False,
        qpos=tuple(float(x) for x in q),
        qvel=(0.0,) * 41,
        target_position_m=(0.0, 0.0, 0.0),
        foundation=TeamMotorFoundation(
            "blue.playmaker",
            frame,
            TeamMotorTarget((0.25,) * 29, (50.0,) * 29, (1.0,) * 29),
            (0.0,) * 29,
            POLICY,
            CONFIG,
        ),
    )


def motor(tmp_path, nonzero=False):
    torch = pytest.importorskip("torch")
    actor = build_ball_residual_actor_critic()
    if nonzero:
        with torch.no_grad():
            actor.actor[-1].bias.fill_(2.0)
    path = tmp_path / "receiver.npz"
    np.savez_compressed(path, **{k: v.detach().numpy() for k, v in actor.state_dict().items()})
    return G1RecurrentReceiver(
        path,
        agent_id="blue.playmaker",
        expected_actor_hash=hash_bytes(path.read_bytes()),
        foundation_hash=POLICY,
        foundation_config_hash=CONFIG,
    )


def test_zero_residual_preserves_exact_representable_foundation_for_100_frames(tmp_path):
    m = motor(tmp_path)
    first = observation()
    m.begin_skill(first)
    for i in range(100):
        o = observation(415 + i, 8.3 + 0.02 * i)
        assert m.propose(o) == o.foundation.target
        assert m.last_observation.shape == (1, 133)
    with pytest.raises(ValueError, match="consecutive"):
        m.propose(observation(515, 10.3))
    with pytest.raises(ValueError, match="already used or faulted"):
        m.begin_skill(observation(516, 10.32))


@pytest.mark.parametrize("bad", ["skip", "duplicate", "clock", "hash", "config", "missing"])
def test_fault_latches_without_implicit_retry_or_filter_restart(tmp_path, bad):
    m = motor(tmp_path)
    m.begin_skill(observation())
    m.propose(observation())
    o = observation(416, 8.32)
    if bad == "skip":
        o = observation(417, 8.34)
    elif bad == "duplicate":
        o = observation()
    elif bad == "clock":
        o = observation(416, 8.34)
    elif bad in {"hash", "config"}:
        field = "policy_hash" if bad == "hash" else "configuration_hash"
        o = replace(o, foundation=replace(o.foundation, **{field: "sha256:" + "c" * 64}))
    else:
        o = replace(o, foundation=None)
    with pytest.raises(ValueError):
        m.propose(o)
    with pytest.raises(ValueError, match="unfaulted"):
        m.propose(observation(416, 8.32))


def test_nonzero_residual_is_bounded_and_preserves_shared_gains(tmp_path):
    m = motor(tmp_path, nonzero=True)
    m.begin_skill(observation())
    previous = np.zeros(29)
    for i in range(100):
        o = observation(415 + i, 8.3 + 0.02 * i)
        proposal = m.propose(o)
        residual = np.asarray(proposal.target_rad) - 0.25
        assert abs(residual).max() <= 0.2500001
        assert abs(residual - previous).max() <= 0.0250001
        assert proposal.kp == o.foundation.target.kp
        assert proposal.kd == o.foundation.target.kd
        previous = residual
    assert previous.min() > 0.20


def test_backend_failure_uses_shared_world_fault_boundary(tmp_path, monkeypatch):
    m = motor(tmp_path)
    m.begin_skill(observation())

    def broken(*args):
        raise RuntimeError("declared backend failure")

    monkeypatch.setattr(m._actor, "forward", broken)
    with pytest.raises(ValueError, match="receiver inference failed"):
        m.propose(observation())
    with pytest.raises(ValueError, match="unfaulted"):
        m.propose(observation())


@pytest.mark.parametrize("bad", ["player", "frame", "mutable", "nonfinite", "hash"])
def test_foundation_snapshot_is_immutable_finite_and_player_tick_bound(bad):
    o = observation()
    if bad == "player":
        with pytest.raises(ValueError):
            replace(o, foundation=replace(o.foundation, agent_id="red.playmaker"))
    elif bad == "frame":
        with pytest.raises(ValueError):
            replace(o, foundation=replace(o.foundation, frame=414))
    else:
        values = {
            "mutable": {"default_angles": [0.0] * 29},
            "nonfinite": {"default_angles": (float("nan"),) * 29},
            "hash": {"configuration_hash": "not-a-hash"},
        }
        with pytest.raises(ValueError):
            replace(o.foundation, **values[bad])
