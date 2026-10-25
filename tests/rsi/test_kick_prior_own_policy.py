import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_kick_prior_actor import tensors  # noqa: E402

from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_logp  # noqa: E402
from rosclaw_soccer.rsi.kick_prior_actor import KickPriorActor  # noqa: E402
from rosclaw_soccer.rsi.kick_prior_own_policy import OwnKickPlayer, OwnKickSession  # noqa: E402

MUJOCO_TO_ISAAC = np.arange(29)[::-1]


class Policy:
    def __init__(self):
        self.action_clip_lo_il = np.full(29, -0.001, np.float32)
        self.action_clip_hi_il = np.full(29, 0.001, np.float32)
        self.default_q_mj = np.zeros(29, np.float32)
        self.action_scale_mj = np.full(29, 0.25, np.float32)


def test_own_draws_replay_raw_law_after_exact_pd_commit_not_clipped_density():
    torch.set_num_threads(1)
    session = OwnKickSession(KickPriorActor(tensors()), seed=41)
    policy = Policy()
    feed = {"obs": np.zeros((1, 547), np.float32)}
    for frame in range(5):
        raw = session.run(["actions"], feed)[0][0]
        assert (
            session.previous_action is None if frame == 0 else session.previous_action is not None
        )
        with pytest.raises(ValueError, match="physical commitment"):
            session.run(["actions"], feed)
        clipped = np.clip(raw, policy.action_clip_lo_il, policy.action_clip_hi_il)
        target = (policy.default_q_mj + policy.action_scale_mj * clipped[MUJOCO_TO_ISAAC]).astype(
            np.float64
        )
        session.commit_executed(policy, target, frame=frame + 100)
    assert session.pending is None and len(session.samples) == 5
    assert sum(s["clipped_components"] for s in session.samples) > 0
    means = torch.from_numpy(np.stack([s["mean"] for s in session.samples]))
    actions = torch.from_numpy(np.stack([s["raw_action"] for s in session.samples]))
    replay = conditional_logp(means, actions, session.log_std, rho=session.rho)
    np.testing.assert_allclose(
        replay.numpy(), [s["raw_logp"] for s in session.samples], atol=1e-5, rtol=0
    )


def test_wrong_executed_target_latches_fault():
    session = OwnKickSession(KickPriorActor(tensors()), seed=42)
    session.run(["actions"], {"obs": np.zeros((1, 547), np.float32)})
    with pytest.raises(ValueError, match="exact actually executed"):
        session.commit_executed(Policy(), np.full(29, 10.0), frame=100)
    assert session.faulted and not session.samples
    with pytest.raises(ValueError):
        session.run(["actions"], {"obs": np.zeros((1, 547), np.float32)})


def test_ar_history_commits_only_after_real_shared_world_advance():
    from test_shared_robonaldo_player import Policy as BasePolicy
    from test_shared_robonaldo_player import setup

    class ToyPolicy(Policy, BasePolicy):
        def __init__(self):
            BasePolicy.__init__(self)
            Policy.__init__(self)
            self.ort_session = OwnKickSession(KickPriorActor(tensors()), seed=43)

        def run(self):
            raw = self.ort_session.run(["actions"], {"obs": np.zeros((1, 547), np.float32)})[0][0]
            clipped = np.clip(raw, self.action_clip_lo_il, self.action_clip_hi_il)
            self.policy_output.actions = (
                self.default_q_mj + self.action_scale_mj * clipped[MUJOCO_TO_ISAAC]
            )
            self.policy_output.kps, self.policy_output.kds = self.kps.copy(), self.kds.copy()
            self.time_step += 1

    world, policy = setup(), ToyPolicy()
    player = OwnKickPlayer(world, "a/", policy, goal=np.array([8.0, 0.0, 0.11]))
    target, _ = player.propose()
    assert not policy.ort_session.samples
    assert policy.ort_session.previous_action is None
    world.advance({"a/": target, "b/": np.zeros(29)})
    player.commit()
    assert len(policy.ort_session.samples) == 1
    assert policy.ort_session.samples[0]["frame"] == 0
    assert world.native_steps == 10
    np.testing.assert_array_equal(policy.ort_session.samples[0]["executed_pd_target"], target)
