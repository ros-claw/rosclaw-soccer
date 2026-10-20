import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from rsl_rl.models import MLPModel  # noqa: E402
from rsl_rl.storage import RolloutStorage  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from test_mjlab_prior_rl import make_actor  # noqa: E402

from rosclaw_soccer.providers.g1.velocity_policy import file_hash  # noqa: E402
from scripts.rsi_mjlab_ball_aware_rl import (  # noqa: E402
    BallAwareCriticMLPModel,
    BallAwarePriorMLPModel,
)
from scripts.rsi_mjlab_prior_rl import RehearsalPPO  # noqa: E402


def test_new_actor_critic_created_before_fresh_optimizer_and_rehearsal_abi(tmp_path):
    import numpy as np

    prior, old_actor, _, _ = make_actor(tmp_path)
    base = TensorDict({"actor": torch.randn(2, 168), "critic": torch.randn(2, 295)}, batch_size=[2])
    groups = dict(actor=["actor"], critic=["critic"])
    old_critic = MLPModel(
        base,
        groups,
        "critic",
        1,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    checkpoint = tmp_path / "parent.pt"
    torch.save(
        dict(
            stable_prior_rehearsal=True,
            actor_observation_dim=168,
            algorithm=dict(
                actor_state_dict=old_actor.state_dict(), critic_state_dict=old_critic.state_dict()
            ),
        ),
        checkpoint,
    )
    obs = TensorDict(
        {
            "actor": torch.cat((base["actor"], torch.randn(2, 9)), 1),
            "critic": torch.cat((base["critic"], torch.randn(2, 9)), 1),
        },
        batch_size=[2],
    )
    arguments = dict(
        parent_checkpoint=str(checkpoint),
        parent_checkpoint_hash=file_hash(checkpoint),
        hidden_dims=[512, 256, 128],
        activation="elu",
        cnn_cfg=None,
    )
    actor = BallAwarePriorMLPModel(
        obs,
        groups,
        "actor",
        29,
        **arguments,
        prior_model=str(tmp_path / "synthetic.onnx"),
        prior_model_hash=file_hash(tmp_path / "synthetic.onnx"),
        obs_normalization=False,
        distribution_cfg=dict(class_name="GaussianDistribution", init_std=0.1, std_type="scalar"),
    )
    critic = BallAwareCriticMLPModel(obs, groups, "critic", 1, **arguments, obs_normalization=True)
    torch.testing.assert_close(actor(obs), old_actor(base), atol=1e-6, rtol=0)
    torch.testing.assert_close(critic(obs), old_critic(base), atol=1e-6, rtol=0)
    bank = tmp_path / "bank.npz"
    old_obs = np.zeros((1200, 98), dtype=np.float32)
    with torch.no_grad():
        actions = prior(torch.zeros((1200, 168))).numpy()
    np.savez_compressed(bank, observation=old_obs, action=actions)
    storage = RolloutStorage("rl", 2, 1, obs, [29], device="cpu")
    algorithm = RehearsalPPO(
        actor,
        critic,
        storage,
        rehearsal_path=bank,
        rehearsal_hash=file_hash(bank),
        rehearsal_reference_features=79,
    )
    assert algorithm.rehearsal_obs["actor"].shape == (1200, 177)
    assert len(algorithm.optimizer.state) == 0
    owned = {id(p) for group in algorithm.optimizer.param_groups for p in group["params"]}
    assert all(id(p) in owned for p in actor.mlp.parameters())
    assert all(id(p) in owned for p in critic.parameters())
    assert algorithm.save()["native_rehearsal"]["reference_features"] == 79
