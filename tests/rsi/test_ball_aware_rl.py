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


def test_new_native_inference_needs_explicit_abi_and_separate_reference_perception(tmp_path):
    import numpy as np
    import yaml
    from test_g1_velocity_policy import config, state

    from rosclaw_soccer.rsi.ball_aware_inputs import BALL_AWARE_ABI
    from rosclaw_soccer.rsi.ball_aware_migration import expand_ball_aware_state
    from rosclaw_soccer.rsi.velocity_prior_actor import LearnedPriorNativeSession

    _, parent, _, _ = make_actor(tmp_path)
    expanded = expand_ball_aware_state(parent.state_dict(), part="actor")
    path = tmp_path / "new177.pt"
    torch.save(
        dict(
            stable_prior_rehearsal=True,
            actor_observation_dim=177,
            ball_aware_abi=BALL_AWARE_ABI,
            algorithm=dict(actor_state_dict=expanded),
            initial_actor_state_dict=expanded,
        ),
        path,
    )
    conf = tmp_path / "deploy.yaml"
    conf.write_text(yaml.safe_dump(config(), sort_keys=False))
    args = dict(
        model=tmp_path / "synthetic.onnx",
        config=conf,
        expected_model_hash=file_hash(tmp_path / "synthetic.onnx"),
        expected_config_hash=file_hash(conf),
        checkpoint=path,
        expected_checkpoint_hash=file_hash(path),
        arm="final",
    )
    with pytest.raises(ValueError, match="checkpoint"):
        LearnedPriorNativeSession(**args)
    zero = LearnedPriorNativeSession(**args, reference_mode="zero79")
    declared = LearnedPriorNativeSession(**args, reference_mode="declared79")
    assert zero.contract_hash != declared.contract_hash
    q, v = state()
    prefix = (q, v, np.zeros(3), np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(
        zero.propose(*prefix, frame=0),
        declared.propose_ball_aware(
            *prefix, np.zeros(70, dtype=np.float32), np.ones(9, dtype=np.float32), frame=0
        ),
        atol=1e-6,
    )
    assert declared.last_reference.shape == (79,)
    with pytest.raises(ValueError, match="separate"):
        declared.propose_ball_aware(
            *prefix, np.zeros(69, dtype=np.float32), np.zeros(10, dtype=np.float32), frame=1
        )
    with pytest.raises(ValueError, match="fault latched"):
        declared.propose_ball_aware(
            *prefix, np.zeros(70, dtype=np.float32), np.zeros(9, dtype=np.float32), frame=1
        )
