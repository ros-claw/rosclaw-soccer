"""Optional CPU plumbing tests; synthetic fixtures are not motion evidence."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
pytest.importorskip("rsl_rl")

from rsl_rl.algorithms import PPO  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from rsl_rl.storage import RolloutStorage  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from test_velocity_prior_actor import tensors  # noqa: E402

from rosclaw_soccer.providers.g1.velocity_policy import file_hash  # noqa: E402
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor  # noqa: E402
from scripts.rsi_mjlab_prior_rl import (  # noqa: E402
    RehearsalPPO,
    VelocityPriorMLPModel,
    load_rehearsal,
)


def prior_fixture(tmp_path: Path):
    values = tensors()
    nodes = [
        onnx.helper.make_node("Sub", ["obs", "obs_normalizer._mean"], ["centered"]),
        onnx.helper.make_node("Div", ["centered", "onnx::Div_24"], ["normalized"]),
    ]
    previous = "normalized"
    for index in (0, 2, 4, 6):
        target = "actions" if index == 6 else f"linear{index}"
        nodes.append(
            onnx.helper.make_node(
                "Gemm",
                [previous, f"mlp.{index}.weight", f"mlp.{index}.bias"],
                [target],
                alpha=1.0,
                beta=1.0,
                transB=1,
            )
        )
        previous = target
        if index != 6:
            previous = f"elu{index}"
            nodes.append(onnx.helper.make_node("Elu", [target], [previous], alpha=1.0))
    graph = onnx.helper.make_graph(
        nodes,
        "synthetic-test-only",
        [onnx.helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, [1, 98])],
        [onnx.helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, [1, 29])],
        [onnx.numpy_helper.from_array(v, k) for k, v in values.items()],
    )
    path = tmp_path / "synthetic.onnx"
    onnx.save(onnx.helper.make_model(graph), path)
    prior = VelocityPriorActor.from_onnx(path, expected_hash=file_hash(path))
    checkpoint = tmp_path / "prior.pt"
    torch.save(
        dict(
            actor_state_dict=prior.state_dict(),
            optimizer_steps=0,
            velocity_features=98,
            reference_features=70,
            original_onnx_hash=file_hash(path),
        ),
        checkpoint,
    )
    return prior, dict(
        prior_model=str(path),
        prior_model_hash=file_hash(path),
        prior_checkpoint=str(checkpoint),
        prior_checkpoint_hash=file_hash(checkpoint),
    )


def make_actor(tmp_path):
    prior, arguments = prior_fixture(tmp_path)
    obs = TensorDict({"actor": torch.randn(2, 168), "critic": torch.randn(2, 4)}, batch_size=[2])
    groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = VelocityPriorMLPModel(
        obs,
        groups,
        "actor",
        29,
        hidden_dims=[512, 256, 128],
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.1,
            "std_type": "scalar",
        },
        **arguments,
    )
    return prior, actor, obs, groups


def test_initialized_rsl_actor_is_prior_and_optimizer_owns_it(tmp_path):
    prior, actor, obs, _ = make_actor(tmp_path)
    assert torch.equal(actor(obs), prior(obs["actor"]))
    before = actor(obs).detach().clone()
    actor.update_normalization(obs)
    assert torch.equal(actor(obs), before)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    owned = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert all(id(p) in owned for p in actor.mlp.parameters())
    assert actor.mlp.mlp[0].weight[:, 98:].count_nonzero() == 0


def test_rehearsal_step_changes_real_weights_and_restores_counters(tmp_path, monkeypatch):
    prior, actor, obs, groups = make_actor(tmp_path)
    rng = np.random.default_rng(11)
    old_obs = rng.normal(size=(1200, 98)).astype(np.float32)
    with torch.no_grad():
        target = prior(
            torch.from_numpy(
                np.concatenate((old_obs, np.zeros((1200, 70), dtype=np.float32)), axis=1)
            )
        ).numpy().copy() + np.float32(0.1)
    archive = tmp_path / "synthetic-rehearsal.npz"
    np.savez_compressed(archive, observation=old_obs, action=target)
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16])
    storage = RolloutStorage("rl", 2, 1, obs, [29], device="cpu")
    algorithm = RehearsalPPO(
        actor,
        critic,
        storage,
        rehearsal_path=archive,
        rehearsal_hash=file_hash(archive),
        learning_rate=1e-4,
    )
    monkeypatch.setattr(PPO, "update", lambda self: {"synthetic_ppo_stub": 0.0})
    before = actor.mlp.mlp[6].weight.detach().clone()
    losses = algorithm.update()
    assert losses["native_rehearsal_mse"] > 0
    assert not torch.equal(before, actor.mlp.mlp[6].weight)
    assert algorithm.rehearsal_updates == algorithm.rehearsal_attempted_updates == 1
    assert algorithm.rehearsal_rows == 256
    saved = algorithm.save()
    algorithm.rehearsal_updates = algorithm.rehearsal_attempted_updates = 0
    algorithm.rehearsal_rows = 0
    algorithm.load(saved)
    assert algorithm.rehearsal_updates == 1
    saved["native_rehearsal"]["bank_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="same committed"):
        algorithm.load(saved)


def test_unbound_rehearsal_rejected_before_loading(tmp_path):
    archive = tmp_path / "invalid.npz"
    np.savez_compressed(
        archive,
        observation=np.zeros((1200, 98), dtype=np.float32),
        action=np.zeros((1200, 29), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="immutable"):
        load_rehearsal(archive, "sha256:" + "0" * 64, "cpu")
