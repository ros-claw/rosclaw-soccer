import io
import zipfile

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.full_kick_inference import G1FrozenFullKickInference
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.full_kick_checkpoint import (
    MAXIMUM_CHECKPOINT_BYTES,
    full_kick_checkpoint_shapes,
    load_full_kick_checkpoint,
)

EPISODE = "sha256:" + "d" * 64


@pytest.fixture
def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.full_kick_learning import build_full_kick_actor_critic

    torch.set_num_threads(1)
    rng = np.random.default_rng(1481)
    shapes = full_kick_checkpoint_shapes()
    reference = {
        key.removeprefix("actor."): (rng.normal(size=shape) * 0.01).astype(np.float32)
        for key, shape in shapes.items()
        if key.startswith("actor.")
    }
    model = build_full_kick_actor_critic(reference)
    with torch.no_grad():
        model.context_adapter.weight.fill_(0.001)
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == shapes
    path = tmp_path / "actor.npz"
    np.savez_compressed(
        path, **{key: value.detach().numpy() for key, value in model.state_dict().items()}
    )
    kwargs = dict(
        expected_actor_hash=hash_bytes(path.read_bytes()),
        agent_id="red.finisher",
        body_hash="sha256:" + "a" * 64,
        reference_contract_hash="sha256:" + "b" * 64,
    )
    return path, kwargs, model


def observation():
    result = np.zeros(554, dtype=np.float32)
    result[547:553] = [0.5, -0.1, 0.2, 0.1, 0.0, 0.0]
    result[553] = 1
    return result


def test_exact_private_inference_rng_and_frozen_parameters(checkpoint):
    import torch

    path, kwargs, model = checkpoint
    before = torch.get_rng_state().clone()
    left = G1FrozenFullKickInference(path, **kwargs, maximum_episode_frames=3)
    right = G1FrozenFullKickInference(path, **(kwargs | dict(agent_id="blue.finisher")))
    assert left.contract_hash != right.contract_hash
    assert all(not p.requires_grad for p in left._model.parameters())
    left.begin_episode(episode_hash=EPISODE)
    right.begin_episode(episode_hash=EPISODE)
    obs = observation()
    for frame in range(3):
        obs[0] = frame
        with torch.no_grad():
            expected, _ = model(torch.from_numpy(obs)[None])
        proposal = left.propose(frame=frame, episode_hash=EPISODE, observation=obs)
        np.testing.assert_array_equal(
            np.asarray(proposal.raw_action, dtype=np.float32), expected[0].numpy()
        )
        assert proposal.agent_id == "red.finisher" and proposal.activation_ceiling == "SIM_ONLY"
        assert proposal.learning_active
        assert proposal.observation_hash == hash_json(obs.tolist())
    right.propose(frame=0, episode_hash=EPISODE, observation=obs)
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError):
        left.propose(frame=3, episode_hash=EPISODE, observation=obs)
    left.begin_episode(episode_hash="sha256:" + "e" * 64)


@pytest.mark.parametrize(
    "kind", ["frame", "bool_frame", "episode", "nan", "shape", "dtype", "context", "bit"]
)
def test_invalid_observation_latches_only_own_sequence(checkpoint, kind):
    path, kwargs, _ = checkpoint
    motor = G1FrozenFullKickInference(path, **kwargs)
    other = G1FrozenFullKickInference(path, **(kwargs | dict(agent_id="red.playmaker")))
    for item in (motor, other):
        item.begin_episode(episode_hash=EPISODE)
    obs = observation()
    frame, episode = 0, EPISODE
    if kind == "frame":
        frame = 1
    if kind == "bool_frame":
        frame = False
    if kind == "episode":
        episode = "sha256:" + "e" * 64
    if kind == "nan":
        obs[0] = np.nan
    if kind == "shape":
        obs = obs[:-1]
    if kind == "dtype":
        obs = obs.astype(np.float64)
    if kind == "context":
        obs[548] = 2
    if kind == "bit":
        obs[553] = 0.5
    with pytest.raises(ValueError):
        motor.propose(frame=frame, episode_hash=episode, observation=obs)
    with pytest.raises(ValueError):
        motor.propose(frame=0, episode_hash=EPISODE, observation=observation())
    with pytest.raises(ValueError):
        motor.begin_episode(episode_hash=EPISODE)
    other.propose(frame=0, episode_hash=EPISODE, observation=observation())
    motor.begin_episode(episode_hash="sha256:" + "f" * 64)


def test_active_episode_cannot_be_replaced_and_inactive_bit_is_not_permission(checkpoint):
    path, kwargs, _ = checkpoint
    motor = G1FrozenFullKickInference(path, **kwargs)
    motor.begin_episode(episode_hash=EPISODE)
    with pytest.raises(ValueError):
        motor.begin_episode(episode_hash="sha256:" + "f" * 64)
    obs = observation()
    obs[553] = 0
    assert not motor.propose(frame=0, episode_hash=EPISODE, observation=obs).learning_active
    motor.end_episode()
    with pytest.raises(ValueError):
        motor.propose(frame=1, episode_hash=EPISODE, observation=obs)
    motor.begin_episode(episode_hash="sha256:" + "f" * 64)
    motor.end_episode()
    with pytest.raises(ValueError):
        motor.begin_episode(episode_hash=EPISODE)


@pytest.mark.parametrize("kind", ["missing", "extra", "dtype", "nan", "large", "hash"])
def test_checkpoint_rejects_unqualified_numeric_shapes(checkpoint, tmp_path, kind):
    path, kwargs, _ = checkpoint
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    if kind == "missing":
        values.pop("logstd")
    if kind == "extra":
        values["extra"] = np.zeros(1, dtype=np.float32)
    if kind == "dtype":
        values["logstd"] = values["logstd"].astype(np.float64)
    if kind == "nan":
        values["logstd"][0] = np.nan
    if kind == "large":
        values["logstd"][0] = 1e5
    target = tmp_path / "bad.npz"
    np.savez_compressed(target, **values)
    expected = "sha256:" + "0" * 64 if kind == "hash" else hash_bytes(target.read_bytes())
    with pytest.raises(ValueError):
        load_full_kick_checkpoint(target, expected_hash=expected)


def test_loader_bounds_actual_read_and_declared_allocation(checkpoint, tmp_path, monkeypatch):
    path, kwargs, _ = checkpoint
    oversized = tmp_path / "oversized.npz"
    oversized.write_bytes(b"x" * (MAXIMUM_CHECKPOINT_BYTES + 1))
    with pytest.raises(ValueError, match="byte bound"):
        load_full_kick_checkpoint(oversized, expected_hash=kwargs["expected_actor_hash"])
    target = tmp_path / "forged.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, dict(descr="<f4", fortran_order=False, shape=(10**12,))
    )
    with (
        zipfile.ZipFile(path) as original,
        zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as forged,
    ):
        for name in original.namelist():
            forged.writestr(
                name, header.getvalue() if name == "logstd.npy" else original.read(name)
            )

    def must_not_load(*args, **kwargs):
        raise AssertionError("np.load called before bounded header validation")

    monkeypatch.setattr(np, "load", must_not_load)
    with pytest.raises(ValueError, match="tensor header"):
        load_full_kick_checkpoint(target, expected_hash=hash_bytes(target.read_bytes()))


def test_loaded_parameters_owned_readonly_and_input_path_not_reopened(checkpoint):
    path, kwargs, _ = checkpoint
    parameters, digest = load_full_kick_checkpoint(
        path, expected_hash=kwargs["expected_actor_hash"]
    )
    assert digest == kwargs["expected_actor_hash"]
    assert all(not value.flags.writeable for value in parameters.values())
    original = parameters["logstd"].copy()
    path.write_bytes(b"replaced after load")
    np.testing.assert_array_equal(parameters["logstd"], original)


def test_duplicate_members_rejected_before_tensor_loading(checkpoint, tmp_path, monkeypatch):
    path, _, _ = checkpoint
    target = tmp_path / "duplicate.npz"
    with zipfile.ZipFile(path) as original, zipfile.ZipFile(target, "w") as duplicate:
        for name in original.namelist():
            duplicate.writestr(name, original.read(name))
        with pytest.warns(UserWarning, match="Duplicate"):
            duplicate.writestr("logstd.npy", original.read("logstd.npy"))

    def must_not_load(*args, **kwargs):
        raise AssertionError("duplicate ZIP reached array allocation")

    monkeypatch.setattr(np, "load", must_not_load)
    with pytest.raises(ValueError, match="ZIP members"):
        load_full_kick_checkpoint(target, expected_hash=hash_bytes(target.read_bytes()))


@pytest.mark.parametrize(
    "override",
    [
        {"agent_id": "../robot"},
        {"body_hash": "bad"},
        {"maximum_episode_frames": True},
        {"maximum_episode_frames": 4097},
    ],
)
def test_invalid_identity_and_episode_bounds_fail_before_model_construction(checkpoint, override):
    path, kwargs, _ = checkpoint
    with pytest.raises(ValueError):
        G1FrozenFullKickInference(path, **(kwargs | override))


def test_episode_replay_memory_is_bounded(checkpoint):
    path, kwargs, _ = checkpoint
    motor = G1FrozenFullKickInference(path, **kwargs)
    motor._used_episode_hashes = {f"sha256:{i:064x}" for i in range(4096)}
    with pytest.raises(ValueError):
        motor.begin_episode(episode_hash=EPISODE)


def test_module_import_does_not_require_torch():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch']=None; "
            "import rosclaw_soccer.providers.g1.full_kick_inference",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
