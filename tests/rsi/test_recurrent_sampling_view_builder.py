"""Complete logical parity and source/state isolation, not physics proof."""

import copy
from pathlib import Path

import pytest
from rosclaw.growth.shared_proof_payload import restore_payload

import rosclaw_soccer.rsi.recurrent_sampling_view_builder as builder_module
from rosclaw_soccer.rsi.recurrent_sampling_motor import make_sampling_view
from rosclaw_soccer.rsi.recurrent_sampling_view_builder import RecurrentSamplingViewBuilder
from rosclaw_soccer.rsi.sampling_model_io import load_sampling_model
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_cached_seed_views_are_complete_original_documents(learned, tmp_path):  # noqa: F811
    original = learned[-1]
    expected = copy.deepcopy(original)
    builder = RecurrentSamplingViewBuilder(original)
    original["parameters"]["head_bias"][0] = 123
    payload = builder.mean_model()
    assert payload == expected
    models, store = tmp_path / "models", tmp_path / ".shared-models"
    models.mkdir()
    store.mkdir()
    write_once(store / f"{builder.payload_hash[7:]}.json.gz", payload)
    for index, seed in enumerate((0, 773, 442610335, 2**32 - 1)):
        envelope = builder.envelope(seed=seed)
        reference = make_sampling_view(expected, seed=seed)
        assert restore_payload(envelope, payload) == reference
        path = models / f"sample-{index}.json.gz"
        write_once(path, envelope)
        assert hash_json(load_sampling_model(path)) == hash_json(reference)
    payload["critic_parameters"]["bias_1"][0] = 123
    assert builder.mean_model() == expected
    contract = builder.contract()
    assert contract["actual_mean_model_hash"] == expected["model_hash"]
    assert contract["hardware_authorized"] is False
    contract["source_pins"].clear()
    assert builder.contract()["source_pins"]


def test_bad_seeds_resealed_model_and_changed_cached_bytes_fail(learned):  # noqa: F811
    original = learned[-1]
    builder = RecurrentSamplingViewBuilder(original)
    for seed in (True, -1, 2**32, 773.0, "773"):
        with pytest.raises(ValueError):
            builder.envelope(seed=seed)
    bad = copy.deepcopy(original)
    bad["hardware_authorized"] = True
    bad["model_hash"] = hash_json({k: v for k, v in bad.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        RecurrentSamplingViewBuilder(bad)
    builder._cache._payload_bytes += b" "
    with pytest.raises(ValueError):
        builder.envelope(seed=773)


def test_repeated_seed_construction_does_not_revalidate_the_mean(learned, monkeypatch):  # noqa: F811
    original = learned[-1]
    builder = RecurrentSamplingViewBuilder(original)

    def forbidden(*args, **kwargs):
        raise AssertionError("repeated complete original allocation")

    monkeypatch.setattr(builder_module, "make_sampling_view", forbidden)
    for seed in range(8):
        assert builder.envelope(seed=seed)["stripped_document"]["seed"] == seed
    # Source drift must still fail, even when only another seed is requested.
    original_read = Path.read_bytes

    def altered(path):
        data = original_read(path)
        return data + b" " if str(path) == builder_module.__file__ else data

    monkeypatch.setattr(Path, "read_bytes", altered)
    with pytest.raises(ValueError):
        builder.envelope(seed=8)
