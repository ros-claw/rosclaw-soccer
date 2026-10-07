"""Complete transport parity; no native physics or learning qualification."""

import copy

import pytest

from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from rosclaw_soccer.rsi.sampling_model_io import load_sampling_model
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once, write_shared_sampling_model
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_plain_gzip_and_shared_complete_documents_match_reference_and_stay_owned(learned, tmp_path):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    view = factory.sampling_view(seed=773)
    models = tmp_path / "models"
    models.mkdir()
    paths = [tmp_path / "plain.json", tmp_path / "gzip.json.gz", models / "shared.json.gz"]
    for path in paths[:2]:
        write_once(path, view)
    write_shared_sampling_model(paths[-1], view)
    for path in paths:
        reference = load_sampling_model(path)
        actual = load_sampling_model(path, recurrent_sampling_factory=factory)
        assert hash_json(actual) == hash_json(reference) == hash_json(view)
        assert actual["mean_model"]["critic_parameters"] == learned[-1]["critic_parameters"]
        actual["mean_model"]["parameters"]["head_bias"][0] = 123
        restored = load_sampling_model(path, recurrent_sampling_factory=factory)
        assert hash_json(restored) == hash_json(view)


def test_complete_law_critic_authority_and_model_family_rejected(learned, tmp_path):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    view = factory.sampling_view(seed=773)
    for index, (field, value) in enumerate(
        (("std_raw", 0.15), ("seed", True), ("promotion_authorized", 0), ("schema", "other"))
    ):
        changed = copy.deepcopy(view)
        changed[field] = value
        changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
        path = tmp_path / f"bad-{index}.json.gz"
        write_once(path, changed)
        with pytest.raises(ValueError):
            load_sampling_model(path, recurrent_sampling_factory=factory)
    changed = copy.deepcopy(view)
    changed["mean_model"]["critic_parameters"]["bias_1"][0] += 1
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    path = tmp_path / "bad-critic.json.gz"
    write_once(path, changed)
    with pytest.raises(ValueError, match="actor/critic/parent"):
        load_sampling_model(path, recurrent_sampling_factory=factory)


def test_private_reader_does_not_skip_missing_or_symlink_shared_payload(learned, tmp_path):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    models = tmp_path / "models"
    models.mkdir()
    path = models / "sample.json.gz"
    write_shared_sampling_model(path, factory.sampling_view(seed=773))
    payload = next((tmp_path / ".shared-models").iterdir())
    target = tmp_path / "original-payload.json.gz"
    payload.rename(target)
    with pytest.raises(ValueError, match="non-symlink"):
        load_sampling_model(path, recurrent_sampling_factory=factory)
    payload.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        load_sampling_model(path, recurrent_sampling_factory=factory)


@pytest.mark.parametrize("factory", [object(), lambda _: None])
def test_external_validator_rejected_before_io(tmp_path, factory):
    with pytest.raises(ValueError, match="exact private"):
        load_sampling_model(tmp_path / "absent.json", recurrent_sampling_factory=factory)
