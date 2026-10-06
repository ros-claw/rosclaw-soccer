"""Source-bound shared recurrent means, not physical execution proof."""

import copy

import pytest

from rosclaw_soccer.rsi.recurrent_sampling_motor import make_sampling_view
from rosclaw_soccer.rsi.sampling_model_io import load_sampling_model
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once, write_shared_sampling_model
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_complete_current_actor_and_critic_shared_across_seed_views(learned, tmp_path):  # noqa: F811
    mean = learned[-1]
    views = [make_sampling_view(mean, seed=778 + i) for i in range(2)]
    models = tmp_path / "models"
    models.mkdir()
    for i, view in enumerate(views):
        path = models / f"sample-{i}.json.gz"
        write_shared_sampling_model(path, view)
        assert hash_json(load_sampling_model(path)) == hash_json(view)
        write_shared_sampling_model(path, view)
    assert len(list((tmp_path / ".shared-models").iterdir())) == 1
    restored = load_sampling_model(models / "sample-0.json.gz")
    assert restored["mean_model"]["critic_parameters"] == mean["critic_parameters"]
    assert restored["mean_model"]["parameters"] == mean["parameters"]
    restored["mean_model"]["parameters"]["head_bias"][0] = 123
    assert (
        load_sampling_model(models / "sample-0.json.gz")["mean_model"]["parameters"]["head_bias"][0]
        != 123
    )


def test_resealed_changed_noise_and_authority_fail_before_store_creation(learned, tmp_path):  # noqa: F811
    views = make_sampling_view(learned[-1], seed=778)
    models = tmp_path / "models"
    models.mkdir()
    for key, value in (
        ("std_raw", 0.15),
        ("rho", 0.8),
        ("hardware_authorized", True),
        ("promotion_authorized", 0),
    ):
        bad = copy.deepcopy(views)
        bad[key] = value
        bad.pop("model_hash")
        bad["model_hash"] = hash_json(bad)
        path = models / f"bad-{key}.json.gz"
        with pytest.raises(ValueError):
            write_shared_sampling_model(path, bad)
        assert not (tmp_path / ".shared-models").exists()
        # Plain transport is not allowed to bypass recurrent law validation.
        write_once(path, bad)
        with pytest.raises(ValueError):
            load_sampling_model(path)
