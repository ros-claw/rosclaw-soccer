import copy

import pytest
from rosclaw.growth.shared_proof_payload import detach_payload

from rosclaw_soccer.rsi.sampling_model_io import load_sampling_model
from rosclaw_soccer.rsi.smooth_memory_motor import make_sampling_view
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_exact_shared_sampling_views import exact_shared_views, publish_shared_views
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_every_complete_view_and_seal_match_old_constructor(smooth_parent, tmp_path):  # noqa: F811
    seeds = list(range(202610335, 202610351))
    before = hash_json(smooth_parent)
    views = exact_shared_views(smooth_parent, seeds)
    (tmp_path / "models").mkdir()
    publish_shared_views(tmp_path, smooth_parent, views)
    for index, seed in enumerate(seeds):
        ordinary = make_sampling_view(smooth_parent, seed=seed, std=0.1, rho=0.9)
        assert views[index] == detach_payload(ordinary, ("mean_model",))[0]
        assert load_sampling_model(tmp_path / f"models/sample-{index}.json.gz") == ordinary
    assert hash_json(smooth_parent) == before
    assert len(list((tmp_path / ".shared-models").iterdir())) == 1


@pytest.mark.parametrize("seeds", [[], [True], [-1], [2**32], [2, 2]])
def test_bad_or_duplicate_seeds_fail_before_construction(seeds):
    with pytest.raises(ValueError, match="seeds"):
        exact_shared_views({}, seeds)


@pytest.mark.parametrize("fault", ["mean", "seed", "logical", "authority"])
def test_corruption_rejected_before_any_payload_publication(smooth_parent, tmp_path, fault):  # noqa: F811
    views = exact_shared_views(smooth_parent, [42])
    mean = copy.deepcopy(smooth_parent)
    if fault == "mean":
        mean["model_hash"] = "sha256:" + "0" * 64
    elif fault == "logical":
        views[0]["logical_document_hash"] = "sha256:" + "0" * 64
    else:
        views[0]["stripped_document"]["seed" if fault == "seed" else "hardware_authorized"] = True
    (tmp_path / "models").mkdir()
    with pytest.raises(ValueError):
        publish_shared_views(tmp_path, mean, views)
    assert not (tmp_path / ".shared-models").exists()
