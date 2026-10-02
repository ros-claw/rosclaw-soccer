import gzip
import json

import pytest

from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.sampling_model_io import load_sampling_model
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once, write_shared_sampling_model
from scripts.rsi_collect_approach_lateral_tracking_v286 import checked_step_input


def view(seed=42):
    value = dict(
        schema="soccer.rsi.smooth_memory_sampling.v1",
        seed=seed,
        rho=0.9,
        std_raw=0.1,
        mean_model={"weights": [0.0, -0.0, 1.234567890123e-12] * 100},
        hardware_authorized=False,
    )
    return {**value, "model_hash": hash_json(value)}


def publish(tmp_path, seed=42):
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    path = models / f"sample-{seed}.json.gz"
    write_shared_sampling_model(path, view(seed))
    return path


def test_complete_views_and_seeds_preserved_with_one_whole_mean(tmp_path):
    a, b = publish(tmp_path), publish(tmp_path, 43)
    assert load_sampling_model(a) == view()
    assert checked_step_input(a) == view()
    assert load_sampling_model(b) == view(43)
    assert len(list((tmp_path / ".shared-models").iterdir())) == 1
    before = a.read_bytes()
    write_shared_sampling_model(a, view())
    assert a.read_bytes() == before


def test_ordinary_historical_model_returned_unchanged(tmp_path):
    for name in ("view.json", "view.json.gz"):
        path = tmp_path / name
        write_once(path, view())
        assert load_sampling_model(path) == view()


@pytest.mark.parametrize("fault", ["payload", "envelope", "symlink", "missing", "escape"])
def test_corrupted_or_nonlocal_sampling_inputs_rejected(tmp_path, fault):
    path = publish(tmp_path)
    payload = next((tmp_path / ".shared-models").iterdir())
    if fault == "missing":
        payload.unlink()
    elif fault == "symlink":
        target = tmp_path / "elsewhere.json.gz"
        payload.rename(target)
        payload.symlink_to(target)
    else:
        target = payload if fault == "payload" else path
        value = load_json_artifact(target)
        if fault == "payload":
            value["weights"][2] *= 2
        elif fault == "escape":
            value["payload_hash"] = "../other.json.gz"
        else:
            value["logical_document_hash"] = "sha256:" + "0" * 64
        with gzip.open(target, "wt", encoding="utf-8") as stream:
            json.dump(value, stream)
    with pytest.raises(ValueError):
        load_sampling_model(path)


def test_unsealed_view_is_not_published(tmp_path):
    (tmp_path / "models").mkdir()
    value = view()
    value["seed"] += 1
    with pytest.raises(ValueError, match="sealed"):
        write_shared_sampling_model(tmp_path / "models/sample-0.json.gz", value)
    assert not (tmp_path / ".shared-models").exists()


def test_step_preflight_rejects_a_modified_plain_view_before_native_allocation(tmp_path):
    value = view()
    value["seed"] += 1
    path = tmp_path / "bad-view.json"
    write_once(path, value)
    with pytest.raises(ValueError, match="complete sealed requested"):
        checked_step_input(path)
