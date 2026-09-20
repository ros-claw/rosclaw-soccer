from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.human_tactical_prior import (
    INTENTS,
    SCHEMA,
    HumanTacticalPrior,
)


def export(path, **changes):
    values = {
        "encoder.0.weight": np.zeros((64, 20), np.float32),
        "encoder.0.bias": np.zeros(64, np.float32),
        "encoder.2.weight": np.zeros((64, 64), np.float32),
        "encoder.2.bias": np.zeros(64, np.float32),
        "waypoint.weight": np.zeros((2, 64), np.float32),
        "waypoint.bias": np.array([0.1, -0.2], np.float32),
        "intent.weight": np.zeros((4, 64), np.float32),
        "intent.bias": np.arange(4, dtype=np.float32),
        "mean": np.zeros(20, np.float32),
        "scale": np.ones(20, np.float32),
        "target_scale": np.array([2, 3], np.float32),
        "schema": np.asarray(SCHEMA),
        "checkpoint_hash": np.asarray("sha256:" + "a" * 64),
        "dataset_manifest_hash": np.asarray("sha256:" + "b" * 64),
        "intent_labels": np.asarray(INTENTS),
    }
    values.update(changes)
    np.savez_compressed(path, **values)
    return hash_bytes(path.read_bytes())


def features():
    result = np.zeros((3, 20), np.float32)
    result[:, 2:4] = [0.01, -0.02]
    result[:, 16] = result[:, 19] = 1
    return result


def test_prediction_and_source_identity(tmp_path: Path):
    path = tmp_path / "prior.npz"
    digest = export(path)
    prior = HumanTacticalPrior(path, expected_hash=digest)
    x = features()
    original = x.copy()
    delta, logits = prior.predict(x, pitch_dimensions_m=(100.0, 50.0))
    np.testing.assert_allclose(delta, np.tile([1.2, -1.6], (3, 1)), atol=1e-6)
    np.testing.assert_array_equal(logits, np.tile([0, 1, 2, 3], (3, 1)))
    np.testing.assert_array_equal(x, original)
    assert prior.artifact_hash == digest
    assert prior.checkpoint_hash == "sha256:" + "a" * 64
    assert delta.dtype == logits.dtype == np.float32


@pytest.mark.parametrize(
    "changes",
    [
        {"scale": np.zeros(20, np.float32)},
        {"target_scale": np.array([1, -1], np.float32)},
        {"mean": np.full(20, np.nan, np.float32)},
        {"mean": np.zeros(20, np.float64)},
        {"mean": np.zeros(19, np.float32)},
        {"schema": np.asarray("wrong")},
        {"schema": np.asarray([SCHEMA])},
        {"checkpoint_hash": np.asarray("not-a-hash")},
        {"intent_labels": np.asarray(INTENTS[::-1])},
        {"unexpected": np.asarray(1)},
        {"mean": np.zeros(300_000, np.float32)},
        {"mean": np.array([object()] * 20, dtype=object)},
    ],
)
def test_reject_invalid_export(tmp_path: Path, changes):
    path = tmp_path / "prior.npz"
    digest = export(path, **changes)
    with pytest.raises(ValueError):
        HumanTacticalPrior(path, expected_hash=digest)


def test_digest_mismatch(tmp_path: Path):
    path = tmp_path / "prior.npz"
    export(path)
    with pytest.raises(ValueError, match="digest"):
        HumanTacticalPrior(path, expected_hash="sha256:" + "0" * 64)


@pytest.mark.parametrize("kind", ["missing", "fraction", "nan", "double", "empty", "width"])
def test_reject_invalid_features(tmp_path: Path, kind):
    path = tmp_path / "prior.npz"
    prior = HumanTacticalPrior(path, expected_hash=export(path))
    x = features()
    if kind == "missing":
        x[:, 12:17] = 0
    elif kind == "fraction":
        x[:, 12:17] = [0, 0, 0, 0.5, 0.5]
    elif kind == "nan":
        x[0, 0] = np.nan
    elif kind == "double":
        x = x.astype(np.float64)
    elif kind == "empty":
        x = x[:0]
    else:
        x = x[:, :-1]
    with pytest.raises(ValueError):
        prior.predict(x, pitch_dimensions_m=(105.0, 68.0))


@pytest.mark.parametrize("dims", [(10.0, 8.0), (float("nan"), 68.0), (105.0, -1.0)])
def test_reject_unvalidated_pitch_transfer(tmp_path: Path, dims):
    path = tmp_path / "prior.npz"
    prior = HumanTacticalPrior(path, expected_hash=export(path))
    with pytest.raises(ValueError, match="domain"):
        prior.predict(features(), pitch_dimensions_m=dims)
