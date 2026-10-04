import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.foundation_observation_capture import (
    FIELDS,
    PREFIX,
    capture_contract,
    validate_capture,
)


def sample():
    rng = np.random.default_rng(456)
    arrays = {PREFIX + k: rng.normal(size=(3, 2, v)).astype(np.float32) for k, v in FIELDS.items()}
    arrays[PREFIX + "decoder_input"][:, :, :64] = arrays[PREFIX + "latent_token"]
    return capture_contract(), arrays


def test_capture_is_finite_full_width_with_explicit_non_learning_boundary():
    contract, arrays = sample()
    before = {k: v.copy() for k, v in arrays.items()}
    validate_capture(contract, arrays, frames=3, lanes=2)
    assert contract["optimizer_updates"] == 0
    assert not contract["foundation_recomputed_independently"]
    assert not contract["hardware_authorized"]
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])


@pytest.mark.parametrize("field", list(FIELDS))
@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype", "nonfinite"])
def test_rejects_incomplete_or_malformed_neural_data(field, corruption):
    contract, arrays = sample()
    key = PREFIX + field
    if corruption == "missing":
        del arrays[key]
    elif corruption == "shape":
        arrays[key] = arrays[key][:, :1]
    elif corruption == "dtype":
        arrays[key] = arrays[key].astype(np.float64)
    else:
        arrays[key][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="capture"):
        validate_capture(contract, arrays, frames=3, lanes=2)


def test_token_mismatch_and_unknown_capture_field_rejected():
    contract, arrays = sample()
    arrays[PREFIX + "decoder_input"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="token"):
        validate_capture(contract, arrays, frames=3, lanes=2)
    _, arrays = sample()
    arrays[PREFIX + "future_return"] = np.zeros((3, 2, 1), np.float32)
    with pytest.raises(ValueError, match="fields"):
        validate_capture(contract, arrays, frames=3, lanes=2)


@pytest.mark.parametrize("key", ["tracker_source_hash", "layout", "hardware_authorized"])
def test_capture_contract_cannot_be_relabelled(key):
    contract, arrays = sample()
    changed = copy.deepcopy(contract)
    changed[key] = "wrong"
    with pytest.raises(ValueError, match="contract/source"):
        validate_capture(changed, arrays, frames=3, lanes=2)


@pytest.mark.parametrize("frames,lanes", [(True, 2), (0, 2), (200001, 2), (3, False), (3, 4097)])
def test_capture_bounds_are_explicit(frames, lanes):
    contract, arrays = sample()
    with pytest.raises(ValueError, match="bounded"):
        validate_capture(contract, arrays, frames=frames, lanes=lanes)
