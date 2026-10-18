from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rsi_swing_release_ablation_pilot import declared_jobs, require_original_prefix


def test_fixed_four_courses_include_unsafe_counterexample():
    jobs = declared_jobs([SimpleNamespace(context_index=i) for i in range(8)])
    assert [(j["group"], j["context_index"]) for j in jobs] == [
        (8, None),
        (56, None),
        (64, 0),
        (65, 1),
    ]
    assert all(j["role"] == "ablation" and j["option_index"] == 4 for j in jobs)
    assert len({j["pair_id"] for j in jobs}) == 4


@pytest.mark.parametrize("indices", [[], list(range(7)), list(range(7)) + [6]])
def test_missing_original_context_bank_rejected(indices):
    with pytest.raises(ValueError):
        declared_jobs([SimpleNamespace(context_index=i) for i in indices])


def test_only_after_contact_may_diverge():
    original = {"a": np.zeros((300, 1, 6)), "b": np.zeros((300, 1, 29))}
    changed = {k: v.copy() for k, v in original.items()}
    changed["b"][75:] = 1
    require_original_prefix(changed, original, 75)
    changed["b"][74] = 1
    with pytest.raises(ValueError):
        require_original_prefix(changed, original, 75)


@pytest.mark.parametrize("kind", ["missing", "extra", "dtype", "shape", "scalar", "clock"])
def test_all_arrays_and_storage_contract_required(kind):
    original = {"a": np.zeros((300, 1, 6))}
    changed = {k: v.copy() for k, v in original.items()}
    if kind == "missing":
        changed.clear()
    elif kind == "extra":
        changed["b"] = changed["a"]
    elif kind == "dtype":
        changed["a"] = changed["a"].astype(np.float32)
    elif kind == "shape":
        changed["a"] = np.zeros((300, 6))
    elif kind == "scalar":
        changed["a"] = np.asarray(0.0)
    else:
        changed["a"] = np.zeros((299, 1, 6))
    with pytest.raises(ValueError):
        require_original_prefix(changed, original, 75)


@pytest.mark.parametrize("cutoff", [0, 301, True, 1.0])
def test_observed_contact_index_required(cutoff):
    with pytest.raises(ValueError):
        require_original_prefix({"a": np.zeros((300, 1))}, {"a": np.zeros((300, 1))}, cutoff)
