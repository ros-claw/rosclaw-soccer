"""Synthetic diagnostic NN fitting; not G1 training or physical success."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi import postcontact_outcome_model as mod


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(6, 121)).astype(np.float32)
    y = np.zeros((6, 13, 2), dtype=bool)
    y[:, :, 1] = True
    y[:3, :6, 0] = True
    y[3:, 6:, 0] = True
    mask = np.ones_like(y)
    model = mod.fit(
        x,
        y,
        mask,
        seed=1019,
        dataset_hash="sha256:" + "a" * 64,
        training_context_indices=list(range(6)),
    )
    return x, y, mask, model


def test_measured_heads_fit_and_numpy_inference(fitted):
    x, y, mask, model = fitted
    p = mod.predict(model, x)
    assert p.shape == (6, 13, 2) and p.dtype == np.float32
    assert np.mean((p[mask] - y[mask]) ** 2) < 0.005
    assert model["training_loss_final"] < model["training_loss_initial"]
    assert model["optimizer_updates"] == 500 and model["independent_training_states"] == 6
    assert model["action_outcome_pairs"] == 78
    assert not model["motor_policy_activation_authorized"] and not model["teacher_qualified"]


def test_determinism_without_global_torch_rng_mutation(fitted):
    import torch

    x, y, mask, model = fitted
    before = torch.random.get_rng_state().clone()
    again = mod.fit(
        x,
        y,
        mask,
        seed=1019,
        dataset_hash="sha256:" + "a" * 64,
        training_context_indices=list(range(6)),
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert again == model


@pytest.mark.parametrize("what", ["retention", "no_pass_observation", "float_labels", "duplicate"])
def test_training_boundary(fitted, what):
    x, y, mask, _ = fitted
    mask = mask.copy()
    indices = list(range(6))
    if what == "retention":
        indices[-1] = 12
    elif what == "no_pass_observation":
        mask[:, :, 0] = False
    elif what == "float_labels":
        y = y.astype(np.float32)
    else:
        indices[-1] = 0
    with pytest.raises(ValueError, match="local-only"):
        mod.fit(
            x,
            y,
            mask,
            seed=1019,
            dataset_hash="sha256:" + "a" * 64,
            training_context_indices=indices,
        )


@pytest.mark.parametrize("what", ["tamper", "nan", "scale", "authority"])
def test_sealed_finite_model_and_no_activation(fitted, what):
    x, _, _, model = fitted
    model = copy.deepcopy(model)
    if what == "tamper":
        model["seed"] = 5
    elif what == "nan":
        model["parameters"]["w1"][0][0] = float("nan")
    elif what == "scale":
        model["parameters"]["scale"][0] = 0.0
    else:
        model["motor_policy_activation_authorized"] = True
    if what not in ("tamper", "nan"):
        model["report_hash"] = mod.hash_json({k: v for k, v in model.items() if k != "report_hash"})
    with pytest.raises(ValueError):
        mod.predict(model, x)


def test_censored_pass_target_is_ignored(fitted):
    x, y, mask, _ = fitted
    mask = mask.copy()
    mask[0, 0, 0] = False
    alt = y.copy()
    alt[0, 0, 0] = not alt[0, 0, 0]
    kwargs = dict(
        seed=2027, dataset_hash="sha256:" + "b" * 64, training_context_indices=list(range(6))
    )
    a = mod.fit(x, y, mask, **kwargs)
    b = mod.fit(x, alt, mask, **kwargs)
    assert a["parameters"] == b["parameters"]
