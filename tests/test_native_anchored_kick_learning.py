import numpy as np
import pytest

from rosclaw_soccer.training.native_anchored_kick_learning import (
    LEARNING_FEATURE,
    SCHEMA,
    build_native_anchored_kick_actor_critic,
    native_anchored_kick_observation,
)


def fixture():
    context = np.zeros(554, np.float32)
    context[553] = 1
    action = np.linspace(-2, 2, 29, dtype=np.float32)
    return context, action


def test_current_native_action_has_explicit_distinct_layout_and_owned_memory():
    context, action = fixture()
    obs = native_anchored_kick_observation(contextual_observation=context, reference_action=action)
    assert obs.shape == (583,) and obs.dtype == np.float32
    assert LEARNING_FEATURE == 582 and "native_anchored" in SCHEMA
    np.testing.assert_array_equal(obs[553:582], action)
    assert obs[582] == 1
    action[:] = 9
    context[:] = 9
    assert not (obs[553:582] == 9).any() and obs[582] == 1


@pytest.mark.parametrize("damage", ["shape", "dtype", "nan", "bound", "context", "bit"])
def test_bad_observation_contract_is_rejected(damage):
    context, action = fixture()
    if damage == "shape":
        action = action[:-1]
    elif damage == "dtype":
        action = action.astype(np.float64)
    elif damage == "nan":
        action[0] = np.nan
    elif damage == "bound":
        action[0] = 10001
    elif damage == "context":
        context[547] = 1.01
    else:
        context[553] = 0.5
    with pytest.raises(ValueError):
        native_anchored_kick_observation(contextual_observation=context, reference_action=action)


def test_zero_initialization_is_exact_and_bit_is_training_metadata_only():
    torch = pytest.importorskip("torch")
    context, action = fixture()
    obs = native_anchored_kick_observation(contextual_observation=context, reference_action=action)
    x = torch.from_numpy(np.tile(obs, (4, 1)))
    model = build_native_anchored_kick_actor_critic()
    mean, value = model(x)
    assert torch.equal(mean, x[:, 553:582]) and not value.any()
    before = x.clone()
    inactive = x.clone()
    inactive[:, 582] = 0
    assert torch.equal(model(inactive)[0], mean)
    assert torch.equal(model(inactive)[1], value)
    assert torch.equal(x, before)
    with torch.no_grad():
        model.actor[-1].bias.fill_(100)
    residual = model(x)[0] - x[:, 553:582]
    assert float(residual.detach().abs().max()) <= 0.100001  # float32 subtraction


@pytest.mark.parametrize("damage", ["shape", "dtype", "nan", "context", "bit"])
def test_model_revalidates_packed_contract(damage):
    torch = pytest.importorskip("torch")
    model = build_native_anchored_kick_actor_critic()
    x = torch.zeros(1, 583)
    if damage == "shape":
        x = x[:, :-1]
    elif damage == "dtype":
        x = x.double()
    elif damage == "nan":
        x[0, 10] = float("nan")
    elif damage == "context":
        x[0, 552] = 2
    else:
        x[0, 582] = -1
    with pytest.raises(ValueError):
        model(x)
