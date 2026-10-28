import pytest
import torch

from rosclaw_soccer.rsi.first_touch_outcome_model import FirstTouchOutcomeModel


def test_outcome_model_has_two_independent_outputs_and_gradients():
    model = FirstTouchOutcomeModel()
    inputs = torch.ones((3, 7))
    result = model(inputs)
    assert result.shape == (3, 2)
    result.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_rejects_invalid_inputs():
    model = FirstTouchOutcomeModel()
    with pytest.raises(ValueError):
        model(torch.zeros((2, 6)))
    invalid = torch.zeros((1, 7))
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(invalid)
