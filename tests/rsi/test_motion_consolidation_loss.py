import pytest

from rosclaw_soccer.rsi.motion_consolidation_loss import weighted_motion_token_loss


def test_loss_weights_sequences_not_number_of_neutral_standing_frames():
    import torch

    tokens = torch.full((2, 300, 64), 0.5, requires_grad=True)
    targets = torch.zeros_like(tokens)
    mask = torch.zeros((2, 300), dtype=torch.bool)
    mask[0, 20:45] = True
    mask[1, 20:] = True
    weights = torch.tensor([0.25, 2.0], dtype=torch.float32)
    loss = weighted_motion_token_loss(tokens, targets, mask, weights)
    assert float(loss.detach()) == pytest.approx(0.28125)
    loss.backward()
    assert not tokens.grad[:, :20].any()
    assert not tokens.grad[0, 45:].any()
    assert tokens.grad[1, 20:].abs().sum() > 0


def test_empty_or_unowned_prefix_labels_rejected():
    import torch

    x = torch.zeros((1, 300, 64))
    mask = torch.zeros((1, 300), dtype=torch.bool)
    with pytest.raises(ValueError):
        weighted_motion_token_loss(x, x, mask, torch.ones(1))
    mask[0, 0] = True
    with pytest.raises(ValueError):
        weighted_motion_token_loss(x, x, mask, torch.ones(1))
