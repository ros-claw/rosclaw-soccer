import pytest

from tests.rsi.test_kinematic_motion_residual import model


def test_zero_classifier_preserves_parent_tokens_value_and_live_recurrent_memory():
    import torch

    from rosclaw_soccer.rsi.categorical_motion import create_categorical_motion

    _, parent = model()
    parent.eval().requires_grad_(False)
    child = create_categorical_motion(parent).eval()
    old_h = new_h = None
    for _ in range(20):
        x = torch.randn(1, 1, 108)
        reference = parent.sample(x, old_h, generator=None, stochastic=False)
        token, value, new_h = child(x, new_h)
        assert torch.equal(token, reference["token"])
        assert torch.equal(value, reference["value"])
        assert torch.equal(new_h, reference["hidden"])
        old_h = reference["hidden"]
    with pytest.raises(ValueError):
        child.network_call_counts()


def test_cross_entropy_updates_only_new_classifier_and_not_old_memory():
    import torch

    from rosclaw_soccer.rsi.categorical_motion import create_categorical_motion

    _, parent = model()
    parent.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    child = create_categorical_motion(parent).train()
    assert not parent.training
    x = torch.randn(2, 8, 108)
    context, classes, _, _ = child.context(x)
    logits = child.logits(context, classes)
    target = (classes + 1).clamp_max(31)
    optimizer = torch.optim.Adam(child.head.parameters(), lr=0.01)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), target.reshape(-1))
    loss.backward()
    assert child.head[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(child.logits(context, classes), logits)
    for k, v in parent.state_dict().items():
        assert torch.equal(v, before[k])
    assert all(p.grad is None for p in parent.parameters())
    with pytest.raises(ValueError):
        child.logits(context * float("nan"), classes)
