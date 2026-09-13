import pytest

from rosclaw_soccer.training.held_policy_sample import HeldPolicySample


@pytest.mark.parametrize("period", [1, 2, 5, 10])
def test_exact_values_and_decision_clock_with_private_copies(period):
    torch = pytest.importorskip("torch")
    hold = HeldPolicySample(period)
    for tick in range(23):
        draw = torch.full((2, 3), float(tick))
        value, decision = hold.advance(draw, tick=tick)
        assert decision == (tick % period == 0)
        assert torch.equal(value, torch.full_like(draw, float(tick // period * period)))
        draw.fill_(-999)
        value.fill_(777)


@pytest.mark.parametrize("period", [0, 11, True, 1.0, "5", None])
def test_period_is_explicit_and_bounded(period):
    with pytest.raises(ValueError):
        HeldPolicySample(period)


@pytest.mark.parametrize("tick", [0, 2, -1, True, 1.0])
def test_broken_clock_is_latched_even_after_correct_retry(tick):
    torch = pytest.importorskip("torch")
    hold = HeldPolicySample(5)
    draw = torch.zeros(2, 3)
    hold.advance(draw, tick=0)
    with pytest.raises(ValueError):
        hold.advance(draw, tick=tick)
    with pytest.raises(RuntimeError, match="invalid"):
        hold.advance(draw, tick=1)


@pytest.mark.parametrize("fault", ["nan", "inf", "grad", "dtype", "shape", "not_tensor"])
def test_invalid_unused_draw_also_latches(fault):
    torch = pytest.importorskip("torch")
    hold = HeldPolicySample(5)
    draw = torch.zeros(2, 3)
    hold.advance(draw, tick=0)
    if fault in ("nan", "inf"):
        draw[0, 0] = float(fault)
    elif fault == "grad":
        draw.requires_grad_()
    elif fault == "dtype":
        draw = draw.double()
    elif fault == "shape":
        draw = draw[:1]
    else:
        draw = [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError):
        hold.advance(draw, tick=1)
    with pytest.raises(RuntimeError, match="invalid"):
        hold.advance(torch.zeros(2, 3), tick=1)
