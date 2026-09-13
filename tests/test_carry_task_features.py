from dataclasses import replace

import pytest

from rosclaw_soccer.training.ball_carry_credit import advance_ball_carry, begin_ball_carry
from rosclaw_soccer.training.carry_task_features import (
    CARRY_TASK_FEATURE_NAMES,
    carry_task_features,
)

torch = pytest.importorskip("torch")


def measured():
    ball = torch.tensor([[0.4, 0.0, 0.115]])
    root = torch.tensor([[0.0, 0.0, 0.75]])
    h = begin_ball_carry(
        ball_position=ball, root_position=root, direction=torch.tensor([[1.0, 0.0]])
    )
    return h, ball, root, torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def features(h, b, r, q):
    return carry_task_features(h, ball_position=b, root_position=r, root_quaternion=q)


def test_initial_causal_features_are_bounded_private_and_named():
    h, b, r, q = measured()
    out = features(h, b, r, q)
    assert len(CARRY_TASK_FEATURE_NAMES) == 12 and out.shape == (1, 12)
    assert torch.all(out.abs() <= 1)
    torch.testing.assert_close(out[0, :4], torch.zeros(4))
    assert out[0, 8] == out[0, 10] == 1
    assert out[0, 7] == out[0, 9] == out[0, 11] == 0
    out.fill_(0)
    assert h.clear_ticks.item() == 20 and h.maximum_separation.item() > 0


def test_signed_lane_progress_heading_and_contact_history():
    h, b, r, q = measured()
    b = b + torch.tensor([[0.5, -0.2, 0.0]])
    r = r + torch.tensor([[0.4, -0.1, 0.0]])
    h = advance_ball_carry(
        h,
        ball_position=b,
        root_position=r,
        foot_contact=torch.tensor([True]),
        nonfoot_contact=torch.tensor([False]),
        body_valid=torch.tensor([True]),
        in_play=torch.tensor([True]),
        tick=1,
        elapsed_sec=0.002,
    )
    q = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
    out = features(h, b, r, q)
    torch.testing.assert_close(out[0, :4], torch.tensor([0.1, 0.08, -0.04, -0.02]))
    assert out[0, 7] == 0.25 and out[0, 8] == 0
    torch.testing.assert_close(out[0, 9:11], torch.tensor([-1.0, 0.0]), atol=1e-6, rtol=0)


def test_equal_current_geometry_different_task_origin_is_observable():
    h, b, r, q = measured()
    other = replace(
        h,
        initial_ball=h.initial_ball - torch.tensor([[0.2, 0.0, 0.0]]),
        forward_progress=torch.tensor([0.2]),
    )
    assert not torch.equal(features(h, b, r, q), features(other, b, r, q))


def test_global_translation_invariance():
    h, b, r, q = measured()
    offset = torch.tensor([[2.0, -3.0, 0.0]])
    other = begin_ball_carry(
        ball_position=b + offset, root_position=r + offset, direction=h.direction
    )
    torch.testing.assert_close(features(h, b, r, q), features(other, b + offset, r + offset, q))


@pytest.mark.parametrize(
    "fault", ["nan", "grad", "shape", "dtype", "stale", "quaternion", "vertical"]
)
def test_invalid_or_stale_measurements_rejected(fault):
    h, b, r, q = measured()
    if fault == "nan":
        b[0, 0] = float("nan")
    elif fault == "grad":
        b.requires_grad_(True)
    elif fault == "shape":
        b = b[:, :2]
    elif fault == "dtype":
        b = b.double()
    elif fault == "stale":
        b[0, 0] += 0.1
    elif fault == "quaternion":
        q *= 2
    else:
        q = torch.tensor([[2**-0.5, 0.0, 2**-0.5, 0.0]])
    with pytest.raises(ValueError):
        features(h, b, r, q)
