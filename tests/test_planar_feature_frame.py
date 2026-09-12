"""Vector basis changes never implicitly change local quantities or latents."""

import pytest

from rosclaw_soccer.training.planar_feature_frame import rotate_planar_feature_pairs

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse_and_untouched_columns(dtype):
    features = torch.arange(24, dtype=dtype).reshape(3, 8)
    before = features.clone()
    angles = torch.tensor([0.0, 0.7, -2.0], dtype=dtype)
    rotated = rotate_planar_feature_pairs(features, angles, pairs=((0, 1), (5, 7)))
    restored = rotate_planar_feature_pairs(rotated, -angles, pairs=((0, 1), (5, 7)))
    assert torch.equal(features, before)
    assert torch.equal(rotated[:, [2, 3, 4, 6]], features[:, [2, 3, 4, 6]])
    torch.testing.assert_close(restored, features)
    assert torch.equal(rotated[0], features[0])


def test_quarter_turn_and_gradients():
    features = torch.tensor([[1.0, 0.0, 5.0]], dtype=torch.float64, requires_grad=True)
    angles = torch.tensor([torch.pi / 2], dtype=torch.float64, requires_grad=True)
    result = rotate_planar_feature_pairs(features, angles, pairs=((0, 1),))
    torch.testing.assert_close(result, torch.tensor([[0.0, 1.0, 5.0]], dtype=torch.float64))
    result.sum().backward()
    assert torch.isfinite(features.grad).all() and torch.isfinite(angles.grad).all()


@pytest.mark.parametrize(
    "pairs",
    [(), [], ((0, 0),), ((0, 1), (1, 2)), ((True, 1),), ((-1, 1),), ((0, 3),), ((0,),), ([0, 1],)],
)
def test_reject_ambiguous_columns(pairs):
    with pytest.raises(ValueError):
        rotate_planar_feature_pairs(torch.zeros(1, 3), torch.zeros(1), pairs=pairs)


@pytest.mark.parametrize("angle", [float("nan"), float("inf"), 3.2, -3.2])
def test_reject_invalid_angles(angle):
    with pytest.raises(ValueError):
        rotate_planar_feature_pairs(torch.zeros(1, 3), torch.tensor([angle]), pairs=((0, 1),))


def test_reject_nonfinite_or_misaligned_inputs():
    for features, angles in [
        (torch.tensor([[float("nan"), 0.0]]), torch.zeros(1)),
        (torch.zeros(1, 2), torch.zeros(2)),
        (torch.zeros(1, 2), torch.zeros(1, dtype=torch.float64)),
        (torch.zeros(1, 2, dtype=torch.int64), torch.zeros(1)),
        (torch.zeros(0, 2), torch.zeros(0)),
        (torch.zeros(1, 2), [0.0]),
    ]:
        with pytest.raises(ValueError):
            rotate_planar_feature_pairs(features, angles, pairs=((0, 1),))
