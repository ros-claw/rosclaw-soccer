import pytest

from rosclaw_soccer.training.hinge_point_jacobian import hinge_point_jacobian


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_signed_axis_order_and_private_output(dtype):
    torch = pytest.importorskip("torch")
    kind = getattr(torch, dtype)
    axis = torch.eye(3, dtype=kind)[None]
    anchor = torch.zeros_like(axis)
    point = torch.tensor([[1.0, 2.0, 3.0]], dtype=kind)
    result = hinge_point_jacobian(axis=axis, anchor=anchor, point=point)
    expected = torch.tensor([[[0, 3, -2], [-3, 0, 1], [2, -1, 0]]], dtype=kind)
    assert torch.equal(result, expected)
    assert result.is_contiguous()
    result.fill_(99)
    assert torch.equal(axis, torch.eye(3, dtype=kind)[None])
    assert torch.equal(anchor, torch.zeros_like(anchor))
    assert torch.equal(point, torch.tensor([[1.0, 2.0, 3.0]], dtype=kind))


def test_translation_invariant_and_matches_rigid_rotation_derivative():
    torch = pytest.importorskip("torch")
    axis = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
    anchor = torch.tensor([[[2.0, -1.0, 5.0]]], dtype=torch.float64)
    point = torch.tensor([[3.0, 1.0, 8.0]], dtype=torch.float64)
    result = hinge_point_jacobian(axis=axis, anchor=anchor, point=point)
    shift = torch.tensor([[[8.0, 9.0, -7.0]]], dtype=torch.float64)
    assert torch.equal(
        result,
        hinge_point_jacobian(axis=axis, anchor=anchor + shift, point=point + shift[:, 0]),
    )
    angle = torch.tensor(1e-6, dtype=torch.float64)
    relative = point[0] - anchor[0, 0]

    def rotate(a):
        c, s = a.cos(), a.sin()
        return torch.stack(
            (c * relative[0] - s * relative[1], s * relative[0] + c * relative[1], relative[2])
        )

    derivative = (rotate(angle) - rotate(-angle)) / (2 * angle)
    torch.testing.assert_close(result[0, :, 0], derivative, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "fault", ["nan", "inf", "unit", "grad", "dtype", "shape", "empty", "huge", "not_tensor"]
)
def test_malformed_geometry_rejected(fault):
    torch = pytest.importorskip("torch")
    axis = torch.tensor([[[0.0, 0.0, 1.0]]])
    anchor = torch.zeros_like(axis)
    point = torch.zeros(1, 3)
    if fault in ("nan", "inf"):
        point[0, 0] = float(fault)
    elif fault == "unit":
        axis *= 2
    elif fault == "grad":
        point.requires_grad_()
    elif fault == "dtype":
        point = point.double()
    elif fault == "shape":
        point = point[0]
    elif fault == "empty":
        axis = axis[:0]
    elif fault == "huge":
        anchor[0, 0, 0] = 1e7
    else:
        axis = [[[0, 0, 1]]]
    with pytest.raises(ValueError):
        hinge_point_jacobian(axis=axis, anchor=anchor, point=point)
