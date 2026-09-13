import pytest

from rosclaw_soccer.providers.g1.task_space_navigation import apply_task_space_navigation

torch = pytest.importorskip("torch")


def arguments():
    return dict(
        raw=torch.zeros(2, 3),
        nominal=torch.tensor([[0.005, 0.004, 0.02], [-0.003, 0.001, -0.01]]),
        previous=torch.zeros(2, 3),
        closed=torch.tensor([False, False]),
    )


def test_zero_and_closed_exactly_preserve_nominal_with_private_output():
    args = arguments()
    result = apply_task_space_navigation(**args)
    assert torch.equal(result, args["nominal"])
    result.zero_()
    assert args["nominal"].abs().sum() > 0
    args["raw"].fill_(100)
    args["closed"].fill_(True)
    assert torch.equal(apply_task_space_navigation(**args), args["nominal"])


def test_original_absolute_and_vector_slew_bounds():
    args = arguments()
    args["raw"] = torch.tensor([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]])
    for _ in range(100):
        result = apply_task_space_navigation(**args)
        delta = result - args["previous"]
        assert (torch.linalg.vector_norm(delta[:, :2], dim=1) <= 0.012001).all()
        assert (delta[:, 2].abs() <= 0.040001).all()
        assert (result.abs() <= torch.tensor([0.7, 0.7, 0.8]) + 1e-7).all()
        args["previous"] = result
        args["nominal"] = result.clone()


@pytest.mark.parametrize(
    "fault", ["nan", "inf", "grad", "dimension", "mask", "nominal_slew", "nominal_limit", "dtype"]
)
def test_invalid_navigation_contract_rejected(fault):
    args = arguments()
    if fault in ("nan", "inf"):
        args["raw"][0, 0] = float(fault)
    elif fault == "grad":
        args["raw"].requires_grad_()
    elif fault == "dimension":
        args["raw"] = torch.zeros(2, 6)
    elif fault == "mask":
        args["closed"] = torch.ones(2)
    elif fault == "nominal_slew":
        args["nominal"].fill_(0.1)
    elif fault == "nominal_limit":
        args["nominal"].fill_(1)
        args["previous"].fill_(1)
    else:
        args["nominal"] = args["nominal"].double()
    with pytest.raises(ValueError):
        apply_task_space_navigation(**args)


def test_closed_mask_is_per_world():
    args = arguments()
    args["raw"].fill_(1)
    args["closed"][0] = True
    result = apply_task_space_navigation(**args)
    assert torch.equal(result[0], args["nominal"][0])
    assert not torch.equal(result[1], args["nominal"][1])
