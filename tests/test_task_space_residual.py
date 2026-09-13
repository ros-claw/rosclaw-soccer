import pytest

from rosclaw_soccer.providers.g1.task_space_residual import apply_task_space_residual

torch = pytest.importorskip("torch")


def inputs():
    return dict(
        raw=torch.zeros(2, 3),
        jacobian=torch.ones(2, 3, 6) * 0.1,
        foot=torch.tensor([0, 1]),
        launch_direction=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        kp=torch.ones(2, 29) * 60,
        nominal=torch.ones(2, 29) * 0.01,
        previous=torch.zeros(2, 29),
        closed=torch.zeros(2, dtype=torch.bool),
    )


def test_zero_and_closed_force_preserve_nominal_exactly():
    data = inputs()
    for closed in (False, True):
        data["closed"].fill_(closed)
        if closed:
            data["raw"].fill_(20)
        result, force = apply_task_space_residual(**data)
        assert torch.equal(result, data["nominal"])
        assert torch.count_nonzero(force) == 0
        result.fill_(999)
        assert data["nominal"].max() == 0.01


def test_nonzero_force_only_changes_selected_leg_and_preserves_caps():
    data = inputs()
    data["raw"].fill_(100)
    result, force = apply_task_space_residual(**data)
    assert torch.linalg.vector_norm(force, dim=1).max() <= 60.00001
    assert result.abs().max() <= 0.250001
    assert (result - data["previous"]).abs().max() <= 0.025001
    for world, selected in enumerate((slice(0, 6), slice(6, 12))):
        assert not torch.equal(result[world, selected], data["nominal"][world, selected])
        mask = torch.ones(29, dtype=torch.bool)
        mask[selected] = False
        assert torch.equal(result[world, mask], data["nominal"][world, mask])


@pytest.mark.parametrize("fault", ["nan", "foot", "kp", "axis", "jacobian", "rate", "grad"])
def test_bad_mapping_rejected(fault):
    data = inputs()
    if fault == "nan":
        data["raw"][0, 0] = float("nan")
    elif fault == "foot":
        data["foot"][0] = 2
    elif fault == "kp":
        data["kp"][0, 0] = 1e-30
    elif fault == "axis":
        data["launch_direction"][0] = 0
    elif fault == "jacobian":
        data["jacobian"][0, 0, 0] = 3
    elif fault == "rate":
        data["nominal"][0, 0] = 0.1
    else:
        data["raw"].requires_grad_()
    with pytest.raises(ValueError):
        apply_task_space_residual(**data)
