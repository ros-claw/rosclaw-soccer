import copy

import pytest

from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree


def tree():
    import torch

    return dict(
        weights=torch.arange(4, dtype=torch.float32),
        optimizer=dict(
            state={
                0: dict(step=torch.tensor(4.0), exp_avg=torch.zeros(4), exp_avg_sq=torch.ones(4))
            },
            param_groups=[dict(params=[0], lr=0.0001, betas=(0.9, 0.999))],
        ),
        rng=torch.arange(16, dtype=torch.uint8),
        promotion=False,
    )


def test_complete_weights_moments_clock_and_rng_match_exactly():
    original = tree()
    exact_tensor_tree(copy.deepcopy(original), original)


@pytest.mark.parametrize(
    "fault", ["weights", "momentum", "step", "rng", "missing", "dtype", "bool"]
)
def test_nonpersistent_or_altered_numerical_state_rejected(fault):
    import torch

    original = tree()
    changed = copy.deepcopy(original)
    if fault == "weights":
        changed["weights"][0] += 1e-10
    elif fault == "momentum":
        changed["optimizer"]["state"][0]["exp_avg"][0] = 0.001
    elif fault == "step":
        changed["optimizer"]["state"][0]["step"] = torch.tensor(1.0)
    elif fault == "rng":
        changed["rng"][1] = 42
    elif fault == "missing":
        del changed["optimizer"]["param_groups"][0]["betas"]
    elif fault == "dtype":
        changed["weights"] = changed["weights"].double()
    else:
        changed["promotion"] = 0
    with pytest.raises(ValueError):
        exact_tensor_tree(changed, original)


def test_equal_numeric_zero_is_not_equal_tensor_bytes():
    import torch

    with pytest.raises(ValueError):
        exact_tensor_tree(torch.tensor([-0.0]), torch.tensor([0.0]))


def test_nonfinite_or_opaque_objects_cannot_pass():
    import torch

    for value in (torch.tensor([float("nan")]), float("inf"), object()):
        with pytest.raises(ValueError):
            exact_tensor_tree(value, value)
