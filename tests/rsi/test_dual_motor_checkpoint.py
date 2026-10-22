import pytest
import torch

from rosclaw_soccer.rsi.dual_motor_checkpoint import validate_network


def expected():
    return dict(mean=torch.zeros(1, 160), denominator=torch.ones(1, 160), weight=torch.zeros(2, 3))


def test_exact_cpu_network_accepted():
    x = expected()
    validate_network({k: v.clone() for k, v in x.items()}, x)


@pytest.mark.parametrize("fault", ["nan", "shape", "dtype", "normalization", "keys"])
def test_bad_weights_or_normalization_rejected(fault):
    x = expected()
    y = {k: v.clone() for k, v in x.items()}
    if fault == "nan":
        y["weight"][0, 0] = float("nan")
    if fault == "shape":
        y["weight"] = torch.zeros(3, 2)
    if fault == "dtype":
        y["weight"] = y["weight"].double()
    if fault == "normalization":
        y["mean"][0, 0] = 1.0
    if fault == "keys":
        y["extra"] = torch.zeros(1)
    with pytest.raises(ValueError):
        validate_network(y, x)
