import pytest
import torch

from rosclaw_soccer.rsi.dual_motor_checkpoint import validate_network, warm_start_binding


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


def test_warm_start_requires_new_protocol_and_unique_separate_predecessor(tmp_path):
    previous = tmp_path / "previous" / "complete.json"
    current = tmp_path / "current" / "complete.json"
    row = dict(
        config=dict(
            protocol="dual256.full_history_dagger_adaptation.v2",
            warm_start_receipt_hash="sha256:previous",
        ),
        input_file_hashes={str(previous): "sha256:previous"},
    )
    assert warm_start_binding(row, current) == (previous, "sha256:previous")
    with pytest.raises(ValueError):
        warm_start_binding(row, previous)
    row["input_file_hashes"][str(tmp_path / "other" / "complete.json")] = "sha256:previous"
    with pytest.raises(ValueError):
        warm_start_binding(row, current)
    row["input_file_hashes"] = {str(previous): "sha256:previous"}
    row["config"]["protocol"] = "dual256.full_history_supervised_adaptation.v1"
    with pytest.raises(ValueError):
        warm_start_binding(row, current)


def test_original_protocol_has_no_warm_start_but_dagger_cannot_omit_it(tmp_path):
    row = dict(config=dict(protocol="dual256.full_history_supervised_adaptation.v1"))
    assert warm_start_binding(row, tmp_path / "complete.json") is None
    row["config"]["protocol"] = "dual256.full_history_dagger_adaptation.v2"
    with pytest.raises(ValueError):
        warm_start_binding(row, tmp_path / "complete.json")
