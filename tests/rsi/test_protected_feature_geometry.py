import pytest
import torch

from rosclaw_soccer.rsi.protected_feature_geometry import protected_feature_geometry


def test_separation_is_feature_based_and_duplicate_conflicts_are_not_hidden():
    protected = torch.zeros(3, 288)
    protected[:, 0] = torch.tensor([-1.0, 0.0, 1.0])
    corrective = torch.zeros(2, 288)
    corrective[:, 0] = torch.tensor([0.0, 0.5])
    result = protected_feature_geometry(protected, corrective)
    assert result["separable"].tolist() == [False, True]
    assert result["prospective_radius"][0] == 0
    assert torch.equal(result["prospective_radius"], result["nearest_protected_distance"] * 0.45)
    assert torch.equal(corrective[:, 0], torch.tensor([0.0, 0.5]))


def test_nearest_search_covers_later_blocks_and_retains_float64_small_distances():
    protected = torch.zeros(520, 288)
    protected[519, 0] = 1
    corrective = protected[519:520].clone()
    corrective[0, 0] += 1e-5
    result = protected_feature_geometry(protected, corrective)
    assert result["nearest_protected_index"].item() == 519
    assert result["nearest_protected_distance"].dtype == torch.float64
    assert result["separable"].item()


def test_large_common_feature_offset_does_not_hide_later_exact_duplicate():
    protected = torch.full((520, 288), 10000.0)
    protected[519, 0] += 0.1
    result = protected_feature_geometry(protected, protected[519:520].clone())
    assert result["nearest_protected_index"].item() == 519
    assert result["nearest_protected_distance"].item() == 0
    assert not result["separable"].item()


@pytest.mark.parametrize("fault", ["empty", "NaN", "wide", "dtype"])
def test_bad_feature_contract_is_rejected(fault):
    p, c = torch.zeros(2, 288), torch.zeros(1, 288)
    if fault == "empty":
        c = c[:0]
    elif fault == "NaN":
        c[0, 0] = float("nan")
    elif fault == "wide":
        c = torch.zeros(1, 289)
    else:
        c = c.double()
    with pytest.raises(ValueError):
        protected_feature_geometry(p, c)
