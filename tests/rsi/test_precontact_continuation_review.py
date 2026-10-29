import numpy as np
import pytest

from scripts.rsi_review_precontact_continuation import paired_health


def test_health_comparison_uses_common_horizon_and_preserves_censoring():
    first, second = np.zeros((5, 2), bool), np.zeros((3, 2), bool)
    first[4, 0] = True
    assert paired_health(first, second) == dict(
        common_controls=3, horizons=[5, 3], healthy=[True, True]
    )
    second[2, 1] = True
    assert paired_health(first, second)["healthy"] == [True, False]
    with pytest.raises(ValueError):
        paired_health(first.astype(float), second)
