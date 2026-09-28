"""Contact-risk diagnostic must be deterministic and cannot receive action leakage."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_contact_risk_probe import fit_logistic, predict


def test_regularized_risk_fit_generalizes_simple_direction_signal() -> None:
    x = np.asarray([[-1.0], [-0.8], [-0.6], [-0.4], [0.4], [0.6], [0.8], [1.0]])
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    trained = fit_logistic(x, y, 0.1)
    p = predict(np.asarray([[-0.5], [0.5]]), trained)
    assert p[0] < 0.5 < p[1]
    assert np.array_equal(p, predict(np.asarray([[-0.5], [0.5]]), trained))


def test_risk_fit_rejects_invalid_or_single_class_batches() -> None:
    x = np.ones((8, 2))
    with pytest.raises(ValueError, match="invalid independent"):
        fit_logistic(x, np.ones(8), 0.1)
    with pytest.raises(ValueError, match="invalid independent"):
        fit_logistic(x, np.asarray([0, 0, 0, 0, 1, 1, 1, 1]), 0.123)
