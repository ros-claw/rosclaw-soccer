import numpy as np
import pytest

from scripts.rsi_diagnose_pass_gradient_conflict import gradient_relation


def test_same_and_opposed_objective_directions_are_not_success_scores():
    g = np.array([1.0, 2.0])
    assert gradient_relation(g, g)["cosine"] == pytest.approx(1.0)
    r = gradient_relation(g, -g)
    assert r["cosine"] == pytest.approx(-1.0) and r["strictly_opposed"]
    z = gradient_relation(g, np.zeros(2))
    assert z["cosine"] is None and not z["strictly_opposed"]


@pytest.mark.parametrize(
    "a,b",
    [
        (np.array([np.nan]), np.ones(1)),
        (np.ones(2), np.ones(1)),
        (np.ones((1, 2)), np.ones((1, 2))),
        (np.array([]), np.array([])),
    ],
)
def test_invalid_gradients_do_not_generate_a_direction(a, b):
    with pytest.raises(ValueError):
        gradient_relation(a, b)
