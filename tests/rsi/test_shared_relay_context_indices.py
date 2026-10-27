import pytest

from rosclaw_soccer.rsi.shared_relay_setup import validate_relay_context_indices


def test_explicit_relay_contexts_are_unique_and_in_range():
    validate_relay_context_indices((2, 3, 5, 7), 22)
    validate_relay_context_indices((1, 4, 20, 21), 22)
    for indices in ((2, 2), (True,), (-1,), (22,), (), [2, 3]):
        with pytest.raises(ValueError):
            validate_relay_context_indices(indices, 22)
