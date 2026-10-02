import pytest

from scripts.rsi_continue_memory_validation import transport_flags


def test_plain_legacy_continuation_remains_explicit_default():
    assert transport_flags(compressed=False, shared=False, proof=False, reserve=False) == []


def test_shared_requires_complete_actual_transport_and_storage_boundary():
    assert transport_flags(compressed=True, shared=True, proof=True, reserve=True) == [
        "--compressed-reports",
        "--shared-model-reports",
    ]


@pytest.mark.parametrize(
    "values",
    [
        (False, True, False, False),
        (True, True, False, False),
        (True, False, True, False),
        (False, False, True, True),
        (1, False, False, False),
    ],
)
def test_partial_or_nonboolean_transport_options_fail_closed(values):
    with pytest.raises(ValueError):
        transport_flags(
            **dict(zip(("compressed", "shared", "proof", "reserve"), values, strict=True))
        )
