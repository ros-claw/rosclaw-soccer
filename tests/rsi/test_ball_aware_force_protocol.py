import pytest

from rosclaw_soccer.rsi.velocity_command_protocol import ball_aware_force_protocol


@pytest.mark.parametrize("value", ["stock", "addressed"])
def test_matching_explicit_protocol(value):
    assert ball_aware_force_protocol({"force_decoder": value}, {"force_decoder": value}) == value


def test_legacy_missing_is_only_unqualified_stock():
    assert ball_aware_force_protocol({}, {}) == "stock"


@pytest.mark.parametrize(
    "report,checkpoint",
    [
        ({"force_decoder": "addressed"}, {}),
        ({}, {"force_decoder": "addressed"}),
        ({"force_decoder": "unknown"}, {"force_decoder": "unknown"}),
        ({"force_decoder": None}, {}),
    ],
)
def test_mismatching_or_unknown_protocol_is_rejected(report, checkpoint):
    with pytest.raises(ValueError, match="decoder required"):
        ball_aware_force_protocol(report, checkpoint)
