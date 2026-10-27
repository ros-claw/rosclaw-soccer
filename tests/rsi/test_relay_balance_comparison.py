import pytest

from rosclaw_soccer.rsi.relay_balance_comparison import compare_balance_outcomes


def test_matched_rescue_regression_never_cancel():
    def result(success, failure, kick):
        return dict(
            return_0p1m_and_healthy=success, body_failure=failure, outgoing_foot_strike=kick
        )

    result1 = compare_balance_outcomes(
        [
            (result(False, True, False), result(True, False, True)),
            (result(True, False, True), result(False, True, False)),
        ]
    )
    assert result1["precise_rescues"] == result1["precise_regressions"] == 1
    assert result1["body_failure_rescues"] == result1["body_failure_regressions"] == 1
    with pytest.raises(ValueError):
        compare_balance_outcomes([({}, {})])
