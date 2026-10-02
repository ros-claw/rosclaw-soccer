import pytest

from scripts.rsi_retest_risk_margin_out_counterexample import risk_gate


def row():
    return dict(
        high_quality=False,
        clean_foot_only=True,
        maximum_lateral_excursion_m=3.5,
        minimum_pelvis_z_m=0.7,
    )


def test_same_failure_not_out_of_play_can_resolve_only_known_regression():
    assert risk_gate(row(), row())["known_counterexample_resolved"] is True


def test_extra_success_cannot_compensate_for_out_of_play():
    candidate = dict(row(), high_quality=True, maximum_lateral_excursion_m=4.000001)
    assert risk_gate(row(), candidate)["known_counterexample_resolved"] is False


def test_body_or_contact_regression_cannot_be_hidden_by_lower_ball_excursion():
    assert (
        risk_gate(row(), dict(row(), minimum_pelvis_z_m=0.64))["known_counterexample_resolved"]
        is False
    )
    assert (
        risk_gate(row(), dict(row(), clean_foot_only=False))["known_counterexample_resolved"]
        is False
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("maximum_lateral_excursion_m", float("nan")),
        ("minimum_pelvis_z_m", float("inf")),
        ("high_quality", "false"),
        ("maximum_lateral_excursion_m", -1),
    ],
)
def test_invalid_risk_claims_rejected(key, value):
    with pytest.raises(ValueError, match="finite measured"):
        risk_gate(row(), dict(row(), **{key: value}))
