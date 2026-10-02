import pytest

from scripts.rsi_fit_risk_margin_memory_motor import risk_margin_return


def outcome(**changes):
    return dict(
        reward=1.0,
        high_quality=True,
        clean_foot_only=True,
        minimum_pelvis_z_m=0.75,
        maximum_lateral_excursion_m=2.0,
        forward_60_m=2.0,
        lateral_60_m=0.2,
        **changes,
    )


def test_good_margin_has_no_extra_penalty() -> None:
    value, margins = risk_margin_return(outcome())
    assert value == 11
    assert margins["boundary_margin_penalty"] == 0
    assert margins["direction_margin_penalty"] == 0


def test_measured_boundary_and_direction_penalized_without_changing_pass_label() -> None:
    row = outcome()
    row.update(maximum_lateral_excursion_m=3.5, lateral_60_m=0.5)
    value, margins = risk_margin_return(row)
    assert value == pytest.approx(7.75)
    assert margins["boundary_margin_penalty"] == 2
    assert row["high_quality"] is True


def test_actual_miss_retained_not_fabricated_or_omitted() -> None:
    row = outcome()
    row.update(high_quality=False, forward_60_m=None, lateral_60_m=None)
    value, margins = risk_margin_return(row)
    assert value == -4
    assert margins["direction_margin_penalty"] == 5
    assert row["forward_60_m"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("forward_60_m", float("nan")),
        ("lateral_60_m", float("inf")),
        ("maximum_lateral_excursion_m", -1),
        ("forward_60_m", None),
    ],
)
def test_invalid_margins_rejected(key, value) -> None:
    row = outcome()
    row[key] = value
    with pytest.raises(ValueError):
        risk_margin_return(row)


def test_missing_direction_cannot_be_labelled_success() -> None:
    row = outcome()
    row.update(forward_60_m=None, lateral_60_m=None)
    with pytest.raises(ValueError):
        risk_margin_return(row)


def test_runtime_identity_does_not_change_reward() -> None:
    row = outcome()
    other = dict(row, seed=999, lane=12, role="keeper")
    assert risk_margin_return(row) == risk_margin_return(other)
