import pytest

from rosclaw_soccer.rsi.measured_lateral_split import fit_measured_lateral_split


def test_split_uses_gap_between_measured_benefit_and_harm() -> None:
    assert fit_measured_lateral_split((0.764,), (0.775,)) == pytest.approx(0.7695)


@pytest.mark.parametrize(
    "benefit,harm",
    [
        ((), (0.775,)),
        ((0.764,), ()),
        ((0.764,), (0.766,)),
        ((float("nan"),), (0.775,)),
        ((0.764,), (0.76,)),
        ((True,), (0.775,)),
    ],
)
def test_ambiguous_or_invalid_skill_labels_fail_closed(benefit, harm) -> None:
    with pytest.raises(ValueError):
        fit_measured_lateral_split(benefit, harm)
