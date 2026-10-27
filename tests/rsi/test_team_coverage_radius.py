import pytest

from rosclaw_soccer.rsi.team_coverage_radius import radius_below_failure


def test_radius_excludes_failure_and_is_strictly_smaller():
    radius = radius_below_failure(0.0027703853773862804, prior_radius=0.004)
    assert radius == pytest.approx(0.00275)
    assert radius < 0.0027703853773862804
    assert radius_below_failure(0.00275, prior_radius=0.004) < 0.00275


@pytest.mark.parametrize("distance", [0.0, -1.0, 0.0041, float("nan")])
def test_radius_rejects_missing_or_outside_failure(distance):
    with pytest.raises(ValueError):
        radius_below_failure(distance, prior_radius=0.004)
