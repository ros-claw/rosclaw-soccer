import math

import pytest

from rosclaw_soccer.world.annular_waypoint import annular_entry_waypoint


def request(**changes):
    args = dict(
        root_xy=(0.3, 0.0),
        contact_xy=(0.0, 0.0),
        entry_offset_xy=(0.0, 0.25),
        final_approach_allowed=False,
    )
    return annular_entry_waypoint(**(args | changes))


def test_inner_root_retreats_without_crossing_contact():
    result = request()
    assert result.phase == "outward"
    assert result.target_xy == pytest.approx((0.5, 0.0))


def test_outer_root_uses_bounded_arc():
    result = request(root_xy=(0.5, 0.0))
    assert result.phase == "orbit"
    assert result.target_xy == pytest.approx((0.5 * math.cos(0.3), 0.5 * math.sin(0.3)))


def test_only_aligned_prepared_root_gets_entry():
    assert request(root_xy=(0.0, 0.5)).target_xy == pytest.approx((0.0, 0.5))
    result = request(root_xy=(0.0, 0.5), final_approach_allowed=True)
    assert result.phase == "approach"
    assert result.target_xy == (0.0, 0.25)
    assert request(final_approach_allowed=True).phase == "outward"


def test_approach_does_not_immediately_reverse_to_retreat():
    assert request(root_xy=(0.0, 0.3), final_approach_allowed=True).phase == "approach"


def test_hysteresis_retains_previous_approach_but_does_not_enter_early():
    root = (0.3 * math.sin(0.2), 0.3 * math.cos(0.2))
    assert request(root_xy=root, final_approach_allowed=True).phase == "outward"
    assert (
        request(root_xy=root, final_approach_allowed=True, approach_committed=True).phase
        == "approach"
    )


def test_hysteresis_exits_on_large_error_or_withdrawn_preparation():
    root = (0.3 * math.sin(0.36), 0.3 * math.cos(0.36))
    assert (
        request(root_xy=root, final_approach_allowed=True, approach_committed=True).phase
        == "outward"
    )
    assert request(root_xy=(0.0, 0.3), approach_committed=True).phase == "outward"


@pytest.mark.parametrize("bad", [1, None, "true"])
def test_hysteresis_cue_requires_explicit_boolean(bad):
    with pytest.raises(ValueError):
        request(approach_committed=bad)


def test_world_translation_and_mirror():
    result = request(root_xy=(0.5, 0.0))
    translated = request(root_xy=(2.5, 1.0), contact_xy=(2.0, 1.0))
    assert translated.target_xy == pytest.approx(
        (result.target_xy[0] + 2.0, result.target_xy[1] + 1.0)
    )
    reflected = request(root_xy=(0.5, 0.0), entry_offset_xy=(0.0, -0.25))
    assert reflected.target_xy == pytest.approx((result.target_xy[0], -result.target_xy[1]))


@pytest.mark.parametrize(
    "changes",
    [
        dict(root_xy=(0.0, 0.0)),
        dict(root_xy=(math.nan, 0.0)),
        dict(contact_xy=(math.inf, 0.0)),
        dict(root_xy=[0.3, 0.0]),
        dict(entry_offset_xy=(0.0, 0.0)),
        dict(entry_offset_xy=(0.5, 0.0)),
        dict(entry_offset_xy=(True, 0.0)),
        dict(radius_m=True),
        dict(radius_m=math.inf),
        dict(radius_m=0.1),
        dict(radius_m=2.1),
        dict(final_approach_allowed=1),
    ],
)
def test_ambiguous_or_invalid_inputs_fail(changes):
    with pytest.raises(ValueError):
        request(**changes)
