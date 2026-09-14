"""Signed geometry evidence cannot be substituted by absent contact force."""

import math

import pytest

from rosclaw_soccer.sim.surface_separation_evidence import SurfaceSeparationTracker


def tracker(**overrides):
    kwargs = dict(
        geometry_contract_hash="sha256:" + "a" * 64,
        pair_ids=("ball:right_shin", "ball:left_shin"),
        physics_dt_sec=0.002,
        duration_sec=0.004,
    )
    return SurfaceSeparationTracker(**(kwargs | overrides))


def sample(item, t, d=0.005):
    item.observe(elapsed_sec=t, signed_distances_m={"ball:right_shin": d, "ball:left_shin": 0.2})


def test_complete_positive_margin_and_detached_result():
    item = tracker()
    assert not item.result()["separated_entire_episode"]
    sample(item, 0.002)
    assert not item.result()["complete"]
    sample(item, 0.004, 0.003)
    result = item.result()
    assert result["separated_entire_episode"] and not result["promotion_eligible"]
    result["minimum_signed_distance_m"]["ball:right_shin"] = -100
    assert item.result()["minimum_signed_distance_m"]["ball:right_shin"] == 0.003


@pytest.mark.parametrize("bad_time", [0.002, 0.004])
def test_any_overlap_latches_rejection_even_without_force(bad_time):
    item = tracker()
    for t in (0.002, 0.004):
        sample(item, t, -0.000001 if t == bad_time else 0.1)
    result = item.result()
    assert result["first_clearance_violation_sec"] == bad_time
    assert result["complete"] and not result["separated_entire_episode"]


@pytest.mark.parametrize(
    "required,actual,passed", [(0.0, 0.0, True), (0.003, 0.003, True), (0.003, 0.0029, False)]
)
def test_declared_clearance_comparator(required, actual, passed):
    item = tracker(minimum_clearance_m=required)
    sample(item, 0.002, actual)
    sample(item, 0.004, actual)
    assert item.result()["separated_entire_episode"] == passed


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "0.1"])
def test_nonfinite_or_wrong_type_is_terminal(value):
    item = tracker()
    with pytest.raises(ValueError):
        sample(item, 0.002, value)
    with pytest.raises(ValueError, match="fault-latched"):
        sample(item, 0.002)
    assert item.result()["faulted"] and not item.result()["separated_entire_episode"]


@pytest.mark.parametrize(
    "distances",
    [
        {},
        {"ball:right_shin": 0.1},
        {"ball:right_shin": 0.1, "ball:left_shin": 0.1, "extra": 1.0},
        [],
    ],
)
def test_missing_extra_or_nonmapping_pair_samples_fault(distances):
    item = tracker()
    with pytest.raises(ValueError):
        item.observe(elapsed_sec=0.002, signed_distances_m=distances)
    assert item.result()["faulted"]


@pytest.mark.parametrize(
    "times", [(0.0,), (0.004,), (0.002, 0.002), (0.002, 0.004, 0.006), (True,), (math.nan,)]
)
def test_clock_gaps_duplicates_extra_and_invalid_samples(times):
    item = tracker()
    with pytest.raises(ValueError):
        for t in times:
            sample(item, t)
    assert item.result()["faulted"] and not item.result()["separated_entire_episode"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"pair_ids": ()},
        {"pair_ids": ("same", "same")},
        {"pair_ids": ["x"]},
        {"pair_ids": ("bad space",)},
        {"geometry_contract_hash": "unbound"},
        {"physics_dt_sec": 0},
        {"physics_dt_sec": True},
        {"duration_sec": 0.003},
        {"minimum_clearance_m": -0.001},
        {"minimum_clearance_m": math.nan},
    ],
)
def test_invalid_contracts_rejected(overrides):
    with pytest.raises(ValueError):
        tracker(**overrides)


def test_hash_binds_geometry_clearance_and_pairs_without_pair_order_dependency():
    original = tracker().result()["contract_hash"]
    assert (
        tracker(pair_ids=("ball:left_shin", "ball:right_shin")).result()["contract_hash"]
        == original
    )
    for overrides in (
        {"geometry_contract_hash": "sha256:" + "b" * 64},
        {"minimum_clearance_m": 0.003},
        {"pair_ids": ("ball:left_shin",)},
        {"duration_sec": 0.006},
    ):
        assert tracker(**overrides).result()["contract_hash"] != original
