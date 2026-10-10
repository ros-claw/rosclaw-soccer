"""Synthetic primitive outcomes do not authenticate physical receipts."""

import copy

import pytest

from rosclaw_soccer.rsi.paired_contact_failure_map import paired_contact_failure_map


def rows():
    return [
        dict(
            index=i,
            outcome=dict(
                first_contact_frame=87,
                clean_foot_only=True,
                high_quality=True,
                maximum_lateral_excursion_m=1.2,
                minimum_pelvis_z_m=0.7,
                forward_60_m=3.2,
                lateral_over_forward_60=0.12,
            ),
        )
        for i in range(52)
    ]


def test_names_local_regression_even_if_another_case_improves():
    old, new = rows(), rows()
    old[30]["outcome"].update(clean_foot_only=False, high_quality=False)
    new[36]["outcome"].update(clean_foot_only=False, high_quality=False, first_contact_frame=76)
    new[50]["outcome"].update(high_quality=False, maximum_lateral_excursion_m=4.4)
    before = copy.deepcopy((old, new))
    result = paired_contact_failure_map(old, new)
    assert result["high_quality_gained_cases"] == [30]
    assert result["high_quality_lost_cases"] == [36, 50]
    assert result["new_out_of_play_cases"] == [50]
    assert result["records"][36]["introduced_failures"] == ["NONFOOT_CONTACT"]
    assert result["records"][36]["first_contact_shift_frames"] == -11
    assert (old, new) == before
    assert all(
        result[k] is False
        for k in (
            "input_receipts_authenticated_here",
            "physical_replay_performed_here",
            "causal_failure_reason_proven",
            "training_data_admitted",
            "fresh_gain_verified",
            "promotion_authorized",
            "hardware_authorized",
        )
    )


@pytest.mark.parametrize("frame", [None, 239, 240, 299])
def test_absent_or_incomplete_post_contact_window(frame):
    old, new = rows(), rows()
    value = new[0]["outcome"]
    value.update(
        first_contact_frame=frame,
        forward_60_m=None,
        lateral_over_forward_60=None,
        high_quality=False,
        clean_foot_only=frame is not None,
    )
    if frame == 239:
        with pytest.raises(ValueError):
            paired_contact_failure_map(old, new)
    else:
        result = paired_contact_failure_map(old, new)
        assert "NO_COMPLETE_POST_CONTACT_WINDOW" in result["records"][0]["candidate_failures"]


def test_safety_separate_from_historical_hq_threshold():
    old, new = rows(), rows()
    new[0]["outcome"]["minimum_pelvis_z_m"] = 0.64
    result = paired_contact_failure_map(old, new)
    assert not result["high_quality_lost_cases"]
    assert result["records"][0]["introduced_failures"] == ["PELVIS_SAFETY_FAILURE"]


def test_exact_historical_thresholds_remain_inclusive():
    old, new = rows(), rows()
    new[0]["outcome"].update(
        minimum_pelvis_z_m=0.65,
        maximum_lateral_excursion_m=4,
        forward_60_m=1,
        lateral_over_forward_60=0.3,
    )
    assert paired_contact_failure_map(old, new)["records"][0]["candidate_failures"] == []


@pytest.mark.parametrize("key", list(rows()[0]["outcome"]))
def test_every_primitive_must_be_explicit_even_when_absent(key):
    old, new = rows(), rows()
    del new[0]["outcome"][key]
    with pytest.raises(ValueError):
        paired_contact_failure_map(old, new)


def test_huge_integer_is_rejected_without_numeric_overflow():
    old, new = rows(), rows()
    new[0]["outcome"]["minimum_pelvis_z_m"] = 10**1000
    with pytest.raises(ValueError):
        paired_contact_failure_map(old, new)


@pytest.mark.parametrize(
    "key,value",
    [
        ("first_contact_frame", True),
        ("first_contact_frame", 300),
        ("first_contact_frame", -1),
        ("high_quality", 1),
        ("clean_foot_only", 1),
        ("maximum_lateral_excursion_m", float("nan")),
        ("maximum_lateral_excursion_m", -1),
        ("minimum_pelvis_z_m", float("inf")),
        ("forward_60_m", None),
        ("forward_60_m", True),
        ("lateral_over_forward_60", float("nan")),
        ("lateral_over_forward_60", -1),
        ("high_quality", False),
    ],
)
def test_reject_invalid_or_inconsistent_primitives(key, value):
    old, new = rows(), rows()
    new[0]["outcome"][key] = value
    with pytest.raises(ValueError):
        paired_contact_failure_map(old, new)


@pytest.mark.parametrize("count", [0, 51, 53])
def test_reject_dropped_or_extra_cases(count):
    with pytest.raises(ValueError):
        paired_contact_failure_map(rows(), (rows() + rows())[:count])


@pytest.mark.parametrize("index", [True, -1, 1])
def test_reject_misaligned_or_boolean_identity(index):
    old, new = rows(), rows()
    new[0]["index"] = index
    with pytest.raises(ValueError):
        paired_contact_failure_map(old, new)
