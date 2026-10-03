import copy

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_summarize_audited_failure_curriculum import (
    check_dataset_shape,
    outcome_labels,
    summarize,
)


def outcome(**changes):
    return {
        "contact_body_indices": [0],
        "first_contact_frame": 70,
        "clean_foot_only": True,
        "high_quality": True,
        "minimum_pelvis_z_m": 0.7,
        "maximum_lateral_excursion_m": 0.5,
        "forward_60_m": 2.0,
        "lateral_60_m": 0.1,
        **changes,
    }


def seal(value):
    value.pop("report_hash", None)
    value["report_hash"] = hash_json(value)
    return value


def manifest():
    records = []
    for seed in (100, 200):
        for sample in (0, 1):
            measured = outcome()
            if seed == 200:
                measured = outcome(
                    high_quality=False,
                    forward_60_m=0.5,
                    lateral_60_m=1.0,
                    maximum_lateral_excursion_m=4.5,
                    minimum_pelvis_z_m=0.6,
                )
            records.append(
                dict(group=len(records), seed=seed, lane=0, sample=sample, outcome=measured)
            )
    return seal(
        dict(
            schema="soccer.rsi.smooth_memory_on_policy_bank.v1",
            partition="TRAIN_CONSUMED",
            sampling_rho=0.9,
            candidate_previous_mean_required=True,
            records=records,
            physical_rollout_count=4,
            frame_sample_count=1080,
            independent_contexts=2,
            data_hash="sha256:" + "a" * 64,
            promotion_authorized=False,
            hardware_authorized=False,
        )
    )


def test_counts_include_every_failure_and_do_not_create_physical_evidence():
    data = manifest()
    original = copy.deepcopy(data)
    result = summarize(data)
    assert data == original
    assert result["physical_rollout_count"] == 4
    assert result["high_quality"] == 2
    assert result["zero_success_courses"] == [[200, 0]]
    assert result["rows"][1]["overlapping_failure_counts"] == {
        "EXCESS_POSTCONTACT_DIRECTION_RATIO": 2,
        "INSUFFICIENT_FORWARD_DISPLACEMENT": 2,
        "LOW_PELVIS": 2,
        "OUT_OF_PLAY": 2,
    }
    assert result["physical_executions_added"] == 0
    assert result["curriculum_changed"] is False
    assert result["promotion_authorized"] is result["hardware_authorized"] is False


def test_quality_does_not_hide_stability_regression():
    data = manifest()
    data["records"][0]["outcome"]["minimum_pelvis_z_m"] = 0.6
    result = summarize(seal(data))
    assert result["rows"][0]["high_quality"] == 2
    assert result["rows"][0]["high_quality_and_safe"] == 1


def test_dataset_must_cover_all_samples_in_order():
    data = manifest()
    groups = np.repeat(np.arange(4), 270)
    check_dataset_shape(data, groups, (1080, 134))
    with pytest.raises(ValueError):
        check_dataset_shape(data, groups, (540, 134))
    with pytest.raises(ValueError):
        check_dataset_shape(data, groups.astype(float), (1080, 134))
    with pytest.raises(ValueError):
        check_dataset_shape(data, groups[::-1], (1080, 134))
    data["physical_rollout_count"] = 2
    with pytest.raises(ValueError):
        check_dataset_shape(data, groups, (1080, 134))


@pytest.mark.parametrize("first", [None, 240, 299])
def test_missing_horizon_is_not_zero_direction_error(first):
    measured = outcome(
        contact_body_indices=[] if first is None else [0],
        first_contact_frame=first,
        clean_foot_only=first is not None,
        high_quality=False,
        forward_60_m=None,
        lateral_60_m=None,
    )
    labels = outcome_labels(measured)
    assert "NO_COMPLETE_POSTCONTACT_HORIZON" in labels
    assert "EXCESS_POSTCONTACT_DIRECTION_RATIO" not in labels
    assert ("NO_CONTACT" in labels) == (first is None)


def test_illegal_contact_label_is_not_confused_with_missing_contact():
    measured = outcome(contact_body_indices=[0, 5], clean_foot_only=False, high_quality=False)
    assert outcome_labels(measured) == ["NONFOOT_CONTACT"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("promotion_authorized", True),
        ("hardware_authorized", 0),
        ("partition", "FRESH"),
        ("physical_rollout_count", 3),
        ("frame_sample_count", 1079),
        ("independent_contexts", 3),
        ("physical_rollout_count", True),
    ],
)
def test_authority_partial_counts_and_wrong_partition_are_rejected(key, value):
    data = manifest()
    data[key] = value
    with pytest.raises(ValueError):
        summarize(seal(data))


def test_stale_seal_is_rejected():
    data = manifest()
    data["records"][0]["sample"] = 9
    with pytest.raises(ValueError):
        summarize(data)


@pytest.mark.parametrize("key,value", [("group", 1), ("sample", 1), ("seed", True)])
def test_resealed_duplicate_or_invalid_identity_is_rejected(key, value):
    data = manifest()
    data["records"][0][key] = value
    with pytest.raises(ValueError):
        summarize(seal(data))


@pytest.mark.parametrize(
    "changes",
    [
        {"high_quality": False},
        {"clean_foot_only": 1},
        {"contact_body_indices": [0, 0]},
        {"contact_body_indices": [True]},
        {"contact_body_indices": []},
        {"first_contact_frame": True},
        {"forward_60_m": None},
        {"minimum_pelvis_z_m": float("nan")},
        {"maximum_lateral_excursion_m": -1},
        {"lateral_60_m": float("inf")},
    ],
)
def test_bad_measurements_cannot_become_success(changes):
    with pytest.raises(ValueError):
        outcome_labels(outcome(**changes))
