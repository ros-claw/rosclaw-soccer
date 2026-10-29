"""Do not silently turn incomplete receiving windows into team capability."""

from dataclasses import replace

import pytest

from rosclaw_soccer.training.receiving_capability import (
    ReceivingAttempt,
    ReceivingAttemptStatus,
    ReceivingObservation,
    receiving_capability_censored_table,
)


def h(i):
    return f"sha256:{i:064x}"


def attempt(i, status):
    obs = ReceivingObservation(
        "red.playmaker",
        0.75,
        0.08,
        h(100),
        h(i),
        status == ReceivingAttemptStatus.SUCCESS,
        status != ReceivingAttemptStatus.UNSAFE,
    )
    return ReceivingAttempt(obs, h(1000 + i), h(2000), status)


def test_censor_is_not_failure_and_bounds_are_only_sample_bounds():
    rows = tuple(attempt(i, status) for i, status in enumerate(ReceivingAttemptStatus, 1))
    table = receiving_capability_censored_table(rows)
    (row,) = table["rows"]
    assert row["status_counts"] == {"success": 1, "failure": 1, "unsafe": 1, "censored": 1}
    assert row["resolved_success_fraction"] == pytest.approx(1 / 3)
    assert row["completion_lower_bound"] == 0.25
    assert row["completion_upper_bound"] == 0.5
    assert table["calibrated_probability_model"] is False
    assert table["episode_independence_authenticated"] is False
    assert table["hardware_authorized"] is False


def test_all_censored_never_fabricates_an_observed_rate():
    (row,) = receiving_capability_censored_table((attempt(1, ReceivingAttemptStatus.CENSORED),))[
        "rows"
    ]
    assert row["resolved_success_fraction"] is None
    assert (row["completion_lower_bound"], row["completion_upper_bound"]) == (0.0, 1.0)


def test_different_files_of_same_episode_do_not_count_twice():
    first = attempt(1, ReceivingAttemptStatus.FAILURE)
    replay = replace(attempt(2, ReceivingAttemptStatus.FAILURE), episode_hash=first.episode_hash)
    with pytest.raises(ValueError, match="one attempt per episode"):
        receiving_capability_censored_table((first, replay))


def test_censored_cannot_hide_known_capture_or_unsafe_contact():
    success = attempt(1, ReceivingAttemptStatus.SUCCESS)
    with pytest.raises(ValueError):
        replace(success, status=ReceivingAttemptStatus.CENSORED)
    unsafe = attempt(2, ReceivingAttemptStatus.UNSAFE)
    with pytest.raises(ValueError):
        replace(unsafe, status=ReceivingAttemptStatus.CENSORED)


def test_policy_and_evidence_provenance_must_not_be_mixed():
    first = attempt(1, ReceivingAttemptStatus.SUCCESS)
    second = attempt(2, ReceivingAttemptStatus.FAILURE)
    with pytest.raises(ValueError, match="different policies"):
        receiving_capability_censored_table(
            (first, replace(second, observation=replace(second.observation, policy_hash=h(101))))
        )
    with pytest.raises(ValueError, match="duplicate evidence"):
        receiving_capability_censored_table(
            (
                first,
                replace(
                    second,
                    observation=replace(
                        second.observation, evidence_hash=first.observation.evidence_hash
                    ),
                ),
            )
        )


def test_invalid_status_does_not_fall_through_to_success_or_failure():
    with pytest.raises(ValueError):
        replace(attempt(1, ReceivingAttemptStatus.FAILURE), status="unknown")


def test_distinct_contexts_are_never_aggregated():
    first = attempt(1, ReceivingAttemptStatus.SUCCESS)
    second = replace(attempt(2, ReceivingAttemptStatus.FAILURE), context_hash=h(2001))
    assert len(receiving_capability_censored_table((first, second))["rows"]) == 2
