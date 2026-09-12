from dataclasses import replace

import pytest

from rosclaw_soccer.training.pass_replica_screen import PassReplica, screen_pass_replicas


def group():
    return tuple(
        PassReplica(
            i, "sha256:" + "a" * 64, "sha256:" + "b" * 64, True, True, False, False, True, 0.05
        )
        for i in range(4)
    )


def test_precision_boundary_and_complete_denominator():
    rows = group()
    rows = (replace(rows[0], plane_error_m=0.1), *rows[1:3], replace(rows[3], plane_error_m=None))
    result = screen_pass_replicas(rows)
    assert (result.attempted, result.eligible, result.precise) == (4, 4, 3)
    assert result.screening_passed and result.failed_replica_ids == (3,)


@pytest.mark.parametrize(
    "changes",
    [
        {"body_safe": False},
        {"ball_in_play": False},
        {"nonfoot_contact": True},
        {"late_contact": True},
    ],
)
def test_one_constraint_failure_rejects_despite_three_precise(changes):
    rows = group()
    result = screen_pass_replicas((replace(rows[0], **changes), *rows[1:]))
    assert result.precise == 3 and not result.screening_passed


def test_two_lucky_successes_are_not_robust_and_no_crossing_is_failure():
    rows = group()
    result = screen_pass_replicas(
        (*rows[:2], replace(rows[2], plane_error_m=None), replace(rows[3], plane_error_m=0.100001))
    )
    assert result.precise == 2 and not result.screening_passed
    assert result.failed_replica_ids == (2, 3)


@pytest.mark.parametrize("field", ["candidate_hash", "context_hash"])
def test_mixing_candidates_or_contexts_rejected(field):
    rows = group()
    with pytest.raises(ValueError, match="same candidate"):
        screen_pass_replicas((replace(rows[0], **{field: "sha256:" + "c" * 64}), *rows[1:]))


def test_dropped_duplicate_or_mutable_replicas_rejected():
    rows = group()
    for invalid in (rows[:3], (rows[1], *rows[1:]), list(rows)):
        with pytest.raises(ValueError):
            screen_pass_replicas(invalid)
    assert screen_pass_replicas(tuple(reversed(rows))) == screen_pass_replicas(rows)


@pytest.mark.parametrize("error", [float("nan"), float("inf"), -0.01, True, "0.01", 1001, 10**400])
def test_invalid_errors_never_count_as_success(error):
    with pytest.raises(ValueError):
        replace(group()[0], plane_error_m=error)


@pytest.mark.parametrize(
    "kwargs", [{"expected_replicas": True}, {"required_precise": 0}, {"required_precise": 5}]
)
def test_invalid_group_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        screen_pass_replicas(group(), **kwargs)


def test_precision_requires_controlled_pass():
    rows = group()
    result = screen_pass_replicas(tuple(replace(row, controlled_pass=False) for row in rows))
    assert result.precise == 0 and not result.screening_passed


@pytest.mark.parametrize("kwargs", [{"body_safe": 1}, {"replica_id": True}, {"candidate_hash": ""}])
def test_invalid_replica_fields_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(group()[0], **kwargs)
