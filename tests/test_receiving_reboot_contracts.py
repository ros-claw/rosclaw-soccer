from dataclasses import replace

import pytest

from rosclaw_soccer.training.receiving_capability import (
    ReceivingObservation,
    receiving_capability_table,
)
from rosclaw_soccer.training.receiving_reachability import (
    SUBSTRATES,
    ReceivingOracleOutcome,
    compare_receiving_oracles,
)


def h(i):
    return f"sha256:{i:064x}"


def outcomes():
    return tuple(
        ReceivingOracleOutcome(
            f"case-{i}",
            s,
            h(i + 1),
            h(100),
            h(101),
            h(102 + j),
            h(1000 + 4 * i + j),
            i,
            16,
            16,
            i % 2 == 0,
            True,
            True,
        )
        for i in range(64)
        for j, s in enumerate(SUBSTRATES)
    )


CASES = tuple(f"case-{i}" for i in range(64))


def test_paired_counts_do_not_authorize_training():
    result = compare_receiving_oracles(outcomes(), case_ids=CASES)
    assert result["counts"]["A0_leg12"]["safe_captures"] == 32
    assert not result["training_authorized"] and not result["promotion_authorized"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_state_hash", h(900)),
        ("physics_hash", h(900)),
        ("evaluation_hash", h(900)),
        ("optimizer_seed", 999),
        ("rollout_budget", 17),
        ("strict_replay", False),
        ("evidence_hash", h(1000)),
    ],
)
def test_mismatched_pair_rejected(field, value):
    rows = list(outcomes())
    rows[1] = replace(rows[1], **{field: value})
    with pytest.raises(ValueError):
        compare_receiving_oracles(tuple(rows), case_ids=CASES)


def test_missing_adapter_is_not_zero_success():
    with pytest.raises(ValueError):
        compare_receiving_oracles(outcomes()[:-1], case_ids=CASES)


def test_unsafe_capture_not_success():
    rows = list(outcomes())
    rows[0] = replace(rows[0], safe=False)
    result = compare_receiving_oracles(tuple(rows), case_ids=CASES)
    assert result["counts"]["A0_leg12"]["safe_captures"] == 31
    assert result["counts"]["A0_leg12"]["unsafe_cases"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("captured", 1),
        ("safe", "true"),
        ("optimizer_seed", True),
        ("rollouts_used", 17),
        ("rollout_budget", 0),
        ("evidence_hash", "bad"),
    ],
)
def test_bad_outcome_rejected(field, value):
    with pytest.raises(ValueError):
        replace(outcomes()[0], **{field: value})


def observation(i=1, **kwargs):
    return replace(
        ReceivingObservation("red.playmaker", 0.75, 0.08, h(10), h(i), True, True), **kwargs
    )


def test_capability_keeps_unknowns_and_counts():
    result = receiving_capability_table((observation(), observation(2, captured=False)))
    assert result["rows"][0]["observed_fraction"] == 0.5
    assert result["rows"][0]["readiness_after"] is None
    assert result["unseen_context"] == "UNKNOWN"
    assert not result["calibrated_probability_model"]


def test_no_replay_double_count_or_mixed_policy():
    for rows in (
        (observation(), observation()),
        (observation(), observation(2, policy_hash=h(11))),
    ):
        with pytest.raises(ValueError):
            receiving_capability_table(rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("speed_mps", float("nan")),
        ("lateral_m", float("inf")),
        ("captured", 1),
        ("speed_mps", True),
        ("agent_id", "unknown"),
    ],
)
def test_bad_capability_input(field, value):
    with pytest.raises(ValueError):
        observation(**{field: value})
