import pytest

from rosclaw_soccer.rsi.receiver_strict_terminal_reward import (
    admit_strict_training,
    strict_terminal_reward,
)


def outcome(**overrides):
    return {
        "strict_both_clean_handoff": False,
        "strict_receiver_clean_handoff": False,
        "passer_foot_precedes_receiver": False,
        "both_bodies_healthy": True,
        "original_handoff": False,
        "passer_nonfoot_substep": None,
        "receiver_nonfoot_substep": None,
        **overrides,
    }


def test_reward_requires_real_clean_success_and_penalizes_nonfoot():
    assert strict_terminal_reward(outcome()) == 0
    assert (
        strict_terminal_reward(
            outcome(
                strict_both_clean_handoff=True,
                strict_receiver_clean_handoff=True,
                passer_foot_precedes_receiver=True,
                original_handoff=True,
            )
        )
        == 2
    )
    assert strict_terminal_reward(outcome(original_handoff=True, receiver_nonfoot_substep=7)) == -1
    assert (
        strict_terminal_reward(outcome(both_bodies_healthy=False, passer_nonfoot_substep=5)) == -2
    )


def test_rejects_inconsistent_or_missing_contact_data():
    with pytest.raises(ValueError):
        strict_terminal_reward(
            outcome(
                strict_both_clean_handoff=True, original_handoff=True, receiver_nonfoot_substep=0
            )
        )
    value = outcome()
    del value["passer_nonfoot_substep"]
    with pytest.raises(ValueError):
        strict_terminal_reward(value)


def test_admission_counts_distinct_conditions_not_seed_repetitions():
    records = []
    for context in range(12):
        for side in (-0.2, 0.2):
            for seed in (92117, 92143, 92171):
                current = outcome()
                if context < 2:
                    current.update(
                        strict_both_clean_handoff=True,
                        strict_receiver_clean_handoff=True,
                        passer_foot_precedes_receiver=True,
                        original_handoff=True,
                    )
                elif context < 4:
                    current.update(receiver_nonfoot_substep=2)
                records.append(
                    dict(
                        context=context,
                        side=side,
                        seed=seed,
                        physical_clean_outcome=current,
                        terminal_reward=strict_terminal_reward(current),
                    )
                )
    summary = admit_strict_training(records)
    assert summary["positive_conditions"] == 4
    assert summary["negative_conditions"] == 4
    assert summary["research_training_data_sufficient"] is True
    assert summary["optimizer_authorized"] is False
    for row in records:
        if row["context"] == 1:
            row["physical_clean_outcome"] = outcome()
            row["terminal_reward"] = 0.0
    summary = admit_strict_training(records)
    assert summary["positive_conditions"] == 2
    assert summary["research_training_data_sufficient"] is False
    records[1] = dict(records[0])
    with pytest.raises(ValueError, match="duplicate"):
        admit_strict_training(records)
