import pytest

from rosclaw_soccer.rsi.receiver_strict_terminal_reward import strict_terminal_reward


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
