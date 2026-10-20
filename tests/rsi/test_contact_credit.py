import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_credit import contact_return_credit


def fixture():
    terms = [
        dict(
            time=-0.001,
            approach=0.0,
            contact=0.0,
            momentum=0.0,
            forward=0.0,
            lateral=0.0,
            nonfoot=0.0,
            balance=0.0,
        )
        for _ in range(280)
    ]
    return terms


def test_moves_measured_football_return_not_safety_or_time_to_contact():
    terms = fixture()
    terms[10]["contact"] = 0.5
    terms[20]["forward"] = 1.0
    terms[21]["lateral"] = -0.2
    terms[50]["nonfoot"] = -5.0
    rewards = np.asarray([0.0] * 20 + [sum(t.values()) for t in terms], dtype=np.float32)
    rewards[-1] -= 3
    result = contact_return_credit(
        rewards, terms, first_contact_frame=30, qualified=False, causal=True
    )
    assert result[30] == pytest.approx(1.299)
    assert result[70] == rewards[70]  # nonfoot safety charge stays at actual time
    assert result[-1] == rewards[-1]  # unsafe terminal charge stays at terminal
    assert float(result.sum()) == pytest.approx(float(rewards.sum()), abs=2e-5)


def test_terminal_success_is_redistributed_without_inventing_total_reward():
    terms = fixture()
    terms[10]["contact"] = 0.5
    rewards = np.asarray([0.0] * 20 + [sum(t.values()) for t in terms], dtype=np.float32)
    rewards[-1] += 5
    result = contact_return_credit(
        rewards, terms, first_contact_frame=30, qualified=True, causal=True
    )
    assert result[30] == pytest.approx(5.499)
    assert result[-1] == pytest.approx(-0.001, abs=1e-6)


def test_no_contact_preserves_data_and_fabricated_contact_credit_rejected():
    terms = fixture()
    rewards = np.asarray([0.0] * 20 + [sum(t.values()) for t in terms], dtype=np.float32)
    result = contact_return_credit(
        rewards, terms, first_contact_frame=None, qualified=False, causal=True
    )
    np.testing.assert_array_equal(result, rewards)
    terms[0]["forward"] = 1
    with pytest.raises(ValueError):
        contact_return_credit(
            rewards, terms, first_contact_frame=None, qualified=False, causal=True
        )


def test_precontact_football_reward_rejected():
    terms = fixture()
    terms[0]["contact"] = 0.5
    rewards = np.zeros(300, dtype=np.float32)
    with pytest.raises(ValueError):
        contact_return_credit(rewards, terms, first_contact_frame=30, qualified=False, causal=True)
