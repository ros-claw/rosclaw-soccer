from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.native_foot_reward import FootStepEvidence, NearFootReward


def step(frame, **kw):
    base = FootStepEvidence(frame, 0.2, 0.2, 0.3, 0.31, 0.0, 0.0, 0.15, 0.15, 0.0, 0.0, 0.79, 1.0)
    return replace(base, **kw)


def test_ball_initial_rolling_does_not_earn_football_progress_or_success():
    r = NearFootReward(initial_ball_y_m=0)
    total = 0
    for frame in range(20, 300):
        value, terms = r.step(step(frame, ball_x_after_m=10.0))
        assert terms["forward"] == terms["momentum"] == terms["contact"] == 0
        total += value
    assert total == pytest.approx(-0.28)
    assert r.terminal(dict(qualified=False), causal_first_touch_possible=True) == 0


def test_foot_impulse_unlocks_progress_but_nonfoot_latches_it_off():
    r = NearFootReward(initial_ball_y_m=0)
    _, terms = r.step(step(20, foot_force_n=2.0, ball_vx_after_mps=0.7))
    assert terms["contact"] > 0 and terms["momentum"] > 0 and terms["forward"] > 0
    _, terms = r.step(step(21, nonfoot_force_n=2.0, ball_x_after_m=5.0))
    assert terms["nonfoot"] == -5 and terms["forward"] == 0
    _, terms = r.step(step(22, foot_force_n=2.0, ball_x_after_m=5.0))
    assert terms["forward"] == terms["contact"] == 0


def test_bad_evidence_or_early_terminal_rejected():
    r = NearFootReward(initial_ball_y_m=0)
    with pytest.raises(ValueError):
        r.step(step(20, foot_force_n=float("nan")))
    assert r.next_frame == 20
    with pytest.raises(ValueError):
        r.terminal(dict(qualified=True), causal_first_touch_possible=True)
