"""Mock protocol checks, not native G1 numerical qualification evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rsi_shadow_prefix_qualification import ShadowPrefixQualification


def test_prefix_provider_rejected_before_body_access():
    from rosclaw_soccer.rsi.full_entry_forecast import nominal_prefix20

    with pytest.raises(ValueError, match="motor provider"):
        nominal_prefix20(None, None, None, motor_provider=17)


def test_shadow_history_requires_complete20_and_fresh_calls():
    p = ShadowPrefixQualification()
    with pytest.raises(ValueError, match="all20"):
        p.verify_closed()
    p.calls = 20
    p.codec = SimpleNamespace(_next_frame=20)
    p.nav = SimpleNamespace(_next_frame=37)
    p.tap = SimpleNamespace(encoder=SimpleNamespace(calls=20), decoder=SimpleNamespace(calls=20))
    assert p.verify_closed()["actual_consumed_history930_byte_exact"]
    p.tap.decoder.calls += 1
    with pytest.raises(ValueError, match="all20"):
        p.verify_closed()


def test_shadow_prefix_cannot_query_a_second_controller():
    p = ShadowPrefixQualification()
    p.calls = 1
    p.nav = object()
    with pytest.raises(ValueError, match="one fresh"):
        p(object(), SimpleNamespace(frame=1))


def test_zero_prefix_motor_provider_forwarded_for_all20(monkeypatch):
    from rosclaw_soccer.rsi import full_entry_forecast as m

    nav = SimpleNamespace(_next_frame=0)
    body = SimpleNamespace(next_frame=0, canonical=lambda: (None, None))
    command = (0.0, 0.0, 0.0)
    monkeypatch.setattr(
        m,
        "ball_entry_decision",
        lambda *a, **kw: SimpleNamespace(
            command=command, target_root_xy_m=(0.0, 0.0), horizontal_error_m=0.0
        ),
    )
    monkeypatch.setattr(m, "_observation", lambda n, b, f, c: f)
    provider = object()
    frames = []

    def tick(n, b, obs, received):
        assert received is callback
        frames.append(obs)
        return {"fixture": np.array([obs])}

    def callback(*args):
        return provider

    monkeypatch.setattr(m, "_tick", tick)
    result = m.nominal_prefix20(nav, body, None, motor_provider=callback)
    assert frames == list(range(20))
    np.testing.assert_array_equal(result["fixture"], np.arange(20))
