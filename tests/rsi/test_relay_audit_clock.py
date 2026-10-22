import numpy as np
import pytest
from test_shared_native_motor_world import world

from rosclaw_soccer.rsi.relay_audit_clock import RelayAuditClock
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters


def test_independent_clock_matches_native_fields_without_calling_producer_advance(monkeypatch):
    reference = world()
    audit = RelayAuditClock(world())
    for step in range(5):
        targets = {
            "a/": np.linspace(-0.02, 0.02, 29) + step * 0.001,
            "b/": np.linspace(0.03, -0.03, 29) - step * 0.001,
        }
        if step == 2:
            parameters = NativePDParameters(np.full(29, 30.0), np.full(29, 2.0), np.full(29, 25.0))
            reference.switch_pd_profile("b/", parameters)
            switch = audit.switch_pd_profile("b/", parameters)
            for key, value in switch["before"].items():
                np.testing.assert_array_equal(value, switch["after"][key])
        expected = reference.advance(targets)
        actual = audit.advance(targets)
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            np.testing.assert_array_equal(expected[key], actual[key])
        assert expected["body_failed"] == actual["body_failed"]
        assert expected["time_before"] == actual["time_before"]
        assert reference.last_ball_contact_bodies == audit.last_ball_contact_bodies
        np.testing.assert_array_equal(
            reference.last_applied_target("b/"), audit.last_applied_target("b/")
        )
    assert audit.native_steps == 50
    # Verify the audit's next physical step has no producer integration call.
    monkeypatch.setattr(type(reference), "advance", lambda *a: pytest.fail("producer clock used"))
    audit.advance(targets)
    assert audit.native_steps == 60


def test_audit_refuses_live_restore_and_dirty_switch():
    initialized = world()
    initialized.advance({p: np.zeros(29) for p in initialized.layouts})
    with pytest.raises(ValueError, match="unused"):
        RelayAuditClock(initialized)
    audit = RelayAuditClock(world())
    audit.data.qpos[0] += 0.01
    with pytest.raises(ValueError, match="boundary"):
        audit.switch_pd_profile("b/", audit.parameters["b/"])
