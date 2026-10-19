"""Synthetic planner arithmetic, NOT original G1 asset qualification."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.sonic_locomotion_reference_audit import replay_locomotion_segments
from rosclaw_soccer.rsi.unlogged_encoder_reference_audit import audit_unlogged_encoder_reference
from tests.rsi.test_sonic_encoder_reference_audit import captured_fixture
from tests.rsi.test_sonic_locomotion_reference_audit import Planner


def test_all300_features_rebuilt_without_fabricating_original_events():
    trace, _ = captured_fixture()
    session = Planner()
    r = audit_unlogged_encoder_reference(session, trace)
    assert session.calls == r["actual_planner_calls"] == 15
    assert r["encoder_frames_compared"] == 300
    assert not r["original_planner_event_logs_authenticated"]
    assert not r["producer_tracker_called"] and not r["teacher_qualified"]


@pytest.mark.parametrize("column", [0, 289, 290, 579, 580, 633, 639])
def test_reference_geometry_velocity_orientation_padding_all_compared(column):
    trace, _ = captured_fixture()
    trace["foundation_neural_encoder_features"][299, 0, column] += 0.001
    with pytest.raises(ValueError, match="mismatch"):
        audit_unlogged_encoder_reference(Planner(), trace)


@pytest.mark.parametrize("bad", ["shape", "dtype", "nonfinite", "command", "quaternion"])
def test_malformed_trace_rejected_before_any_planner_call(bad):
    trace, _ = captured_fixture()
    if bad == "shape":
        trace["foundation_neural_encoder_features"] = trace["foundation_neural_encoder_features"][
            :-1
        ]
    elif bad == "dtype":
        trace["foundation_neural_encoder_features"] = trace[
            "foundation_neural_encoder_features"
        ].astype(np.float64)
    elif bad == "nonfinite":
        trace["foundation_neural_encoder_features"][10, 0, 0] = np.nan
    elif bad == "command":
        trace["navigation_command"][0, 0, 0] = 2
    else:
        trace["canonical_qpos"][0, 0, 3] = 0
    p = Planner()
    with pytest.raises(ValueError):
        audit_unlogged_encoder_reference(p, trace)
    assert p.calls == 0


def test_legacy_event_hash_requirement_not_weakened():
    trace, events = captured_fixture()
    events[0]["reference_hash"] = "not-the-segment"
    with pytest.raises(ValueError, match="reference differs"):
        replay_locomotion_segments(Planner(), trace, events, 2)
    with pytest.raises(ValueError, match="fifteen"):
        replay_locomotion_segments(Planner(), trace, [], 2)
