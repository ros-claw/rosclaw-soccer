from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker
from rosclaw_soccer.rsi.sonic_encoder_reference_audit import (
    audit_encoder_reference_consumption,
    low_latency_legacy_features,
)
from rosclaw_soccer.rsi.sonic_locomotion_reference_audit import replay_locomotion_segments
from tests.rsi.test_sonic_locomotion_reference_audit import Planner, fixture


def tracker(reference):
    model = SimpleNamespace(
        device="cpu",
        variant="low_latency",
        qualification=SimpleNamespace(reference_stride=1, heading_normalized=False),
    )
    return BatchedSonicTracker(model, reference[None], low_latency_legacy_encoder_layout=True)


def captured_fixture(mode=2):
    trace, events = fixture(mode)
    recorded = np.empty((300, 1, 640))
    q, v = trace["canonical_qpos"], np.zeros((1, 41))

    def observe(frame, reference):
        producer = tracker(reference)
        for tick in range(frame, frame + 20):
            recorded[tick] = producer.encoder_features(tick, q[tick], v).numpy()

    replay_locomotion_segments(Planner(), trace, events, mode, reference_observer=observe)
    trace["foundation_neural_encoder_features"] = recorded
    return trace, events


def test_analytic_stationary_pose_preserves_legacy_orientation_offset():
    reference = np.zeros((10, 36))
    reference[:, 3] = 1
    result = low_latency_legacy_features(reference, reference[0, 3:7])
    np.testing.assert_array_equal(result[:580], 0)
    expected = np.tile([1, 0, 0, 1, 0, 0], 9)
    np.testing.assert_array_equal(result[580:634], expected)
    np.testing.assert_array_equal(result[634:], 0)


def test_numpy_matches_producer_nontrivial_body_and_reference_orientations():
    reference = np.zeros((50, 36))
    angle = np.linspace(-0.4, 0.6, 50)
    reference[:, 3], reference[:, 5] = np.cos(angle / 2), np.sin(angle / 2)
    reference[:, 7:] = np.sin(np.arange(50)[:, None] * 0.03 + np.arange(29)[None] * 0.05)
    q = np.zeros((1, 43))
    q[0, 3], q[0, 4] = np.cos(0.15), np.sin(0.15)
    actual = tracker(reference).encoder_features(7, q, np.zeros((1, 41))).numpy()[0]
    rebuilt = low_latency_legacy_features(reference[7:17], q[0, 3:7])
    np.testing.assert_allclose(rebuilt, actual, atol=1e-5, rtol=0)


@pytest.mark.parametrize("mode", [2, 3])
def test_all_frames_compare_independent_numpy_features(mode):
    trace, events = captured_fixture(mode)
    session = Planner()
    result = audit_encoder_reference_consumption(session, trace, events, mode)
    assert session.calls == result["actual_cpu_onnx_planner_calls"] == 15
    assert result["compared_encoder_frames"] == 300
    assert not result["physical_improvement_proven"]
    assert not result["teacher_qualified"]


@pytest.mark.parametrize("column", [0, 289, 290, 579, 580, 633, 639])
def test_detects_changed_positions_velocities_rotations_and_padding(column):
    trace, events = captured_fixture()
    trace["foundation_neural_encoder_features"][299, 0, column] += 0.001
    with pytest.raises(ValueError, match="planner-to-encoder mismatch"):
        audit_encoder_reference_consumption(Planner(), trace, events, 2)


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype", "nan"])
def test_invalid_recording_rejected_before_planner_call(fault):
    trace, events = captured_fixture()
    if fault == "missing":
        del trace["foundation_neural_encoder_features"]
    elif fault == "shape":
        trace["foundation_neural_encoder_features"] = np.zeros((299, 1, 640))
    elif fault == "dtype":
        trace["foundation_neural_encoder_features"] = trace[
            "foundation_neural_encoder_features"
        ].astype(np.float32)
    else:
        trace["foundation_neural_encoder_features"][0, 0, 0] = np.nan
    session = Planner()
    with pytest.raises(ValueError, match="recorded encoder"):
        audit_encoder_reference_consumption(session, trace, events, 2)
    assert session.calls == 0


def test_observer_cannot_corrupt_internal_reference_history():
    trace, events = fixture(2)
    expected = replay_locomotion_segments(Planner(), trace, events, 2)
    rows = replay_locomotion_segments(
        Planner(),
        trace,
        events,
        2,
        reference_observer=lambda frame, reference: reference.fill(0),
    )
    assert rows == expected
