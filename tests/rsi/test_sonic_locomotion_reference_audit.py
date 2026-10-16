import copy

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_navigation import SonicNavigationConfig, _StreamingBackend
from rosclaw_soccer.rsi.sonic_locomotion_reference_audit import replay_locomotion_segments


class Planner:
    """Test fixture, never a physical planner qualification."""

    def __init__(self):
        self.calls = 0

    def run(self, names, feed):
        self.calls += 1
        n = 12 + 4 * (int(feed["random_seed"][0]) % 3)
        initial = np.mean(feed["context_mujoco_qpos"], axis=1, keepdims=True)
        output = np.repeat(initial, n, axis=1).astype(np.float32)
        output[0, :, 0] += np.linspace(0, float(feed["target_vel"][0]) / 3, n, dtype=np.float32)
        output[0, :, 7] += np.linspace(0, float(feed["mode"][0]) / 5, n, dtype=np.float32)
        return output, np.asarray([n], dtype=np.int64)


def fixture(mode):
    q = np.zeros((300, 1, 43))
    q[:, 0, 2] = 0.793
    q[:, 0, 3] = 1.0
    command = np.zeros((300, 1, 3))
    command[:, 0, 0] = 1.4
    command[60, 0, 0] = 0.8
    command[80, 0, 0] = 0.0
    command[100, 0, 0] = 0.02
    backend = object.__new__(_StreamingBackend)
    backend.navigation = SonicNavigationConfig(
        maximum_frames=300,
        planner_seed=30300,
        model_variant="low_latency",
        experimental_maximum_speed_mps=1.5,
        experimental_locomotion_mode=mode,
    )
    backend._planner = Planner()
    backend.planner_calls = 0
    backend.events = []
    backend.facing = 0.0
    backend.command = tuple(command[0, 0])
    backend.reference = backend._generate_reference(q[0, 0, :36])
    for frame in range(20, 300, 20):
        backend.command = tuple(command[frame, 0])
        backend.refresh_reference(None, frame)
    return dict(canonical_qpos=q, navigation_command=command), backend.events


@pytest.mark.parametrize("mode", [2, 3])
def test_reconstructs_all_fifteen_segments_without_navigation_backend(mode):
    trace, events = fixture(mode)
    session = Planner()
    rows = replay_locomotion_segments(session, trace, events, mode)
    assert session.calls == len(rows) == 15
    assert [r["frame"] for r in rows] == list(range(0, 300, 20))
    assert [r["reference_hash"] for r in rows] == [e["reference_hash"] for e in events]
    assert [r["planner_mode"] for r in rows][3:6] == [1, 0, 1]


@pytest.mark.parametrize("fault", ["short", "mode", "time", "command", "reference", "initial_pose"])
def test_replay_rejects_missing_or_changed_provenance(fault):
    trace, events = fixture(3)
    if fault == "short":
        events.pop()
    elif fault == "mode":
        events[0]["planner_mode"] = 2
    elif fault == "time":
        events[1]["frame"] = 21
    elif fault == "command":
        events[1]["command"] = (1.3, 0.0, 0.0)
    elif fault == "reference":
        events[1]["reference_hash"] = "sha256:" + "0" * 64
    else:
        trace["canonical_qpos"][0, 0, 7] = 0.1
    with pytest.raises(ValueError):
        replay_locomotion_segments(Planner(), trace, events, 3)


@pytest.mark.parametrize("fault", ["nonfinite", "nonunit", "yaw", "speed", "bool_mode"])
def test_out_of_contract_inputs_rejected_before_planner_call(fault):
    trace, events = fixture(3)
    mode = 3
    if fault == "nonfinite":
        trace["canonical_qpos"][2, 0, 0] = np.nan
    elif fault == "nonunit":
        trace["canonical_qpos"][2, 0, 3] = 0.0
    elif fault == "yaw":
        trace["navigation_command"][2, 0, 2] = 0.1
    elif fault == "speed":
        trace["navigation_command"][2, 0, 0] = 1.6
    else:
        mode = True
    session = Planner()
    with pytest.raises(ValueError):
        replay_locomotion_segments(session, trace, events, mode)
    assert session.calls == 0


def test_audit_does_not_mutate_observations_or_recorded_events():
    trace, events = fixture(3)
    before, recorded = copy.deepcopy(trace), copy.deepcopy(events)
    replay_locomotion_segments(Planner(), trace, events, 3)
    for key in trace:
        np.testing.assert_array_equal(trace[key], before[key])
    assert events == recorded


@pytest.mark.parametrize("mode", [2, 3])
def test_public_reference_audit_binds_exact_provider_contract(monkeypatch, tmp_path, mode):
    from types import SimpleNamespace

    from rosclaw_soccer.providers.g1 import sonic_navigation as nav
    from rosclaw_soccer.rsi.sonic_locomotion_reference_audit import audit_locomotion_reference
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

    trace, events = fixture(mode)
    foundation = hash_json("unit-test-foundation-not-a-physical-model")

    class Backend:
        def __init__(self, *args):
            self.qualification = SimpleNamespace(qualification_hash=foundation)

    monkeypatch.setattr(nav, "_StreamingBackend", Backend)
    motor = nav.G1SonicNavigation(
        None,
        "cpu.first_touch.0",
        SonicNavigationConfig(
            maximum_frames=300,
            planner_seed=30300,
            model_variant="low_latency",
            experimental_maximum_speed_mps=1.5,
            experimental_locomotion_mode=mode,
            inference_threads=1,
        ),
    )
    path = tmp_path / "unit-fixture.onnx"
    path.write_bytes(b"unit-test-placeholder; not an ONNX asset")
    contract = dict(
        schema="soccer.rsi.sonic_locomotion_mode_probe.v1",
        navigation_contract_hash=motor.contract_hash,
        planner_hash=hash_bytes(path.read_bytes()),
        planner_seed=30300,
        maximum_navigation_speed_m_s=1.5,
        high_speed_threshold_m_s=0.8,
        high_speed_planner_mode=mode,
        low_speed_modes_unchanged=True,
        physical_joint_and_torque_limits_unchanged=True,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report = dict(
        locomotion_mode_probe=contract,
        locomotion_mode_planner_events=events,
        foundation_hash=foundation,
        report_hash=hash_json("fixture-report"),
        physical_trace_hash=hash_json("fixture-trace"),
    )
    import sys

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=lambda *a, **k: Planner()),
    )
    result = audit_locomotion_reference(report, trace, path)
    assert result["actual_cpu_onnx_planner_calls"] == 15  # Mocked unit calls only.
    assert result["teacher_qualified"] is False
    assert result["foundation_encoder_reference_consumption_independently_checked"] is False
    contract["navigation_contract_hash"] = hash_json("wrong-foundation-binding")
    with pytest.raises(ValueError, match="binding"):
        audit_locomotion_reference(report, trace, path)
