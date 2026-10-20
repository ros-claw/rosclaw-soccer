"""Provider contract/scheduling fixtures, not G1 physical or ONNX qualification."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1 import sonic_navigation as navigation
from rosclaw_soccer.providers.g1.sonic_navigation import SonicNavigationConfig, _StreamingBackend
from rosclaw_soccer.sim.contracts import hash_json

LIMITS = ((-2.0, 2.0),) * 29


def config(**changes):
    return SonicNavigationConfig(
        **(
            dict(
                maximum_frames=300,
                model_variant="low_latency",
                record_default_planner_events=True,
                inference_threads=1,
            )
            | changes
        )
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(experimental_reference_blend=1),
        dict(experimental_measured_joint_context=1),
        dict(experimental_measured_joint_context=True, experimental_context_joint_limits=LIMITS),
        dict(experimental_context_joint_limits=LIMITS),
        dict(experimental_reference_blend=True, maximum_frames=600),
        dict(experimental_reference_blend=True, record_default_planner_events=False),
        dict(experimental_reference_blend=True, model_variant="sonic_v1_1"),
        dict(experimental_reference_blend=True, onnx_graph_encoder_layout=True),
        dict(experimental_reference_blend=True, experimental_command_replanning=True),
        dict(experimental_reference_blend=True, experimental_maximum_speed_mps=1.5),
        dict(experimental_reference_blend=True, pose_reference=object()),
        dict(experimental_reference_blend=True, latent_schedule=object()),
    ],
)
def test_unqualified_or_mixed_profile_refused(changes):
    with pytest.raises(ValueError):
        config(**changes)


@pytest.mark.parametrize(
    "limits",
    [
        None,
        list(LIMITS),
        LIMITS[:28],
        ((0.0, 0.0),) * 29,
        ((False, 2.0),) * 29,
        ((0.0, float("inf")),) * 29,
        tuple([-2.0, 2.0] for _ in range(29)),
    ],
)
def test_measurement_requires_immutable_valid_native_metadata(limits):
    with pytest.raises(ValueError):
        config(
            experimental_reference_blend=True,
            experimental_measured_joint_context=True,
            experimental_context_joint_limits=limits,
        )


@pytest.mark.parametrize("measured", [False, True])
def test_only_future_reference_changes_and_trace_is_owned_and_serializable(measured):
    obj = object.__new__(_StreamingBackend)
    obj.navigation = config(
        experimental_reference_blend=True,
        experimental_measured_joint_context=measured,
        experimental_context_joint_limits=LIMITS if measured else None,
    )
    obj.command = obj.last_planned_command = (0.3, 0.0, 0.0)
    obj.last_planned_frame = 0
    obj.reference = np.zeros((410, 36))
    obj.reference[:, 3] = 1
    old = obj.reference.copy()
    obj._history = ["never_reset"]
    obj.events = []
    contexts = []

    def plan(context, frame):
        contexts.append(context.copy())
        obj.events.append(dict(frame=frame))
        result = np.repeat(context[:1], 40, axis=0)
        result[:, 7:] += 0.2
        return result

    obj.plan = plan
    q = np.zeros(43)
    q[3] = 1
    q[7:36] = 0.2
    saved = q.copy()
    obj.refresh_reference(SimpleNamespace(qpos=q), 20)
    np.testing.assert_array_equal(obj.reference[:30], old[:30])
    np.testing.assert_array_equal(obj.reference[30], old[30])
    np.testing.assert_allclose(obj.reference[39:, 7:], 0.32 if measured else 0.2, atol=1e-16)
    np.testing.assert_array_equal(q, saved)
    assert obj._history == ["never_reset"]
    assert len(contexts) == 1
    np.testing.assert_allclose(contexts[0][:, 7:], 0.12 if measured else 0.0)
    trace = obj.events[0]["reference_stitch"]
    q[7] = 1.5
    assert trace["measured_qpos"][7] == 0.2
    assert json.loads(json.dumps(trace))["start_frame"] == 30


def test_disabled_config_matches_independent_legacy_hash(monkeypatch):
    class FixtureBackend:
        def __init__(self, *args):
            self.qualification = SimpleNamespace(qualification_hash="sha256:" + "a" * 64)

    monkeypatch.setattr(navigation, "_StreamingBackend", FixtureBackend)
    motor = navigation.G1SonicNavigation(None, "tracker", config())
    expected = hash_json(
        dict(
            schema="rosclaw_soccer.g1_sonic_navigation.v1",
            agent_id="tracker",
            config=dict(
                maximum_frames=300,
                planner_seed=30300,
                replan_frames=20,
                lookahead_frames=10,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
            foundation="sha256:" + "a" * 64,
            command="post-clearance world vx vy <= .7 m/s, yaw rate <= 1.5 rad/s",
            activation_ceiling="SIM_ONLY",
        )
    )
    assert motor.contract_hash == expected
    blend = navigation.G1SonicNavigation(None, "tracker", config(experimental_reference_blend=True))
    measured = navigation.G1SonicNavigation(
        None,
        "tracker",
        config(
            experimental_reference_blend=True,
            experimental_measured_joint_context=True,
            experimental_context_joint_limits=LIMITS,
        ),
    )
    assert len({motor.contract_hash, blend.contract_hash, measured.contract_hash}) == 3
