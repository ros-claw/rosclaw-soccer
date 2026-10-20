from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.physics.sonic_navigation_forecast import (
    SonicNavigationForecaster,
    _clone_body,
    clone_navigation,
    navigation_state_hash,
)
from rosclaw_soccer.providers.g1.sonic_navigation import (
    G1SonicNavigation,
    SonicNavigationConfig,
    _StreamingBackend,
)
from rosclaw_soccer.providers.g1.sonic_runup import (
    _VARIANTS,
    G1SonicQualification,
    G1SonicRunupConfig,
    G1SonicRunupController,
)
from rosclaw_soccer.rsi.full_task_branch import integration_hash
from tests.physics.test_g1_target_episode import body


class FakeSession:
    """No inference: cloning/ownership test only, never G1 neural evidence."""

    def get_providers(self):
        return ["CPUExecutionProvider"]


def navigation(monkeypatch):
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    config = SonicNavigationConfig(
        model_variant="low_latency", inference_threads=1, record_default_planner_events=True
    )
    q = np.r_[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, G1SonicRunupController.default_angles]
    controller = object.__new__(_StreamingBackend)
    binding = "sha256:" + "1" * 64
    controller.__dict__.update(
        navigation=config,
        command=(0.3, 0.0, 0.0),
        facing=0.0,
        planner_calls=1,
        last_planned_frame=0,
        last_planned_command=(0.3, 0.0, 0.0),
        events=[dict(frame=0, command=(0.3, 0.0, 0.0), reference_hash=binding)],
        latent_records=[],
        config=G1SonicRunupConfig(model_variant="low_latency"),
        _variant=_VARIANTS["low_latency"],
        qualification=G1SonicQualification(
            eligible=True,
            planner_hash=binding,
            encoder_hash=binding,
            decoder_hash=binding,
            observation_config_hash=binding,
            planner_input_count=11,
            encoder_input_size=1247,
            decoder_input_size=994,
            decoder_output_size=29,
            model_variant="low_latency",
            reference_stride=1,
            heading_normalized=False,
            errors=(),
        ),
        _planner=FakeSession(),
        _encoder=FakeSession(),
        _decoder=FakeSession(),
        _history=deque(
            [tuple(np.zeros(n) for n in (3, 29, 29, 29, 3)) for _ in range(10)], maxlen=10
        ),
        reference=np.repeat(q[None], 410, axis=0),
        reference_digest=binding,
        action=np.zeros(29, dtype=np.float32),
        target=G1SonicRunupController.default_angles.copy(),
        _history_handoff_binding=None,
        _kp=np.ones(29),
        _kd=np.ones(29),
        _action_scale=np.ones(29),
    )
    nav = object.__new__(G1SonicNavigation)
    nav.__dict__.update(
        agent_id="tracker",
        config=config,
        backend=controller,
        contract_hash=binding,
        _next_frame=20,
        _faulted=False,
        _retired=False,
        _origin_frame=0,
        _ready_from_handoff=False,
        _ready_from_observation=False,
        _boundary_observation_hash=None,
        _batch_pending=False,
    )
    return nav


def test_only_sessions_shared_every_mutable_controller_buffer_private(monkeypatch):
    nav = navigation(monkeypatch)
    before = navigation_state_hash(nav)
    cloned = clone_navigation(nav)
    assert navigation_state_hash(cloned) == before
    assert cloned.backend._planner is nav.backend._planner
    assert cloned.backend._encoder is nav.backend._encoder
    assert cloned.backend._decoder is nav.backend._decoder
    assert cloned.backend.navigation is cloned.config
    cloned.backend.default_angles[0] += 0.1
    cloned.backend.reference[30, 7] += 0.2
    cloned.backend._history[0][0][0] += 0.3
    cloned.backend.events.append(dict(frame=20))
    cloned.backend._kp[0] = 2
    assert navigation_state_hash(nav) == before


@pytest.mark.parametrize("measured", [False, True])
def test_context_blend_metadata_and_event_graph_are_owned(monkeypatch, measured):
    nav = navigation(monkeypatch)
    configured = replace(
        nav.config,
        maximum_frames=300,
        experimental_reference_blend=True,
        experimental_measured_joint_context=measured,
        experimental_context_joint_limits=((-2.0, 2.0),) * 29 if measured else None,
    )
    nav.config = nav.backend.navigation = configured
    nav.backend.events[0]["planner_inputs"] = {"context": [[0.0] * 36 for _ in range(4)]}
    before = navigation_state_hash(nav)
    cloned = clone_navigation(nav)
    assert navigation_state_hash(cloned) == before
    cloned.backend.events[0]["planner_inputs"]["context"][0][7] = 0.1
    assert navigation_state_hash(nav) == before
    assert (
        cloned.config.experimental_context_joint_limits
        == configured.experimental_context_joint_limits
    )
    assert navigation_state_hash(cloned) != before
    # A forecast can itself be cloned without losing the private default array.
    further = clone_navigation(clone_navigation(nav))
    assert navigation_state_hash(further) == before


@pytest.mark.parametrize(
    "change", ["unknown", "pending", "faulted", "reference", "nonfinite", "foundation"]
)
def test_unqualified_controller_graph_rejected_before_copy(change, monkeypatch):
    nav = navigation(monkeypatch)
    if change == "unknown":
        nav.backend.unqualified_state = object()
    elif change == "pending":
        nav._batch_pending = True
    elif change == "faulted":
        nav._faulted = True
    elif change == "reference":
        nav.config = replace(
            nav.config, model_variant="sonic_v1_1", record_default_planner_events=False
        )
        nav.backend.navigation = nav.config
    elif change == "nonfinite":
        nav.backend.action[0] = np.nan
    elif change == "foundation":
        nav.backend.qualification = replace(
            nav.backend.qualification, eligible=False, errors=("toy",)
        )
    with pytest.raises(ValueError):
        clone_navigation(nav)


def test_actual_toy_body_clones_warm_state_and_keeps_prefix_accounting():
    original = body(record_all_ball_contacts=True)
    for frame in range(20):
        original.step(frame, np.zeros(29))
    before = integration_hash(original.model, original.data)
    branch = _clone_body(original)
    assert branch.model is original.model and branch.data is not original.data
    assert branch.next_frame == 20 and branch.integrated_steps == 200
    assert integration_hash(branch.model, branch.data) == before
    assert not np.shares_memory(branch.kp, original.kp)
    branch.step(20, np.zeros(29))
    assert branch.integrated_steps - 200 == 10
    assert original.integrated_steps == 200 and original.next_frame == 20
    assert integration_hash(original.model, original.data) == before


@pytest.mark.parametrize("budget", [0, 399, 401, 600010, True])
def test_explicit_bounded_native_budget_required(budget):
    with pytest.raises(ValueError):
        SonicNavigationForecaster(world_hash="sha256:" + "1" * 64, max_native_steps=budget)
