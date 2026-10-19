"""Tiny synthetic ONNX planner, not official SONIC/G1 skill evidence."""

from collections import deque
from dataclasses import replace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from rosclaw_soccer.providers.g1.sonic_navigation import (
    G1SonicNavigation,
    SonicNavigationConfig,
    _StreamingBackend,
)
from rosclaw_soccer.providers.g1.sonic_navigation_branch import fork_plain_navigation
from rosclaw_soccer.providers.g1.sonic_runup import (
    _VARIANTS,
    G1SonicQualification,
    G1SonicRunupConfig,
    G1SonicRunupController,
    _sonic_control_parameters,
)
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


def session():
    shapes = {
        "context_mujoco_qpos": (TensorProto.FLOAT, [1, 4, 36]),
        "target_vel": (TensorProto.FLOAT, [1]),
        "mode": (TensorProto.INT64, [1]),
        "movement_direction": (TensorProto.FLOAT, [1, 3]),
        "facing_direction": (TensorProto.FLOAT, [1, 3]),
        "random_seed": (TensorProto.INT64, [1]),
        "has_specific_target": (TensorProto.INT64, [1, 1]),
        "specific_target_positions": (TensorProto.FLOAT, [1, 4, 3]),
        "specific_target_headings": (TensorProto.FLOAT, [1, 4]),
        "allowed_pred_num_tokens": (TensorProto.INT64, [1, 11]),
        "height": (TensorProto.FLOAT, [1]),
    }
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["target_vel", "gain"], ["offset"]),
            helper.make_node("Add", ["context_mujoco_qpos", "offset"], ["segment"]),
            helper.make_node("Identity", ["four"], ["count"]),
        ],
        "synthetic-planner",
        [helper.make_tensor_value_info(k, *v) for k, v in shapes.items()],
        [
            helper.make_tensor_value_info("segment", TensorProto.FLOAT, [1, 4, 36]),
            helper.make_tensor_value_info("count", TensorProto.INT64, [1]),
        ],
        initializer=[
            numpy_helper.from_array(np.asarray([0.0005], dtype=np.float32), "gain"),
            numpy_helper.from_array(np.asarray([4], dtype=np.int64), "four"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )


def observation(nav, frame, speed=1.4):
    q = np.zeros(43)
    q[2], q[3], q[39] = 0.793, 1.0, 1.0
    q[0] = frame * 0.002
    q[7:36] = G1SonicRunupController.default_angles
    return TeamMotorObservation(
        agent_id=nav.agent_id,
        frame=frame,
        time_sec=frame * 0.02,
        intent="other",
        prospective_owner=False,
        qpos=tuple(float(x) for x in q),
        qvel=(0.0,) * 41,
        target_position_m=(0.0, 0.0, 0.0),
        navigation_command=(speed, 0.0, 0.0),
        navigation_envelope=nav.navigation_envelope,
    )


def advance(nav, frame, speed=1.4):
    obs = observation(nav, frame, speed)
    if frame == 0:
        nav.start_from_observation(obs)
    reference = nav.prepare_batched_proposal(obs)
    result = nav.commit_batched_action(np.full(29, 0.001 * frame, dtype=np.float32))
    return reference, result


def blank_navigation():
    cfg = SonicNavigationConfig(
        maximum_frames=300,
        planner_seed=30300,
        model_variant="low_latency",
        experimental_maximum_speed_mps=1.5,
        inference_threads=1,
        record_default_planner_events=True,
    )
    nav = object.__new__(G1SonicNavigation)
    nav.agent_id, nav.config, nav.contract_hash = "cpu.first_touch.0", cfg, "sha256:" + "a" * 64
    nav._next_frame, nav._origin_frame = 0, 0
    nav._faulted = nav._retired = nav._ready_from_handoff = nav._ready_from_observation = (
        nav._batch_pending
    ) = False
    nav._boundary_observation_hash = None
    b = object.__new__(_StreamingBackend)
    b.navigation, b.config, b._variant = (
        cfg,
        G1SonicRunupConfig(model_variant="low_latency", execution_duration_sec=4.5),
        _VARIANTS["low_latency"],
    )
    b.command = b.last_planned_command = (0.0, 0.0, 0.0)
    b.facing = 0.0
    b.planner_calls = b.last_planned_frame = 0
    b.events, b.latent_records = [], []
    b._planner = b._encoder = b._decoder = session()
    b._history = deque(maxlen=10)
    b.reference = np.empty((0, 36))
    b.reference_digest = ""
    b.action = np.zeros(29, dtype=np.float32)
    b.target = G1SonicRunupController.default_angles.copy()
    b._history_handoff_binding = None
    b._kp, b._kd, b._action_scale = _sonic_control_parameters(1.0, (1.0,) * 29)
    b.qualification = G1SonicQualification(
        eligible=True,
        planner_hash="synthetic",
        encoder_hash="synthetic",
        decoder_hash="synthetic",
        observation_config_hash="synthetic",
        planner_input_count=11,
        encoder_input_size=1247,
        decoder_input_size=994,
        decoder_output_size=29,
        model_variant="low_latency",
        reference_stride=1,
        heading_normalized=False,
        errors=(),
    )
    nav.backend = b
    return nav


@pytest.fixture
def navigation():
    nav = blank_navigation()
    for f in range(20):
        advance(nav, f)
    return nav


def test_fork_crosses_two_real_planner_refreshes_with_own_state(navigation):
    parent = navigation
    branch = fork_plain_navigation(parent, next_frame=20)
    altered = fork_plain_navigation(parent, next_frame=20)
    saved = parent.backend.reference.copy()
    for f in range(20, 45):
        a, at = advance(branch, f)
        different, _ = advance(altered, f, speed=0.7)
        if f == 20:
            assert np.array_equal(parent.backend.reference, saved)
            assert parent.backend.planner_calls == 1 and branch.backend.planner_calls == 2
            assert not np.array_equal(a, different)
        b, bt = advance(parent, f)
        assert a.tobytes() == b.tobytes() and at == bt
        assert branch.backend.events == parent.backend.events
    assert branch.backend.planner_calls == 3 and altered.backend.planner_calls == 3
    assert branch.backend._planner is parent.backend._planner
    assert not np.shares_memory(branch.backend.reference, parent.backend.reference)
    assert not np.shares_memory(branch.backend.action, parent.backend.action)
    assert not np.shares_memory(branch.backend._kp, parent.backend._kp)
    for old, new in zip(parent.backend._history, branch.backend._history, strict=True):
        for a, b in zip(old, new, strict=True):
            assert a.tobytes() == b.tobytes() and not np.shares_memory(a, b)


@pytest.mark.parametrize(
    "bad",
    [
        "extra",
        "backend_extra",
        "pending",
        "faulted",
        "clock",
        "calls",
        "reference",
        "nan",
        "history",
        "config",
        "session",
        "callback",
        "typed_clock",
        "tensor_shape",
        "nested_callback",
        "oversize",
    ],
)
def test_unreviewed_state_rejected_before_any_advance(navigation, bad):
    n = navigation
    if bad == "extra":
        n.new_state = []
    elif bad == "backend_extra":
        n.backend.new_state = []
    elif bad == "pending":
        n._batch_pending = True
    elif bad == "faulted":
        n._faulted = True
    elif bad == "clock":
        n._next_frame += 1
    elif bad == "typed_clock":
        n._next_frame = 20.0
    elif bad == "calls":
        n.backend.planner_calls += 1
    elif bad == "reference":
        n.backend.reference = np.zeros((411, 36))
    elif bad == "nan":
        n.backend.action[0] = np.nan
    elif bad == "history":
        n.backend._history.pop()
    elif bad == "config":
        n.config = replace(n.config, onnx_graph_encoder_layout=True)
    elif bad == "session":
        n.backend._planner = object()
    elif bad == "tensor_shape":
        n.backend._kp = np.ones(28)
    elif bad == "nested_callback":
        n.backend.events[0]["callback"] = lambda: None
    elif bad == "oversize":
        n.backend.events[0]["oversize"] = np.zeros((1024, 1024, 2))
    else:
        n.backend.events.append({"callback": lambda: None})
    calls = n.backend.planner_calls
    with pytest.raises(ValueError):
        fork_plain_navigation(n, next_frame=20)
    assert n.backend.planner_calls == calls
