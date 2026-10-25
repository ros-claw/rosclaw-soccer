"""Synthetic real-ORT tests; no G1 physics or locomotion success claims."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from test_g1_velocity_policy import parameters, state

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.providers.g1.experimental_velocity_envelope import (
    ExperimentalVelocityEnvelopeStream,
)
from rosclaw_soccer.providers.g1.velocity_policy import phase_delta
from rosclaw_soccer.providers.g1.velocity_stream_branch import (
    fork_velocity_stream,
    velocity_stream_state_hash,
)


def synthetic_stream(kind=ContinuousVelocityStream):
    helper = onnx.helper
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["obs", "indices"], ["previous"], axis=1),
            helper.make_node("Add", ["previous", "bias"], ["actions"]),
        ],
        "synthetic_action_history",
        [helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, [1, 98])],
        [helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, [1, 29])],
        [
            onnx.numpy_helper.from_array(np.arange(69, 98, dtype=np.int64), "indices"),
            onnx.numpy_helper.from_array(np.full((1, 29), 0.001, np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    result = object.__new__(kind)
    result.params = parameters()
    result.session = ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    result.contract_hash = "sha256:" + "a" * 64
    result.horizon_controls = 360
    result.phase = phase_delta()
    result.last_action = np.zeros(29, np.float32)
    result.last_observation = np.zeros(98, np.float32)
    result.next_frame = result.neural_calls = result.attempted_neural_calls = 0
    result.faulted = False
    return result


@pytest.mark.parametrize("kind", [ContinuousVelocityStream, ExperimentalVelocityEnvelopeStream])
def test_copied_cpu_nn_has_exact_consecutive_future_and_private_histories(kind):
    source = synthetic_stream(kind)
    q, v = state()
    command = np.array([0.8, 0.2, 0.1], np.float32)
    for frame in range(57):
        source.propose(q, v, np.zeros(3), command, frame=frame)
    before = velocity_stream_state_hash(source, completed_control=57)
    ghost = fork_velocity_stream(source, completed_control=57)
    assert ghost.session is source.session and ghost.params is source.params
    for frame in range(57, 89):
        target = ghost.propose(q, v, np.zeros(3), command, frame=frame)
        assert velocity_stream_state_hash(source, completed_control=frame) == before
        actual = source.propose(q, v, np.zeros(3), command, frame=frame)
        np.testing.assert_array_equal(target, actual)
        before = velocity_stream_state_hash(source, completed_control=frame + 1)
        assert velocity_stream_state_hash(ghost, completed_control=frame + 1) == before
    ghost.last_action[:] = 10
    assert not np.array_equal(source.last_action, ghost.last_action)


@pytest.mark.parametrize(
    "mutation",
    ["resource", "fault", "clock", "attempt", "phase", "nan", "dtype", "params", "session"],
)
def test_unknown_resource_fault_or_incomplete_boundary_is_not_forkable(mutation):
    stream = synthetic_stream()
    if mutation == "resource":
        stream.driver = object()
    elif mutation == "fault":
        stream.faulted = True
    elif mutation == "clock":
        stream.next_frame = 1
    elif mutation == "attempt":
        stream.attempted_neural_calls = 1
    elif mutation == "phase":
        stream.phase = np.float32(np.nan)
    elif mutation == "nan":
        stream.last_action[0] = np.inf
    elif mutation == "dtype":
        stream.last_observation = stream.last_observation.astype(np.float64)
    elif mutation == "params":
        stream.params = object()
    elif mutation == "session":
        stream.session = object()
    with pytest.raises(ValueError):
        fork_velocity_stream(stream, completed_control=0)


@pytest.mark.parametrize("boundary", [True, -1, 1.0, 361])
def test_no_fabricated_completed_clock(boundary):
    with pytest.raises(ValueError):
        fork_velocity_stream(synthetic_stream(), completed_control=boundary)
