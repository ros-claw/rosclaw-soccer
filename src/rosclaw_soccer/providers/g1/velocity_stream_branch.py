"""Copy completed SIM velocity NN state, never reset phase or borrow future data.

Only immutable parameters and a stateless CPU ORT session may be shared.
This is a software primitive, not a locomotion or handoff effectiveness proof.
"""

import copy
import re
from dataclasses import asdict

import numpy as np

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.providers.g1.experimental_velocity_envelope import (
    ExperimentalVelocityEnvelopeStream,
)
from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.sim.contracts import hash_json

VelocityStream = ContinuousVelocityStream | ExperimentalVelocityEnvelopeStream
_FIELDS = frozenset(
    {
        "params",
        "session",
        "horizon_controls",
        "contract_hash",
        "phase",
        "last_action",
        "last_observation",
        "next_frame",
        "neural_calls",
        "attempted_neural_calls",
        "faulted",
    }
)


def velocity_stream_state_hash(stream: VelocityStream, *, completed_control: int) -> str:
    import onnxruntime as ort

    if (
        type(stream) not in (ContinuousVelocityStream, ExperimentalVelocityEnvelopeStream)
        or set(vars(stream)) != _FIELDS
        or stream.faulted is not False
        or type(completed_control) is not int
        or type(stream.horizon_controls) is not int
        or not 301
        <= stream.horizon_controls
        <= (6000 if type(stream) is ContinuousVelocityStream else 1200)
        or not 0 <= completed_control <= stream.horizon_controls
        or any(
            type(x) is not int or x != completed_control
            for x in (stream.next_frame, stream.neural_calls, stream.attempted_neural_calls)
        )
        or type(stream.phase) is not np.float32
        or not np.isfinite(stream.phase)
        or not 0 <= stream.phase < 1
        or type(stream.params) is not VelocityPolicyParameters
        or type(stream.contract_hash) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", stream.contract_hash) is None
        or type(stream.session) is not ort.InferenceSession
        or stream.session.get_providers() != ["CPUExecutionProvider"]
        or any(
            type(x) is not np.ndarray
            or x.shape != shape
            or x.dtype != np.float32
            or not np.isfinite(x).all()
            for x, shape in ((stream.last_action, (29,)), (stream.last_observation, (98,)))
        )
    ):
        raise ValueError("exact completed finite original CPU SIM velocity stream required")
    params = asdict(stream.params)
    VelocityPolicyParameters(**params)  # Revalidate the immutable data, no resource allocation.
    return str(
        hash_json(
            dict(
                stream_type=type(stream).__name__,
                params=params,
                contract_hash=stream.contract_hash,
                horizon_controls=stream.horizon_controls,
                completed_control=completed_control,
                phase=float(stream.phase),
                last_action=stream.last_action.tolist(),
                last_observation=stream.last_observation.tolist(),
                faulted=False,
            )
        )
    )


def fork_velocity_stream(stream: VelocityStream, *, completed_control: int) -> VelocityStream:
    before = velocity_stream_state_hash(stream, completed_control=completed_control)
    cloned = object.__new__(type(stream))
    for key, value in vars(stream).items():
        setattr(cloned, key, value if key in ("session", "params") else copy.deepcopy(value))
    if (
        velocity_stream_state_hash(cloned, completed_control=completed_control) != before
        or velocity_stream_state_hash(stream, completed_control=completed_control) != before
        or np.shares_memory(stream.last_action, cloned.last_action)
        or np.shares_memory(stream.last_observation, cloned.last_observation)
    ):
        raise ValueError("velocity branch must retain exact clock and private action history")
    return cloned
