"""Measured-state SONIC muscle decoder for a learned64 motion-intention policy.

SIM research only. No simulator, robot, transport, reference planner or actuator
handle enters this adapter. A future-reference encoder is available EXPLICITLY
for teacher-label extraction; pure student execution uses decode_motion only.
"""

import collections
from types import SimpleNamespace
from typing import Any

import numpy as np


class FoundationMotorCodec:
    def __init__(self, qualified_backend: Any) -> None:
        qualification = qualified_backend.qualification
        if qualification.model_variant != "low_latency" or qualification.eligible is not True:
            raise ValueError("qualified frozen low-latency muscle network required")
        if getattr(qualified_backend.config, "onnx_graph_encoder_layout", False):
            raise ValueError("original584 encoder layout required")
        self._backend = qualified_backend
        self._history: collections.deque[tuple[np.ndarray, ...]] = collections.deque(maxlen=10)
        self._next_frame = 0
        self._observed = False
        self._proposal: np.ndarray | None = None
        self._previous = np.zeros(29, dtype=np.float32)
        self._q: np.ndarray | None = None
        self._faulted = False

    def observe(self, q: np.ndarray, v: np.ndarray, previous: np.ndarray, *, frame: int) -> None:
        try:
            if (
                self._faulted
                or self._observed
                or type(frame) is not int
                or frame != self._next_frame
                or not 0 <= frame < 300
            ):
                raise ValueError("consecutive uncommitted-free H300 measured observation required")
            for a, shape in ((q, (43,)), (v, (41,)), (previous, (29,))):
                if (
                    type(a) is not np.ndarray
                    or a.shape != shape
                    or a.dtype.kind != "f"
                    or not np.isfinite(a).all()
                ):
                    raise ValueError("finite own measured canonical state required")
            if (
                previous.dtype != np.float32
                or previous.tobytes() != self._previous.tobytes()
                or abs(np.linalg.norm(q[3:7]) - 1) > 1e-4
            ):
                raise ValueError(
                    "actual committed previous action and unit root quaternion required"
                )
            entry = self._backend._history_entry(SimpleNamespace(qpos=q, qvel=v), previous)
            if frame == 0:
                self._history.extend(tuple(a.copy() for a in entry) for _ in range(10))
            else:
                self._history.append(tuple(a.copy() for a in entry))
            self._q = q.copy()
            self._observed = True
        except Exception:
            self._faulted = True
            raise

    def encode_teacher_reference(self, future_poses: np.ndarray) -> np.ndarray:
        """Privileged label-only encoder; never an input to a learned student."""
        try:
            if (
                self._faulted
                or not self._observed
                or self._proposal is not None
                or future_poses.shape != (10, 36)
                or not np.isfinite(future_poses).all()
            ):
                raise ValueError("pending own state and explicit finite teacher window required")
            self._backend.reference = future_poses.copy()
            encoder_input = self._backend._encoder_observation(SimpleNamespace(qpos=self._q), 0)
            token: np.ndarray = self._backend._encoder.run(
                None, {self._backend._encoder.get_inputs()[0].name: encoder_input}
            )[0]
            if token.shape != (1, 64) or token.dtype != np.float32 or not np.isfinite(token).all():
                raise ValueError("finite64 teacher motion token required")
            return np.array(token[0], copy=True)
        except Exception:
            self._faulted = True
            raise

    def decode_motion(self, token: np.ndarray) -> np.ndarray:
        """Decode a student's token with ONLY its own ten measured body states."""
        try:
            if (
                self._faulted
                or not self._observed
                or self._proposal is not None
                or token.shape != (64,)
                or token.dtype != np.float32
                or not np.isfinite(token).all()
            ):
                raise ValueError("one finite64 motion proposal per measured state required")
            decoder = np.zeros((1, 994), dtype=np.float32)
            decoder[0, :64] = token
            offsets = (64, 94, 384, 674, 964, 994)
            history = list(self._history)
            for field, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
                decoder[0, start:end] = np.asarray(
                    [entry[field] for entry in history], dtype=np.float32
                ).reshape(-1)
            action: np.ndarray = self._backend._decoder.run(
                None, {self._backend._decoder.get_inputs()[0].name: decoder}
            )[0][0]
            if action.shape != (29,) or action.dtype != np.float32 or not np.isfinite(action).all():
                raise ValueError("finite29 muscle action required")
            self._proposal = action.copy()
            return action.copy()
        except Exception:
            self._faulted = True
            raise

    def commit_consumed(self, action: np.ndarray, *, foundation_prefix: bool = False) -> None:
        """Record actual execution, never silently replace a decoded proposal."""
        try:
            if (
                self._faulted
                or not self._observed
                or action.shape != (29,)
                or action.dtype != np.float32
                or not np.isfinite(action).all()
            ):
                raise ValueError("pending own observation and actual finite29 action required")
            if type(foundation_prefix) is not bool or (
                foundation_prefix and (self._next_frame >= 20 or self._proposal is not None)
            ):
                raise ValueError(
                    "external Foundation consumption is restricted to original prefix20"
                )
            if not foundation_prefix and (
                self._proposal is None or self._proposal.tobytes() != action.tobytes()
            ):
                raise ValueError("actual action differs from own muscle decoder output")
            self._previous = action.copy()
            self._proposal = None
            self._observed = False
            self._next_frame += 1
        except Exception:
            self._faulted = True
            raise
