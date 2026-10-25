"""SIM skill continuation with actual-target history, not stale shadow actions."""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.providers.g1.experimental_velocity_envelope import (
    ExperimentalVelocityEnvelopeStream,
)
from rosclaw_soccer.providers.g1.velocity_policy import target29


class ObservedVelocityContinuation:
    """One shadow inference and one completed actual-target observation per tick.

    Never modifies a physical target to improve codec agreement. A non-shadow
    target is represented in float32 to at most 5e-7 rad decoding error, without
    clipping. Original velocity execution preserves its exact original action.
    This representation contract is not a physical handoff qualification.
    """

    def __init__(
        self, stream: ContinuousVelocityStream | ExperimentalVelocityEnvelopeStream
    ) -> None:
        if type(stream) not in (ContinuousVelocityStream, ExperimentalVelocityEnvelopeStream):
            raise ValueError("explicit owned SIM velocity stream required")
        if stream.next_frame != 0 or stream.neural_calls != 0 or stream.faulted:
            raise ValueError("unused stream; no phase/history reinitialization permitted")
        self.stream = stream
        self.pending: np.ndarray | None = None
        self.pending_frame: int | None = None
        self.failed = False

    def propose(self, *args: Any, frame: int) -> np.ndarray:
        if self.failed or self.pending is not None:
            raise ValueError("continuation fault or uncommitted prior observation")
        self.failed = True
        target = self.stream.propose(*args, frame=frame)
        self.pending = target.copy()
        self.pending_frame = frame
        self.failed = False
        return target.copy()

    def observe_executed(self, target: np.ndarray, *, completed_control: int) -> dict[str, Any]:
        if self.failed or self.pending is None or self.pending_frame is None:
            raise ValueError("one admitted shadow proposal before actual observation required")
        self.failed = True
        if (
            type(completed_control) is not int
            or completed_control != self.pending_frame + 1
            or self.stream.next_frame != completed_control
            or type(target) is not np.ndarray
            or target.shape != (29,)
            or target.dtype != np.float64
            or not np.isfinite(target).all()
            or np.max(abs(target)) > 10
        ):
            raise ValueError("next completed control and actual finite HG29 target required")
        original = np.array_equal(target, self.pending)
        action = self.stream.last_action.copy()
        if not original:
            params = self.stream.params
            action = (
                (target - np.asarray(params.action_offset, np.float64))
                / np.asarray(params.action_scale, np.float64)
            ).astype(np.float32)
        decoded = target29(action, self.stream.params)
        error = float(np.max(abs(decoded - target)))
        if error > 5e-7:
            raise ValueError("actual target not representable within declared float32 error")
        # Only the stream's private history changes, never a physical target.
        self.stream.last_action = action.copy()
        receipt = dict(
            schema="soccer.observed_velocity_continuation.v1",
            completed_control=completed_control,
            original_shadow_action_retained=original,
            maximum_target_decode_error_rad=error,
            maximum_encoded_action_magnitude=float(np.max(abs(action))),
            phase=float(self.stream.phase),
            actual_target_history=action.tolist(),
            physical_target_modified=False,
            phase_reset=False,
            hardware_authorized=False,
        )
        self.pending = self.pending_frame = None
        self.failed = False
        return receipt
