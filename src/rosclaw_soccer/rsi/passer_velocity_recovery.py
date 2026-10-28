"""SIM-only passer velocity shadow with actual-action handoff history.

This adapter does not step physics, change gains, select a role, or authorize
hardware. Its candidate handoff must be evaluated in a live shared world.
"""

import numpy as np

from rosclaw_soccer.rsi.paid_velocity_shadow import policy_action_from_hg_target


class PasserVelocityRecovery:
    def __init__(self, world, prefix: str, stream) -> None:
        if prefix not in world.layouts or world.control_steps != 0 or world.failed:
            raise ValueError("one fresh healthy shared player and its own stream required")
        if stream.next_frame != 0 or stream.neural_calls != 0:
            raise ValueError("fresh private continuous velocity history required")
        self.world, self.prefix, self.stream = world, prefix, stream
        self.boundary = world.control_steps
        self.active = False
        self.pending = None

    def activate(self, *, motion_complete: bool, passer_foot_seen: bool) -> None:
        if (
            type(motion_complete) is not bool
            or type(passer_foot_seen) is not bool
            or not motion_complete
            or not passer_foot_seen
            or self.active
            or self.pending is not None
            or self.world.failed
            or self.world.control_steps != self.boundary
        ):
            raise ValueError("healthy own completed motion and measured pass boundary required")
        self.active = True

    def observe(self, command_body: np.ndarray) -> np.ndarray:
        if (
            self.pending is not None
            or self.world.failed
            or self.world.control_steps != self.boundary
            or not isinstance(command_body, np.ndarray)
            or command_body.shape != (3,)
            or command_body.dtype != np.float32
            or not np.isfinite(command_body).all()
        ):
            raise ValueError("one finite own command at actual shared boundary required")
        layout = self.world.layouts[self.prefix]
        q, v = layout.canonical(self.world.data)
        target = self.stream.propose(
            q, v, layout.angular_velocity_body(self.world.data), command_body, frame=self.boundary
        )
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError("finite velocity candidate required")
        self.pending = np.asarray(target, dtype=np.float64).copy()
        return self.pending.copy()

    def commit(self) -> dict:
        if self.pending is None or self.world.control_steps != self.boundary + 1:
            raise ValueError("one physically advanced shared control required")
        actual = np.asarray(self.world.last_applied_target(self.prefix), dtype=np.float64)
        if actual.shape != (29,) or not np.isfinite(actual).all():
            raise ValueError("actual player target required")
        if self.active:
            if not np.array_equal(actual, self.pending):
                raise ValueError("active velocity candidate must be the applied target")
        else:
            self.stream.last_action = policy_action_from_hg_target(
                actual,
                np.asarray(self.stream.params.action_offset, np.float64),
                np.asarray(self.stream.params.action_scale, np.float64),
                np.arange(29),
            )
        row = dict(
            active=self.active,
            proposed_target=self.pending.copy(),
            actual_target=actual.copy(),
            action_memory_after=self.stream.last_action.copy(),
        )
        self.boundary = self.world.control_steps
        self.pending = None
        return row
