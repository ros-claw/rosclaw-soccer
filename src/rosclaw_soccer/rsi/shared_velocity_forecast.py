"""Owned one-control CPU velocity-NN forecast on the native SIM body.

This qualifies a component only: it cannot yet predict a future skill switch
or a multi-player game. The actual donor is never advanced or reinitialized.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.velocity_stream_branch import (
    VelocityStream,
    fork_velocity_stream,
    velocity_stream_state_hash,
)
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_target_forecast import SharedTargetForecaster


class SharedVelocityForecaster:
    def __init__(
        self,
        world: SharedNativeMotorWorld,
        stream: VelocityStream,
        *,
        expected_model_hash: str,
        native_step_budget: int,
    ) -> None:
        if len(world.layouts) != 1 or world._private_policy_owners:
            raise ValueError("one plain current SIM locomotion world required")
        velocity_stream_state_hash(stream, completed_control=world.control_steps)
        self.world, self.stream = world, stream
        self.prefix = next(iter(world.layouts))
        params = world.parameters[self.prefix]
        if not np.array_equal(params.stiffness, stream.params.stiffness) or not np.array_equal(
            params.damping, stream.params.damping
        ):
            raise ValueError("same current velocity PD profile required")
        self.native = SharedTargetForecaster(
            world, expected_model_hash=expected_model_hash, native_step_budget=native_step_budget
        )
        self.neural_calls = 0

    def counters(self) -> dict[str, int]:
        return dict(self.native.counters(), future_velocity_nn_calls=self.neural_calls)

    def forecast(self, command: np.ndarray) -> dict[str, Any]:
        world, stream = self.world, self.stream
        world._require_current_boundary()
        if (
            self.native.attempted_steps + 10 > self.native.budget
            or self.neural_calls + 1 > self.native.budget // 10
        ):
            raise RuntimeError("native velocity prediction budget exhausted; no retry extension")
        pd = world.parameters[self.prefix]
        if not np.array_equal(pd.stiffness, stream.params.stiffness) or not np.array_equal(
            pd.damping, stream.params.damping
        ):
            raise ValueError("velocity future cannot silently use a switched kick PD profile")
        before = velocity_stream_state_hash(stream, completed_control=world.control_steps)
        ghost = fork_velocity_stream(stream, completed_control=world.control_steps)
        lo = world.layouts[self.prefix]
        q, v = lo.canonical(world.data)
        calls = stream.neural_calls
        try:
            target = ghost.propose(
                q, v, lo.angular_velocity_body(world.data), command, frame=world.control_steps
            )
            prediction = self.native.forecast({self.prefix: target})
        finally:
            self.neural_calls += ghost.neural_calls - calls
            if velocity_stream_state_hash(stream, completed_control=world.control_steps) != before:
                raise ValueError("actual locomotion stream changed during copied prediction")
        return dict(
            **prediction,
            joint_target=target.copy(),
            observation=ghost.last_observation.copy(),
            action=ghost.last_action.copy(),
            phase_after=float(ghost.phase),
            actual_completed_control=stream.next_frame,
            copied_next_frame=ghost.next_frame,
            donor_stream_hash=before,
            copied_stream_after_hash=velocity_stream_state_hash(
                ghost, completed_control=world.control_steps + 1
            ),
            whole_handoff_prediction_qualified=False,
        )
