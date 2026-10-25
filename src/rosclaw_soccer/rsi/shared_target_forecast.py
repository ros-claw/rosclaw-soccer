"""Owned one-control forecasts using the SAME shared native PD executor.

This is a SIM teacher primitive, not a neural policy or real-robot authority.
All players' held targets must be explicit. No root/ball forcing, gain rewrite,
hidden torque layer, or uncharged forecast is permitted.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld
from rosclaw_soccer.sim.contracts import hash_bytes


def compiled_model_hash(model: Any) -> str:
    import mujoco

    binary = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    return str(hash_bytes(binary.tobytes()))


class SharedTargetForecaster:
    def __init__(
        self, world: SharedNativeMotorWorld, *, expected_model_hash: str, native_step_budget: int
    ) -> None:
        import mujoco

        world._require_current_boundary()
        if (
            type(native_step_budget) is not int
            or not 10 <= native_step_budget <= 600000
            or native_step_budget % 10
        ):
            raise ValueError("explicit bounded whole-control forecast budget required")
        if compiled_model_hash(world.model) != expected_model_hash or (
            world.model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
            or world._torque_layer_owners
        ):
            raise ValueError("exact callback-free Euler body and unmodified shared PD law required")
        self.world = world
        self.model_hash = expected_model_hash
        self.budget = native_step_budget
        self.attempted_steps = 0
        self.integrated_steps = 0
        self.calls = 0
        self.scratch = mujoco.MjData(world.model)

    def counters(self) -> dict[str, int]:
        return dict(
            forecast_calls=self.calls,
            attempted_native_steps=self.attempted_steps,
            integrated_native_steps=self.integrated_steps,
            maximum_native_steps=self.budget,
        )

    def forecast(self, targets: Mapping[str, np.ndarray]) -> dict[str, Any]:
        import mujoco

        donor = self.world
        donor._require_current_boundary()
        if donor._torque_layer_owners or compiled_model_hash(donor.model) != self.model_hash:
            raise ValueError("donor model/control law changed since forecast admission")
        if self.attempted_steps + 10 > self.budget:
            raise RuntimeError("forecast budget exhausted; no automatic extension or retry")
        mujoco.mj_copyData(self.scratch, donor.model, donor.data)
        owned = SharedNativeMotorWorld(
            donor.model,
            self.scratch,
            players=donor.parameters,
            ball_body_name=donor.model.body(donor.ball).name,
            measure_interplayer_contact=donor.measure_interplayer_contact,
        )
        self.calls += 1
        self.attempted_steps += 10  # Rejected/failed attempts also consume the declared budget.
        try:
            result = owned.advance(targets)
        finally:
            self.integrated_steps += owned.native_steps
            donor._require_current_boundary()  # The actual execution world must not be modified.
        if owned.native_steps != 10:
            raise ValueError("one complete native control forecast required")
        return dict(
            **result,
            donor_control_frame=donor.control_steps,
            donor_time=float(donor.data.time),
            predicted_time=float(self.scratch.time),
            native_steps=owned.native_steps,
            donor_unchanged=True,
            prediction_is_not_actual_execution=True,
            promotion_authorized=False,
            hardware_authorized=False,
        )
