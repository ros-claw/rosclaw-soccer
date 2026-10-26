"""Private CFtrack NN/native futures, preserving measured donor software and clock."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_contact_reference import (
    CFtrackContactPlan,
    condition_cftrack_contact_reference,
)
from rosclaw_soccer.rsi.cftrack_motor_residual_plan import (
    CFtrackMotorResidualPlan,
    MotorTargetRejected,
)
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash, copy_cftrack_policy
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


class SharedCFtrackForecaster:
    def __init__(
        self,
        player: SharedOmniContactPlayer,
        *,
        expected_policy_type: type[Any],
        expected_model_hash: str,
        native_step_budget: int,
    ) -> None:
        import mujoco

        w = player.world
        w._require_current_boundary()
        if (
            type(player) is not SharedOmniContactPlayer
            or type(player.policy) is not expected_policy_type
            or player.failed
            or player.pending is not None
            or player.boundary != w.control_steps
            or len(w.layouts) != 1
            or w._torque_layer_owners
            or w.model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
            or compiled_model_hash(w.model) != expected_model_hash
            or type(native_step_budget) is not int
            or not 10 <= native_step_budget <= 60000
            or native_step_budget % 10
        ):
            raise ValueError("qualified synchronous single-player CFtrack Euler forecast required")
        copy_cftrack_policy(player.policy)
        self.player, self.model_hash = player, expected_model_hash
        self.policy_type = expected_policy_type
        self.budget = native_step_budget
        self.calls = self.attempted_steps = self.integrated_steps = 0
        self.future_nn_calls = self.attempted_nn_calls = 0

    def counters(self) -> dict[str, int]:
        return dict(
            forecast_calls=self.calls,
            attempted_native_steps=self.attempted_steps,
            integrated_native_steps=self.integrated_steps,
            future_nn_calls=self.future_nn_calls,
            attempted_future_nn_calls=self.attempted_nn_calls,
            maximum_native_steps=self.budget,
        )

    def forecast(
        self,
        *,
        controls: int,
        contact_plan: CFtrackContactPlan | None = None,
        motor_plan: CFtrackMotorResidualPlan | None = None,
    ) -> dict[str, Any]:
        import mujoco

        source, donor = self.player, self.player.world
        donor._require_current_boundary()
        if (
            source.failed
            or source.pending is not None
            or source.boundary != donor.control_steps
            or type(source.policy) is not self.policy_type
            or compiled_model_hash(donor.model) != self.model_hash
            or type(controls) is not int
            or not 1 <= controls <= 360
            or self.attempted_steps + controls * 10 > self.budget
            or (
                motor_plan is not None
                and (source.motor_plan is not None or contact_plan is not None)
            )
        ):
            raise ValueError(
                "current pre-proposal CFtrack boundary and reserved finite budget required"
            )
        before = cftrack_state_hash(source.policy)
        if motor_plan is not None:
            if type(motor_plan) is not CFtrackMotorResidualPlan:
                raise ValueError("explicit bounded CFtrack motor plan required")
            motor_plan.require_donor(policy_hash=before, counter=source.policy.counter_step)
        policy = copy_cftrack_policy(source.policy)
        receipt = None
        if contact_plan is not None:
            receipt = condition_cftrack_contact_reference(policy, contact_plan)
        # Charge the entire request BEFORE physical allocation, never refund
        # failed/early-terminated work into an unbounded retry loop.
        self.attempted_steps += controls * 10
        self.calls += 1
        data = mujoco.MjData(donor.model)
        mujoco.mj_copyData(data, donor.model, donor.data)
        owned = SharedNativeMotorWorld(
            donor.model,
            data,
            players=donor.parameters,
            ball_body_name=donor.model.body(donor.ball).name,
        )
        ghost = object.__new__(SharedOmniContactPlayer)
        ghost.world, ghost.prefix, ghost.policy = owned, source.prefix, policy
        ghost.reference_source = source.reference_source
        ghost.motor_plan = source.motor_plan
        ghost.layout = owned.layouts[source.prefix]
        ghost.boundary, ghost.commits = 0, source.commits
        ghost.pending, ghost.failed = None, False
        owned._private_policy_owners[source.prefix] = (
            id(policy),
            id(policy.state_cmd),
            id(policy.policy_output),
        )
        rows, targets, decoded = [], [], []
        failure: dict[str, Any] | None = None
        try:
            if motor_plan is not None:
                ghost.install_motor_plan(motor_plan)
            for _ in range(controls):
                self.attempted_nn_calls += 1
                previous_counter = policy.counter_step
                try:
                    target, observed = ghost.propose()
                finally:
                    # A successful NN can be followed by a rejected residual
                    # before physics. Account for that call, not only rows.
                    self.future_nn_calls += policy.counter_step - previous_counter
                row = owned.advance({source.prefix: target})
                targets.append(target.copy())
                decoded.append(observed["nn_decoded_target"])
                rows.append(row)
                ghost.commit()
                if owned.failed:
                    break
        except Exception as error:
            # Preserve already integrated private evidence, including a first
            # proposal failure. An error is never a qualified empty future.
            failure = dict(type=type(error).__name__, message=str(error))
            if type(error) is MotorTargetRejected:
                failure["target_rejection"] = error.target_rejection
        finally:
            self.integrated_steps += owned.native_steps
            donor._require_current_boundary()
            if cftrack_state_hash(source.policy) != before:
                source.failed = True
                raise ValueError("CFtrack forecast mutated donor software/FK/model state")
        arrays = {
            k: np.concatenate([row[k] for row in rows]) if rows else np.empty(shape)
            for k, shape in (
                ("qpos", (0, donor.model.nq)),
                ("qvel", (0, donor.model.nv)),
                ("torque", (0, donor.model.nu)),
                ("ball_forces", (0, 1, 3)),
            )
        }
        return dict(
            **arrays,
            joint_targets=np.array(targets).reshape(-1, 29),
            nn_decoded_targets=np.array(decoded).reshape(-1, 29),
            controls=len(rows),
            native_steps=owned.native_steps,
            body_failed=owned.failed,
            runtime_failure=failure,
            donor_control_step=donor.control_steps,
            donor_time_sec=float(donor.data.time),
            donor_policy_unchanged=True,
            donor_body_unchanged=True,
            private_fk_data=True,
            enter_or_history_reset_called=False,
            reference_conditioning=receipt,
            hardware_authorized=False,
        )
