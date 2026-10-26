"""Research-only private CFtrack skill on an owned native SIM body.

The caller pins and constructs the external, transport-free policy in a fresh
process. This adapter imports no external deployment runner and does not load
assets, move bodies, change gains, approve a policy or advance physics.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld


class SharedOmniContactPlayer:
    def __init__(
        self,
        world: SharedNativeMotorWorld,
        prefix: str,
        policy: Any,
        *,
        expected_policy_type: type[Any],
        goal: np.ndarray,
        reference_source: str = "CFgen",
    ) -> None:
        world._require_current_boundary()
        if (
            type(policy) is not expected_policy_type
            or prefix not in world.layouts
            or policy.task != "kickball"
            or reference_source not in ("CFgen", "NPZmotion")
            or policy.reference_source != reference_source
            or policy.counter_step != 0
            or policy.async_stage_thread is not None
            or goal.shape != (3,)
            or goal.dtype.kind != "f"
            or not np.isfinite(goal).all()
            or policy.num_obs != 1244
            or policy.single_obs_dim != 141
            or policy.history_len != 5
            or tuple(policy.future_frames) != (0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50)
        ):
            raise ValueError("fresh pinned kick-only CFtrack1244 private SIM policy required")
        inputs, outputs = policy.ort_session.get_inputs(), policy.ort_session.get_outputs()
        if (
            [(x.name, list(x.shape)) for x in inputs] != [("obs", [1, 1244]), ("time_step", [1, 1])]
            or not outputs
            or (outputs[0].name, list(outputs[0].shape)) != ("actions", [1, 29])
            or policy.ort_session.get_providers() != ["CPUExecutionProvider"]
        ):
            raise ValueError("qualified CPU CFtrack1244/time_step-to-HG29 interface required")
        order, inverse = np.asarray(policy.lab2mj), np.asarray(policy.mj2lab)
        if (
            order.shape != (29,)
            or inverse.shape != (29,)
            or order.dtype.kind not in "iu"
            or inverse.dtype.kind not in "iu"
            or sorted(order.tolist()) != list(range(29))
            or sorted(inverse.tolist()) != list(range(29))
            or not np.array_equal(order[inverse], np.arange(29))
        ):
            raise ValueError("mutually inverse HG29/Lab joint permutations required")
        param = world.parameters[prefix]
        if not np.array_equal(policy.kps_lab[order], param.stiffness) or not np.array_equal(
            policy.kds_lab[order], param.damping
        ):
            raise ValueError("explicit world PD gains must match the declared CFtrack profile")
        identities = (id(policy), id(policy.state_cmd), id(policy.policy_output))
        owners = world._private_policy_owners
        if prefix in owners or any(set(identities) & set(ids) for ids in owners.values()):
            raise ValueError("one private CFtrack policy/state/output per player required")
        self.world, self.prefix, self.policy = world, prefix, policy
        self.reference_source = reference_source
        self.layout = world.layouts[prefix]
        self.boundary, self.commits = world.control_steps, 0
        self.pending: np.ndarray | None = None
        self.motor_plan: CFtrackMotorResidualPlan | None = None
        self.failed = False
        owners[prefix] = identities
        try:
            self._observe()
            policy.goal_pos_override = goal.astype(np.float32, copy=True)
            policy.enter()
            world._require_current_boundary()
            if policy.async_stage_thread is not None or policy.reference_source != reference_source:
                raise ValueError("asynchronous stage planning is not qualified")
        except BaseException:
            self.failed = True
            raise

    def install_motor_plan(self, plan: CFtrackMotorResidualPlan) -> None:
        from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash

        w = self.world
        w._require_current_boundary()
        if (
            type(plan) is not CFtrackMotorResidualPlan
            or self.failed
            or self.pending is not None
            or self.boundary != w.control_steps
            or self.motor_plan is not None
            or self.prefix in w._torque_layer_owners
            or not np.all(w.model.jnt_limited[self.layout.joints])
        ):
            raise ValueError(
                "one fresh donor-bound motor plan and explicit physical joint limits required"
            )
        plan.require_donor(
            policy_hash=cftrack_state_hash(self.policy), counter=self.policy.counter_step
        )
        limits = w.model.jnt_range[self.layout.joints].astype(np.float64)
        if not np.isfinite(limits).all() or (limits[:, 0] >= limits[:, 1]).any():
            raise ValueError("finite ordered HG29 physical joint limits required")
        self.motor_plan = plan

    def _observe(self) -> dict[str, Any]:
        w, p, lo = self.world, self.policy, self.layout
        q, v = lo.canonical(w.data)
        s = p.state_cmd
        s.q, s.dq = q[7:36].copy(), v[6:35].copy()
        s.base_pos, s.base_quat = q[:3].copy(), q[3:7].copy()
        s.lin_vel, s.ang_vel = v[:3].copy(), v[3:6].copy()
        qw, qx, qy, qz = s.base_quat
        s.gravity_ori = np.array(
            [
                2 * (-qz * qx + qw * qy),
                -2 * (qz * qy + qw * qx),
                1 - 2 * (qw * qw + qz * qz),
            ],
            np.float32,
        )
        # Match the author's measured object convention: cached native xpos
        # at this boundary, NOT future reference-object positions.
        s.ball_pos, s.ball_quat = w.data.xpos[lo.ball].copy(), w.data.xquat[lo.ball].copy()
        s.obj_pos, s.obj_quat = s.ball_pos.copy(), s.ball_quat.copy()
        return dict(control_step=w.control_steps, canonical_qpos=q, canonical_qvel=v)

    def propose(self) -> tuple[np.ndarray, dict[str, Any]]:
        w, p = self.world, self.policy
        w._require_current_boundary()
        if self.failed or self.pending is not None or self.boundary != w.control_steps:
            raise ValueError("one fresh private CFtrack proposal per shared boundary required")
        try:
            if p.reference_source != self.reference_source:
                raise ValueError("admitted CFtrack reference source changed")
            measured = self._observe()
            old_tick = p.counter_step
            p.run()
            w._require_current_boundary()
            if (
                p.counter_step != old_tick + 1
                or p.reference_source != self.reference_source
                or p.async_stage_thread is not None
                or not np.array_equal(p.policy_output.kps, w.parameters[self.prefix].stiffness)
                or not np.array_equal(p.policy_output.kds, w.parameters[self.prefix].damping)
            ):
                raise ValueError("one synchronous CFtrack tick with unchanged PD required")
            target = np.asarray(p.policy_output.actions, dtype=np.float64)
            if target.shape != (29,) or not np.isfinite(target).all():
                raise ValueError("finite HG29 CFtrack output required")
            measured["nn_decoded_target"] = target.copy()
            if self.motor_plan is not None:
                target = self.motor_plan.apply(
                    target,
                    counter=old_tick,
                    limits_hg=w.model.jnt_range[self.layout.joints].astype(np.float64),
                )
            measured["motor_residual"] = target - measured["nn_decoded_target"]
            self.pending = target.copy()
        except BaseException:
            self.failed = True
            raise
        return target.copy(), measured

    def commit(self) -> None:
        w = self.world
        w._require_current_boundary(allow_failed=True)
        if self.failed or self.pending is None or w.control_steps != self.boundary + 1:
            raise ValueError("one shared physical advance before CFtrack commitment required")
        if not np.array_equal(w.last_applied_target(self.prefix), self.pending):
            self.failed = True
            raise ValueError("actual executed target differs from private CFtrack proposal")
        self.pending = None
        self.boundary = w.control_steps
        self.commits += 1
