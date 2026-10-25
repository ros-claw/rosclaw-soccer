"""Private RoboNaldo neural skill on a measured shared SIM body.

Asset loading/qualification belongs to the experiment, not this adapter. Each
player supplies its own already-created simulation policy. No transport,
physics reset, integration, gain change or automatic policy promotion occurs.
"""

import contextlib
import io
from typing import TYPE_CHECKING, Any

import numpy as np

from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld

if TYPE_CHECKING:
    from rosclaw_soccer.providers.g1.kick_measured_history import KickMeasuredSample


class SharedRoboNaldoPlayer:
    """One proposal and exact actual-target commitment per shared boundary.

    Calling enter initializes SOFTWARE history from this body's measured state.
    It is an explicit cold skill entry, not a claim of seamless learned handoff.
    Any failed policy call poisons this adapter rather than retrying mutated
    private history. The shared world still owns all physical advancement.
    """

    def __init__(
        self,
        world: SharedNativeMotorWorld,
        prefix: str,
        policy: Any,
        *,
        goal: np.ndarray,
    ) -> None:
        import mujoco

        world._require_current_boundary()
        goal = np.asarray(goal)
        if (
            prefix not in world.layouts
            or goal.shape != (3,)
            or goal.dtype.kind != "f"
            or not np.isfinite(goal).all()
            or policy.runtime_mode != "sim"
            or policy.use_body_frame_ball
            or policy.time_step != 0
        ):
            raise ValueError("fresh SIM policy, own player and finite world goal required")
        inputs, outputs = policy.ort_session.get_inputs(), policy.ort_session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].name != "obs"
            or list(inputs[0].shape) != [1, 547]
            or outputs[0].name != "actions"
            or list(outputs[0].shape) != [1, 29]
        ):
            raise ValueError("qualified fixed547-to-HG29 policy interface required")
        param = world.parameters[prefix]
        if not np.array_equal(policy.kps, param.stiffness) or not np.array_equal(
            policy.kds, param.damping
        ):
            raise ValueError("explicit existing shared PD profile must match the policy")
        torso = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, prefix + "torso_link")
        if torso < 0:
            raise ValueError("own named torso_link required")
        layout = world.layouts[prefix]
        ancestor = torso
        while ancestor not in (0, layout.pelvis):
            ancestor = int(world.model.body_parentid[ancestor])
        if ancestor != layout.pelvis:
            raise ValueError("torso must belong to this player's actual body")
        owners = world._private_policy_owners
        identities = (id(policy), id(policy.state_cmd), id(policy.policy_output))
        if prefix in owners or any(set(identities) & set(ids) for ids in owners.values()):
            raise ValueError("one private policy/state/output per player required")
        self.world, self.prefix, self.policy = world, prefix, policy
        self.layout, self.torso = layout, torso
        self.goal = goal.astype(np.float32, copy=True)
        self.boundary = world.control_steps
        self.pending: np.ndarray | None = None
        self.failed = False
        self.proposals = 0
        self.commits = 0
        self.measured_entry_receipt: str | None = None
        owners[prefix] = identities
        self._observe()
        policy.target_pos_w = self.goal.copy()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                policy.enter()
        except BaseException:
            self.failed = True
            raise

    def install_measured_entry(
        self, samples: tuple["KickMeasuredSample", ...], *, reference_frame: int
    ) -> str:
        """Replace cold software history before the first proposal, never physics.

        The caller must already have declared/switched the PD profile. This
        method does not claim torque continuity or account for root momentum.
        A failed installation poisons the private skill, not the shared world.
        """
        from rosclaw_soccer.providers.g1.kick_measured_history import (
            install_measured_kick_history,
        )
        from rosclaw_soccer.providers.g1.kick_warmstart import rebase_kick_reference

        w, p = self.world, self.policy
        w._require_current_boundary()
        if (
            self.failed
            or self.proposals
            or self.pending is not None
            or self.boundary != w.control_steps
            or self.measured_entry_receipt is not None
            or type(reference_frame) is not int
            or not 0 <= reference_frame < p.motion_total_steps
        ):
            raise ValueError("one pre-proposal measured entry at the same boundary required")
        try:
            self._observe()
            # Install validates the ACTUAL five samples before buffer mutation;
            # unlike prepare_kick_handoff this never repeats today's state.
            receipt = install_measured_kick_history(
                p, samples, frame=w.control_steps, time_sec=float(w.data.time)
            )
            rebase_kick_reference(p, entry_frame=reference_frame)
            p.time_step = int(p.WARMUP_STEPS) + reference_frame
        except BaseException:
            self.failed = True
            raise
        self.measured_entry_receipt = receipt
        return receipt

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        w, layout = self.world, self.layout
        q, v = layout.canonical(w.data)
        s = self.policy.state_cmd
        s.q, s.dq = q[7:36].copy(), v[6:35].copy()
        s.tau_est = w.data.ctrl[layout.ai].copy()
        s.root_lin_vel_b, s.root_ang_vel_b = v[:3].copy(), v[3:6].copy()
        s.torso_pos_w = w.data.xpos[self.torso].copy()
        s.torso_quat_w = w.data.xquat[self.torso].copy()
        s.pelvis_pos_w, s.pelvis_quat_w = q[:3].copy(), q[3:7].copy()
        s.ball_pos_w = w.data.xpos[layout.ball].copy()
        s.ball_vel_w = v[35:38].copy()
        s.ball_valid = True
        return q, v

    def propose(self) -> tuple[np.ndarray, dict[str, Any]]:
        w, p = self.world, self.policy
        w._require_current_boundary()
        if self.failed or self.pending is not None or self.boundary != w.control_steps:
            raise ValueError("one fresh proposal per unchanged healthy shared boundary required")
        if not np.array_equal(p.target_pos_w, self.goal):
            self.failed = True
            raise ValueError("declared goal changed outside this skill")
        q, v = self._observe()
        before = int(p.time_step)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                p.run()
            target = np.asarray(p.policy_output.actions, dtype=np.float64)
            param = w.parameters[self.prefix]
            if (
                target.shape != (29,)
                or not np.isfinite(target).all()
                or int(p.time_step) != before + 1
                or not np.array_equal(p.policy_output.kps, param.stiffness)
                or not np.array_equal(p.policy_output.kds, param.damping)
            ):
                raise ValueError("finite HG29 target, one policy tick and unchanged gains required")
        except BaseException:
            self.failed = True
            raise
        self.pending = target.copy()
        self.proposals += 1
        return target.copy(), dict(
            canonical_qpos=q,
            canonical_qvel=v,
            policy_tick_before=before,
            policy_tick_after=int(p.time_step),
            warmup_without_NN=before < int(p.WARMUP_STEPS),
            goal=self.goal.copy(),
            proposed_target=target.copy(),
            control_step=w.control_steps,
        )

    def commit(self) -> None:
        if self.failed or self.pending is None or self.world.control_steps != self.boundary + 1:
            raise ValueError("exactly one shared physical control after proposal required")
        if not np.array_equal(self.pending, self.world.last_applied_target(self.prefix)):
            self.failed = True
            raise ValueError("executed target differs; private action history is not valid")
        self.pending = None
        self.boundary = self.world.control_steps
        self.commits += 1
