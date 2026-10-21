"""Reusable SIM-only episode mechanics for the separately qualified teacher.

External research assets/implementations are injected after runner provenance
checks. No policy, driver, automatic reset, observation padding, or reward here.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout


@dataclass
class NativeStepCounter:
    native_steps: int = 0


class PaidNativeEpisode:
    def __init__(
        self,
        env: Any,
        clip: Any,
        parameters: dict[str, np.ndarray],
        order: np.ndarray,
        effort: np.ndarray,
        build_observation: Any,
        torso_index: int,
    ) -> None:
        import mujoco

        if mujoco.get_mjcb_control() is not None or env.dt != 0.002:
            raise ValueError("callback-free native2ms simulation required")
        self.env, self.clip = env, clip
        self.layout = NativeVelocityLayout(env.model, robot_prefix="", ball_body_name="soccer_ball")
        self.offset, self.scale, self.kp, self.kd = [
            parameters[name][order].copy()
            for name in ("default_joint_pos", "action_scale", "joint_stiffness", "joint_damping")
        ]
        self.order, self.effort = order.copy(), effort.copy()
        self.build_observation, self.torso_index = build_observation, torso_index
        m = env.model
        self.feet = [m.body(s + "_ankle_roll_link").id for s in ("left", "right")]
        self.robot = set(env.body_ids.values())
        if (
            m.nq != 43
            or m.nv != 41
            or m.nu != 29
            or not np.isclose(m.body_mass[self.layout.ball], 0.43, rtol=0, atol=1e-12)
            or not 1 <= clip.length <= 300
            or any(
                x.shape != (29,) or not np.isfinite(x).all()
                for x in (self.offset, self.scale, self.kp, self.kd, self.effort)
            )
        ):
            raise ValueError("qualified standard-ball29motor episode required")
        self.frame, self.done = 0, True
        self.termination_reason = "NOT_STARTED"
        self.work = NativeStepCounter()

    @property
    def native_steps(self) -> int:
        return self.work.native_steps

    def reset(self, *, ball: np.ndarray, velocity: np.ndarray, goal: np.ndarray) -> None:
        if not self.done:
            raise ValueError("no hidden episode reset before terminal boundary")
        if (
            ball.shape != (3,)
            or velocity.shape != (2,)
            or goal.shape != (3,)
            or not all(np.isfinite(x).all() for x in (ball, velocity, goal))
        ):
            raise ValueError("finite explicit world ball/velocity/goal required")
        self.env.reset(self.clip, np.zeros(2), ball, velocity)
        normal = goal[:2] / np.linalg.norm(goal[:2])
        self.env.set_goal_marker(goal, normal)
        self._begin_local(goal, np.zeros(29, np.float32))

    def _begin_local(self, goal: np.ndarray, previous_action: np.ndarray) -> None:
        for contact in self.env.data.contact:
            x, y = (int(self.env.model.geom_bodyid[g]) for g in contact.geom)
            if (x == self.layout.ball and y in self.robot) or (
                y == self.layout.ball and x in self.robot
            ):
                raise ValueError("initial ball/body overlap; no resampling")
        self.goal = goal.copy()
        self.frame, self.done, self.ready = 0, False, False
        self.termination_reason = "ACTIVE"
        self.last_action = previous_action.copy()
        self.ref_pos = np.zeros(29, np.float32)
        self.ref_vel = np.zeros(29, np.float32)
        self.ref_ang = np.zeros(3, np.float32)
        self.warning = tuple(int(x.number) for x in self.env.data.warning)
        self.initial = {
            k: getattr(self.env.data, k).copy()
            for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "mocap_pos", "mocap_quat")
        }
        self.initial["time"] = np.array(self.env.data.time)
        # Reset only prepares state. Each observation below refreshes native
        # kinematics at the same boundary as the qualified cached-standard arm.

    def begin_current_state(
        self, clip: Any, *, goal: np.ndarray, preserve_reference: bool = False
    ) -> None:
        """Explicit new reference boundary; do not write ANY mjData field.

        Actual last executed action is carried. A caller may explicitly reset
        its neural h/c for a new skill, but may not erase body state or time.
        Body-failure terminals cannot be restarted by this entry.
        """
        if not self.done or self.termination_reason != "REFERENCE_COMPLETE":
            raise ValueError("completed healthy reference boundary required")
        if not 1 <= clip.length <= 300 or goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("bounded next reference and finite existing goal required")
        if not np.array_equal(goal, self.env.data.mocap_pos[self.env.goal_mocap_id]):
            raise ValueError("handoff must preserve the declared world target")
        q, _ = self.layout.canonical(self.env.data)
        if q[2] < 0.65 or 1 - 2 * (q[4] ** 2 + q[5] ** 2) < 0.9:
            raise ValueError("unhealthy actual body cannot start another skill")
        if tuple(int(x.number) for x in self.env.data.warning) != self.warning:
            raise ValueError("prior native warning cannot be hidden by a new skill")
        previous = self.last_action.copy()
        references = None
        if preserve_reference:
            references = (self.ref_pos.copy(), self.ref_vel.copy(), self.ref_ang.copy())
            if any(
                x.shape != shape or not np.isfinite(x).all()
                for x, shape in zip(references, ((29,), (29,), (3,)), strict=True)
            ):
                raise ValueError("finite actual cached references required for declared carry")
        self._begin_local(goal, previous)
        self.clip = clip
        if references is not None:
            self.ref_pos, self.ref_vel, self.ref_ang = references

    def admit_reference_recovery(self) -> None:
        """NEW diagnostic boundary at reference end, not an old exam completion.

        No mjData field is written and no reference/tail is padded. The caller
        must declare that the old 60-control fixed-target tail was NOT run.
        A next skill cannot begin until a separate active recovery completes.
        """
        q, v = self.layout.canonical(self.env.data)
        if (
            self.done
            or self.ready
            or self.frame != self.clip.length
            or self.termination_reason != "ACTIVE"
            or not np.isfinite(q).all()
            or not np.isfinite(v).all()
            or q[2] < 0.65
            or 1 - 2 * (q[4] ** 2 + q[5] ** 2) < 0.9
            or tuple(int(x.number) for x in self.env.data.warning) != self.warning
        ):
            raise ValueError("only a healthy actual reference end may admit NEW recovery")
        self.done = True
        self.termination_reason = "RECOVERY_ADMITTED"

    def advance_hold(self) -> dict[str, Any]:
        """Advance a separately declared post-reference PD hold, no neural/ref padding."""
        import mujoco

        if not self.done or self.termination_reason != "REFERENCE_COMPLETE":
            raise ValueError("only healthy completed references can hold")
        mujoco.mj_forward(self.env.model, self.env.data)
        row = self._integrate_target(False)
        row["post_reference_hold"] = True
        if row["body_failed"]:
            self.termination_reason = "BODY_FAILURE"
        return row

    def observation(self) -> np.ndarray:
        import mujoco

        if self.done or self.frame >= self.clip.length or self.ready:
            raise ValueError("one causal observation per real neural control required")
        mujoco.mj_forward(self.env.model, self.env.data)
        obs = self.build_observation(
            self.env,
            self.clip,
            self.frame,
            self.last_action,
            self.ref_pos,
            self.ref_vel,
            self.ref_ang,
            self.env.ball_pos,
            self.goal,
            self.offset,
            obs_dim=None,
        )
        if obs.shape != (160,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
            raise ValueError("exact finite160 observations without fallback required")
        self.ready = True
        return np.array(obs, copy=True)

    def forces(self) -> np.ndarray:
        import mujoco

        m, d = self.env.model, self.env.data
        value = np.zeros(3)
        for ci in range(d.ncon):
            c = d.contact[ci]
            x, y = (int(m.geom_bodyid[g]) for g in c.geom)
            other = y if x == self.layout.ball else x if y == self.layout.ball else -1
            if other in self.robot:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, wrench)
                role = self.feet.index(other) if other in self.feet else 2
                value[role] = max(value[role], np.linalg.norm(wrench[:3]))
        return value

    def step(self, action: np.ndarray | None) -> dict[str, Any]:
        import mujoco

        if self.done:
            raise ValueError("terminal episode requires explicit reset")
        neural = self.frame < self.clip.length
        m, d = self.env.model, self.env.data
        if neural:
            if (
                not self.ready
                or action is None
                or action.shape != (29,)
                or action.dtype != np.float32
                or not np.isfinite(action).all()
            ):
                raise ValueError("exact actual29 action after observation required")
            self.target = self.offset + action[self.order] * self.scale
            self.last_action = action.copy()
            self.ref_pos, self.ref_vel = (
                self.clip.joint_pos[self.frame],
                self.clip.joint_vel[self.frame],
            )
            self.ref_ang = self.clip.body_ang_vel_w[self.frame, self.torso_index]
        else:
            if action is not None:
                raise ValueError("held target tail must not add a neural proposal")
            mujoco.mj_forward(m, d)
        row = self._integrate_target(neural)
        self.frame += 1
        self.ready = False
        self.done = row["body_failed"] or self.frame == self.clip.length + 60
        self.termination_reason = (
            "BODY_FAILURE"
            if row["body_failed"]
            else "REFERENCE_COMPLETE"
            if self.done
            else "ACTIVE"
        )
        return row

    def _integrate_target(self, neural: bool) -> dict[str, Any]:
        import mujoco

        m, d = self.env.model, self.env.data
        row = dict(
            native_qpos_before=d.qpos.copy(),
            native_qvel_before=d.qvel.copy(),
            target=self.target.copy(),
            neural_control=neural,
            time_before=float(d.time),
        )
        samples = []
        for _ in range(10):
            torque = np.clip(
                self.kp * (self.target - self.env.joint_pos) - self.kd * self.env.joint_vel,
                -self.effort,
                self.effort,
            )
            if not np.isfinite(torque).all():
                raise ValueError("nonfinite native PD torque; no repair")
            d.ctrl[self.layout.ai] = torque
            mujoco.mj_step(m, d)
            self.work.native_steps += 1
            if tuple(int(x.number) for x in d.warning) != self.warning or not np.array_equal(
                d.actuator_force[self.layout.ai], torque
            ):
                raise ValueError("native warning or actual bounded torque law differs")
            q, v = self.layout.canonical(d)
            samples.append(dict(qpos=q, qvel=v, forces=self.forces(), torque=torque.copy()))
        row.update({k: np.stack([s[k] for s in samples]) for k in samples[0]})
        q = row["qpos"]
        row["body_failed"] = bool(
            ((q[:, 2] < 0.65) | (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2) < 0.9)).any()
        )
        return row
