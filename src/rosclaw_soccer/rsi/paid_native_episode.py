"""Reusable SIM-only episode mechanics for the separately qualified teacher.

External research assets/implementations are injected after runner provenance
checks. No policy, driver, automatic reset, observation padding, or reward here.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout


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
        self.frame, self.done, self.native_steps = 0, True, 0

    def reset(self, *, ball: np.ndarray, velocity: np.ndarray, goal: np.ndarray) -> None:
        import mujoco

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
        for contact in self.env.data.contact:
            x, y = (int(self.env.model.geom_bodyid[g]) for g in contact.geom)
            if (x == self.layout.ball and y in self.robot) or (
                y == self.layout.ball and x in self.robot
            ):
                raise ValueError("initial ball/body overlap; no resampling")
        self.goal = goal.copy()
        self.frame, self.done, self.ready = 0, False, False
        self.last_action = np.zeros(29, np.float32)
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
        assert mujoco.get_mjcb_control() is None

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
            self.native_steps += 1
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
        self.frame += 1
        self.ready = False
        self.done = row["body_failed"] or self.frame == self.clip.length + 60
        return row
