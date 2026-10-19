"""SIM-only native held-target body for comparing different motion policies.

Owns one continuous MuJoCo state across policy handoffs. A caller may replay
targets or recompute a policy, but must report which; this body proves neither
policy causality nor teacher qualification. No hardware/runtime integration.
"""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState


class G1TargetEpisode:
    """Fixed H300, ten native2ms steps per held canonical29-joint target."""

    def __init__(
        self,
        model: Any,
        *,
        world_hash: str,
        kp: np.ndarray,
        kd: np.ndarray,
        course: tuple[float, float, float],
        defaults: np.ndarray,
    ) -> None:
        checked = G1PDTorqueForecaster(
            model, expected_world_hash=world_hash, kp=kp, kd=kd, max_forecast_steps=3000
        )
        initial = NativeFirstTouchInitialState(course, defaults)
        self.model, self.data = model, mujoco.MjData(model)
        self.qi, self.vi, self.ai = checked._qi, checked._vi, checked._ai
        self.ball, self.bg, self.pelvis = checked._ball, checked._bg, checked._pelvis
        self.bq, self.bv = checked._bq, checked._bv
        self.kp, self.kd, self.caps = kp.copy(), kd.copy(), checked._limits.copy()
        self.contacts = [
            int(model.body(name).id)
            for name in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_ankle_pitch_link",
                "right_ankle_pitch_link",
                "left_knee_link",
                "right_knee_link",
            )
        ]
        d = self.data
        d.qpos[:7] = (0, 0, 0.793, 1, 0, 0, 0)
        d.qpos[self.qi] = defaults
        d.qpos[self.bq : self.bq + 7] = (*course[:2], 0.13, 1, 0, 0, 0)
        d.qvel[self.bv : self.bv + 6] = (course[2], 0, 0, 0, course[2] / 0.11, 0)
        initial.verify(list(course), *self.canonical())
        mujoco.mj_forward(model, d)
        self.warnings = tuple(int(w.number) for w in d.warning)
        self.next_frame = self.attempted_steps = self.integrated_steps = 0
        self.failed = False

    def canonical(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        return (
            np.concatenate((d.qpos[:7], d.qpos[self.qi], d.qpos[self.bq : self.bq + 7])),
            np.concatenate((d.qvel[:6], d.qvel[self.vi], d.qvel[self.bv : self.bv + 6])),
        )

    def step(self, frame: int, target: np.ndarray) -> dict[str, np.ndarray]:
        if (
            self.failed
            or type(frame) is not int
            or frame != self.next_frame
            or not 0 <= frame < 300
            or type(target) is not np.ndarray
            or target.shape != (29,)
            or target.dtype != np.float64
            or not np.isfinite(target).all()
            or np.max(np.abs(target)) > 1000
            or mujoco.get_mjcb_control() is not None
            or not np.isclose(self.data.time, frame * 0.02, atol=1e-8, rtol=0)
        ):
            raise ValueError("sequential finite canonical SIM target and current clock required")
        # Any exception after admission makes this body unusable; never retry a
        # partly integrated frame or silently reset warmstart/contact history.
        self.failed = True
        model, d = self.model, self.data
        q, v = self.canonical()
        values = dict(
            canonical_qpos=q[None], canonical_qvel=v[None], joint_target_rad=target[None].copy()
        )
        pd, torques, actual, heights, native_q, native_v = [], [], [], [], [], []
        forces = np.zeros(6)
        for _ in range(10):
            raw = (target - d.qpos[self.qi]) * self.kp - d.qvel[self.vi] * self.kd
            tau = np.clip(raw, -self.caps, self.caps)
            d.ctrl[self.ai] = tau
            pd.append(raw.copy())
            torques.append(tau.copy())
            old_time = float(d.time)
            self.attempted_steps += 1
            mujoco.mj_step(model, d)
            self.integrated_steps += 1
            if (
                not np.isclose(d.time, old_time + 0.002, atol=1e-12, rtol=0)
                or tuple(int(w.number) for w in d.warning) != self.warnings
                or any(
                    not np.isfinite(a).all()
                    for a in (d.qpos, d.qvel, d.qacc, d.actuator_force, d.qfrc_constraint)
                )
            ):
                raise ValueError("native body warning/reset/nonfinite physics")
            actual.append(d.actuator_force[self.ai].copy())
            heights.append(float(d.xpos[self.pelvis, 2]))
            nq, nv = self.canonical()
            native_q.append(nq)
            native_v.append(nv)
            for ci in range(d.ncon):
                c = d.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                if self.bg not in (g1, g2):
                    continue
                body = int(model.geom_bodyid[g2 if g1 == self.bg else g1])
                if body in self.contacts:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, d, ci, wrench)
                    slot = self.contacts.index(body)
                    forces[slot] = max(forces[slot], float(np.linalg.norm(wrench[:3])))
        values.update(
            pd_requested_torque_nm=np.asarray(pd)[None],
            torque_nm=np.asarray(torques)[None],
            actual_actuator_force_nm=np.asarray(actual)[None],
            pelvis_z_per_substep_m=np.asarray(heights)[None],
            force_n=forces[None],
            ball_position_after_step_m=d.xpos[self.ball].copy()[None],
            target_body_canonical_qpos_after_substep=np.asarray(native_q)[None],
            target_body_canonical_qvel_after_substep=np.asarray(native_v)[None],
        )
        self.next_frame += 1
        self.failed = False
        return values

    def review(self) -> dict[str, Any]:
        if (
            self.failed
            or self.next_frame != 300
            or self.attempted_steps != 3000
            or self.integrated_steps != 3000
        ):
            raise ValueError("whole native H300 target body required")
        return dict(
            schema="soccer.physics.g1_target_body.v1",
            frames=300,
            attempted_steps=self.attempted_steps,
            integrated_steps=self.integrated_steps,
            policy_causality_verified=False,
            teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
