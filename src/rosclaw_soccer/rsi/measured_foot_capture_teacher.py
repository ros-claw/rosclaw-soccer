"""Measured, bounded SIM foot-capture teacher over a frozen velocity motor.

Only scratch kinematics changes. Executed PD controls and ALL physical work
belong to the shared world. This teacher is NOT the raw neural motor action.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_foot_taskspace import POSE_SCALE, FootTaskSpaceIntent


@dataclass(frozen=True)
class CaptureTeacherConfig:
    anticipation_s: float = 0.30
    maximum_joint_delta_rad: float = 0.20
    delta_slew_rad_per_control: float = 0.03
    toe_up_rad: float = 0.10
    contact_front_m: float = 0.25
    foot_height_below_ball_m: float = 0.0
    shin_clearance: bool = True

    def validate(self) -> None:
        if (
            not 0.15 <= self.anticipation_s <= 0.40
            or not 0 < self.maximum_joint_delta_rad <= 0.20
            or not 0 < self.delta_slew_rad_per_control <= 0.05
            or not 0 <= self.toe_up_rad <= 0.25
            or not 0.20 <= self.contact_front_m <= 0.40
            or not 0 <= self.foot_height_below_ball_m <= 0.06
            or not all(
                np.isfinite(x)
                for x in (
                    self.anticipation_s,
                    self.maximum_joint_delta_rad,
                    self.delta_slew_rad_per_control,
                    self.toe_up_rad,
                    self.contact_front_m,
                    self.foot_height_below_ball_m,
                )
            )
            or type(self.shin_clearance) is not bool
        ):
            raise ValueError("finite bounded capture-teacher declaration required")


class MeasuredFootCaptureTeacher:
    def __init__(
        self, world: Any, prefix: str, params: Any, *, config: CaptureTeacherConfig
    ) -> None:
        config.validate()
        self.world, self.layout, self.config = world, world.layouts[prefix], config
        self.offset = np.array(params.action_offset, np.float32)
        self.scale = np.array(params.action_scale, np.float32)
        self.solvers = [
            FootTaskSpaceIntent(
                world.model, foot=side, center=105, latent=np.zeros((2, 6)), robot_prefix=prefix
            )
            for side in ("left", "right")
        ]
        self.next_control = world.control_steps
        self.selected = -1
        self.contact_seen = False
        self.previous_delta = np.zeros(29)

    def propose(
        self, nominal: np.ndarray, *, previous_forces: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import mujoco

        w, layout, cfg = self.world, self.layout, self.config
        if (
            w.failed
            or w.control_steps != self.next_control
            or nominal.shape != (29,)
            or nominal.dtype != np.float32
            or not np.isfinite(nominal).all()
            or previous_forces.shape != (3,)
            or not np.isfinite(previous_forces).all()
            or (previous_forces < 0).any()
        ):
            raise ValueError(
                "own consecutive shared-world boundary and measured contact feedback required"
            )
        d, m = w.data, w.model
        root = d.xpos[layout.pelvis]
        rotation = d.xmat[layout.pelvis].reshape(3, 3)
        ball = d.xpos[layout.ball]
        _, v = layout.canonical(d)
        rel = rotation.T @ (ball - root)
        velocity = rotation.T @ (v[35:38] - v[:3])
        closing = max(0.20, -float(velocity[0]))
        ttc = max(0.0, float(rel[0] - cfg.contact_front_m) / closing)
        if self.selected >= 0 and previous_forces[self.selected] > 10:
            self.contact_seen = True
        near = bool(-0.10 <= rel[0] <= 1.2 and abs(rel[1]) <= 0.55 and ball[2] <= 0.30)
        active = near and ((velocity[0] < -0.30 and ttc <= cfg.anticipation_s) or self.contact_seen)
        if active and self.selected < 0:
            costs = []
            for solver in self.solvers:
                foot = rotation.T @ (d.xpos[solver.body] - root)
                costs.append(abs(float(foot[1] - rel[1])) + 0.2 * abs(float(foot[0] - 0.1)))
            self.selected = int(np.argmin(costs))
        if self.selected >= 0 and previous_forces[self.selected] > 10:
            self.contact_seen = True
        requested = np.zeros(29)
        offset = np.zeros(6)
        iterations = 0
        if active:
            solver = self.solvers[self.selected]
            nominal_target = self.offset + nominal * self.scale
            solver.kin.qpos[:] = d.qpos
            solver.kin.qpos[layout.qi] = nominal_target
            mujoco.mj_kinematics(m, solver.kin)
            mujoco.mj_comPos(m, solver.kin)
            point = solver.kin.xpos[solver.body] + solver.kin.xmat[solver.body].reshape(
                3, 3
            ) @ np.array([0.12, 0, -0.025])
            prediction_s = min(ttc, 0.15) if not self.contact_seen else 0.0
            desired = ball + prediction_s * v[35:38] - 0.11 * rotation[:, 0]
            desired[2] = ball[2] - cfg.foot_height_below_ball_m
            strength = (
                1.0
                if self.contact_seen
                else float(np.clip((cfg.anticipation_s - ttc) / 0.15, 0, 1))
            )
            offset[:3] = (
                np.clip(rotation.T @ (desired - point), -2 * POSE_SCALE[:3], 2 * POSE_SCALE[:3])
                * strength
            )
            offset[4] = -cfg.toe_up_rad * strength
            ep = SimpleNamespace(
                env=SimpleNamespace(model=m, data=d, pelvis_body_id=layout.pelvis),
                layout=layout,
                offset=self.offset,
                scale=self.scale,
                order=np.arange(29),
                frame=w.control_steps,
            )
            _, plan = solver.solve_offset(
                ep,
                nominal,
                offset,
                shin_clearance=cfg.shin_clearance,
                observed_shin_clearance=cfg.shin_clearance,
            )
            requested = np.clip(
                plan["joint_delta_HG"], -cfg.maximum_joint_delta_rad, cfg.maximum_joint_delta_rad
            )
            iterations = plan["ik_iterations"]
        delta = self.previous_delta + np.clip(
            requested - self.previous_delta,
            -cfg.delta_slew_rad_per_control,
            cfg.delta_slew_rad_per_control,
        )
        delta = np.clip(delta, -cfg.maximum_joint_delta_rad, cfg.maximum_joint_delta_rad)
        applied_action = (nominal + delta / self.scale).astype(np.float32)
        applied_target = (applied_action * self.scale + self.offset).astype(np.float64)
        self.previous_delta = delta.copy()
        self.next_control += 1
        return applied_target, dict(
            teacher_active=active,
            teacher_foot=self.selected,
            teacher_ttc_s=ttc,
            teacher_contact_seen=self.contact_seen,
            teacher_delta_HG=delta,
            teacher_pose_offset=offset,
            teacher_ik_iterations=iterations,
            teacher_applied_action=applied_action,
            teacher_is_neural_action=False,
        )
