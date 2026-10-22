"""Bounded Cartesian foot intents over a causal full-body research actor.

IK uses a separate kinematic-only mjData. Actual simulation state, contacts,
model dimensions and PD gains are never edited. This is a composite teacher,
not the actor's own output or a hardware controller.

Jacobian pipeline: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
"""

import re
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

POSE_SCALE = np.array([0.04, 0.035, 0.04, 0.15, 0.2, 0.15])


def phase_intent(frame: int, center: int, latent: np.ndarray) -> np.ndarray:
    if latent.shape != (2, 6) or not np.isfinite(latent).all() or (np.abs(latent) > 1).any():
        raise ValueError("two finite bounded six-dimensional foot intents required")
    t = (frame - np.array([center - 12, center + 8])) / 30.0
    weight = np.where(np.abs(t) < 1, (1 - t * t) ** 3, 0.0)
    return np.asarray((weight[:, None] * latent * POSE_SCALE).sum(axis=0))


def rotation_vector_matrix(vector: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.asarray(np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew))


class FootTaskSpaceIntent:
    def __init__(
        self, model: Any, *, foot: str, center: int, latent: np.ndarray, robot_prefix: str = ""
    ) -> None:
        import mujoco

        if (
            foot not in ("left", "right")
            or not 30 <= center <= 200
            or type(robot_prefix) is not str
            or (robot_prefix and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}/", robot_prefix) is None)
        ):
            raise ValueError("explicit source-planned contact foot/phase required")
        phase_intent(center, center, latent)
        self.model, self.foot, self.center, self.latent = model, foot, center, latent.copy()
        self.robot_prefix = robot_prefix
        self.kin = mujoco.MjData(model)
        self.risk = mujoco.MjData(model)
        self.body = model.body(robot_prefix + foot + "_ankle_roll_link").id
        names = [
            foot + "_" + s + "_joint"
            for s in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        ]
        joints = [model.joint(robot_prefix + n).id for n in names]
        self.qi = np.array([model.jnt_qposadr[j] for j in joints])
        self.vi = np.array([model.jnt_dofadr[j] for j in joints])
        self.hg = np.array([G1_DDS_JOINT_NAMES.index(n) for n in names])

    def __call__(self, ep: Any, nominal: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        return self.solve_offset(ep, nominal, phase_intent(ep.frame, self.center, self.latent))

    def solve_offset(
        self,
        ep: Any,
        nominal: np.ndarray,
        offset: np.ndarray,
        *,
        shin_clearance: bool = False,
        observed_shin_clearance: bool = False,
        clearance_profile: str = "standard-v1",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Bounded current-body Cartesian proposal, no physical state writes.

        Allows a measured contact-centering teacher without inventing a past
        pose or changing the original four-iteration/.2rad IK authority.
        """
        import mujoco

        from rosclaw_soccer.rsi.capsule_clearance import sphere_capsule_clearance
        from rosclaw_soccer.rsi.shin_clearance_profile import shin_clearance_profile

        horizon, constraint_weight = shin_clearance_profile(clearance_profile)
        if clearance_profile != "standard-v1" and not observed_shin_clearance:
            raise ValueError("predictive priority profile requires observed shin clearance")

        if observed_shin_clearance and not shin_clearance:
            raise ValueError("observed clearance requires explicit shin constraint")

        if (
            offset.shape != (6,)
            or not np.isfinite(offset).all()
            or (np.abs(offset) > 2 * POSE_SCALE + 1e-12).any()
        ):
            raise ValueError("finite Cartesian offset inside original phase authority required")
        requested = np.zeros(29)
        if not offset.any() and not shin_clearance:
            return nominal.copy(), dict(
                pose_offset=offset,
                joint_delta_HG=requested,
                position_residual_m=0.0,
                orientation_residual_rad=0.0,
                increment_limited=False,
                ik_iterations=0,
            )
        # Only this scratch data is assigned: NO write to ep.env.data.
        self.kin.qpos[:] = ep.env.data.qpos
        baseline = ep.offset + nominal[ep.order] * ep.scale
        self.kin.qpos[ep.layout.qi] = baseline

        def kinematics() -> tuple[np.ndarray, np.ndarray]:
            mujoco.mj_kinematics(self.model, self.kin)
            mujoco.mj_comPos(self.model, self.kin)
            if observed_shin_clearance:
                self.risk.qpos[:] = ep.env.data.qpos
                # A joint TARGET is not the observed pose. Apply only the
                # proposed correction to actual joints for obstacle prediction.
                self.risk.qpos[ep.layout.qi] += self.kin.qpos[ep.layout.qi] - baseline
                mujoco.mj_kinematics(self.model, self.risk)
                mujoco.mj_comPos(self.model, self.risk)
            matrix = self.kin.xmat[self.body].reshape(3, 3).copy()
            point = self.kin.xpos[self.body] + matrix @ np.array([0.12, 0, -0.025])
            return np.asarray(point), np.asarray(matrix)

        point, matrix = kinematics()
        base = ep.env.data.xmat[ep.env.pelvis_body_id].reshape(3, 3)
        desired_p = point + base @ offset[:3]
        desired_r = rotation_vector_matrix(base @ offset[3:]) @ matrix
        nominal_leg = baseline[self.hg]
        limited = False
        clearance = None
        if shin_clearance:
            shin_geom = self.model.geom(self.robot_prefix + self.foot + "_shin").id
            shin_body = self.model.geom_bodyid[shin_geom]
            ball_geom = self.model.geom("soccer_ball_geom").id
            if (
                self.model.geom_type[shin_geom] != mujoco.mjtGeom.mjGEOM_CAPSULE
                or self.model.geom_type[ball_geom] != mujoco.mjtGeom.mjGEOM_SPHERE
                or abs(self.model.geom_size[ball_geom, 0] - 0.11) > 1e-12
            ):
                raise ValueError("unchanged .11m sphere and compiled shin capsule required")
            actual_ball = ep.env.data.xpos[ep.layout.ball].copy()
            clearance_data = self.risk if observed_shin_clearance else self.kin
            if observed_shin_clearance:
                _, actual_velocity = ep.layout.canonical(ep.env.data)
                ball_velocity = actual_velocity[35:38].copy()

            def separation() -> tuple[float, np.ndarray, np.ndarray]:
                return sphere_capsule_clearance(
                    actual_ball,
                    float(self.model.geom_size[ball_geom, 0]),
                    clearance_data.geom_xpos[shin_geom].copy(),
                    clearance_data.geom_xmat[shin_geom].reshape(3, 3)[:, 2].copy(),
                    float(self.model.geom_size[shin_geom, 1]),
                    float(self.model.geom_size[shin_geom, 0]),
                )

        for _ in range(4):
            point, matrix = kinematics()
            relative = desired_r @ matrix.T
            angular = (
                np.array(
                    [
                        relative[2, 1] - relative[1, 2],
                        relative[0, 2] - relative[2, 0],
                        relative[1, 0] - relative[0, 1],
                    ]
                )
                * 0.5
            )
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.kin, jp, jr, point, self.body)
            jac = np.vstack((jp[:, self.vi], 0.15 * jr[:, self.vi]))
            error = np.r_[desired_p - point, 0.15 * angular]
            if shin_clearance:
                clearance, near, outward = separation()
                sj = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, clearance_data, sj, None, near, shin_body)
                prediction = clearance
                if observed_shin_clearance:
                    closing_speed = float(outward @ (sj @ ep.env.data.qvel - ball_velocity))
                    prediction += horizon * min(0.0, closing_speed)
                if prediction < 0.015:
                    constraint_row = outward @ sj[:, self.vi]
                    constraint_error = 0.015 - prediction
                    if constraint_weight != 1.0:
                        # Weighted least squares: multiply BOTH sides by sqrt(w).
                        constraint_row = constraint_row * np.sqrt(constraint_weight)
                        constraint_error *= np.sqrt(constraint_weight)
                    jac = np.vstack((jac, constraint_row))
                    error = np.r_[error, constraint_error]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(len(error)), error)
            step = np.clip(delta, -0.08, 0.08)
            limited |= bool((step != delta).any())
            leg = self.kin.qpos[self.qi] + step
            constrained = nominal_leg + np.clip(leg - nominal_leg, -0.2, 0.2)
            limited |= bool((constrained != leg).any())
            self.kin.qpos[self.qi] = constrained
        point, matrix = kinematics()
        requested[self.hg] = self.kin.qpos[self.qi] - nominal_leg
        action = nominal.copy()
        action[ep.order] = (action[ep.order] + requested / ep.scale).astype(np.float32)
        relative = desired_r @ matrix.T
        residual_angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        information = dict(
            pose_offset=offset,
            joint_delta_HG=requested,
            position_residual_m=float(np.linalg.norm(desired_p - point)),
            orientation_residual_rad=residual_angle,
            increment_limited=limited,
            ik_iterations=4,
        )
        if shin_clearance:
            information.update(
                shin_clearance_after_scratch_m=separation()[0],
                requested_shin_surface_margin_m=0.015,
                obstacle_center_actual_world=actual_ball.copy(),
            )
            if observed_shin_clearance:
                information.update(
                    clearance_from_observed_pose_plus_correction=True,
                    closing_velocity_horizon_s=horizon,
                )
                if clearance_profile != "standard-v1":
                    information.update(
                        clearance_profile=clearance_profile,
                        clearance_constraint_weight=constraint_weight,
                    )
        return action, information
