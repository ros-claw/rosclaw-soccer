"""Bounded Cartesian foot intents over a causal full-body research actor.

IK uses a separate kinematic-only mjData. Actual simulation state, contacts,
model dimensions and PD gains are never edited. This is a composite teacher,
not the actor's own output or a hardware controller.

Jacobian pipeline: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
"""

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
    def __init__(self, model: Any, *, foot: str, center: int, latent: np.ndarray) -> None:
        import mujoco

        if foot not in ("left", "right") or not 30 <= center <= 200:
            raise ValueError("explicit source-planned contact foot/phase required")
        phase_intent(center, center, latent)
        self.model, self.foot, self.center, self.latent = model, foot, center, latent.copy()
        self.kin = mujoco.MjData(model)
        self.body = model.body(foot + "_ankle_roll_link").id
        names = [
            foot + "_" + s + "_joint"
            for s in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        ]
        joints = [model.joint(n).id for n in names]
        self.qi = np.array([model.jnt_qposadr[j] for j in joints])
        self.vi = np.array([model.jnt_dofadr[j] for j in joints])
        self.hg = np.array([G1_DDS_JOINT_NAMES.index(n) for n in names])

    def __call__(self, ep: Any, nominal: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        return self.solve_offset(ep, nominal, phase_intent(ep.frame, self.center, self.latent))

    def solve_offset(
        self, ep: Any, nominal: np.ndarray, offset: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Bounded current-body Cartesian proposal, no physical state writes.

        Allows a measured contact-centering teacher without inventing a past
        pose or changing the original four-iteration/.2rad IK authority.
        """
        import mujoco

        if (
            offset.shape != (6,)
            or not np.isfinite(offset).all()
            or (np.abs(offset) > 2 * POSE_SCALE + 1e-12).any()
        ):
            raise ValueError("finite Cartesian offset inside original phase authority required")
        requested = np.zeros(29)
        if not offset.any():
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
            matrix = self.kin.xmat[self.body].reshape(3, 3).copy()
            point = self.kin.xpos[self.body] + matrix @ np.array([0.12, 0, -0.025])
            return np.asarray(point), np.asarray(matrix)

        point, matrix = kinematics()
        base = ep.env.data.xmat[ep.env.pelvis_body_id].reshape(3, 3)
        desired_p = point + base @ offset[:3]
        desired_r = rotation_vector_matrix(base @ offset[3:]) @ matrix
        nominal_leg = baseline[self.hg]
        limited = False
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
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(6), error)
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
        return action, dict(
            pose_offset=offset,
            joint_delta_HG=requested,
            position_residual_m=float(np.linalg.norm(desired_p - point)),
            orientation_residual_rad=residual_angle,
            increment_limited=limited,
            ik_iterations=4,
        )
