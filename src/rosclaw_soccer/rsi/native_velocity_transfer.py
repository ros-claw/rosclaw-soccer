"""Read-only SIM layout adapter for a qualified canonical velocity actor.

The new training world stores the ball before the robot and groups actuators.
The old actor still receives robot-first canonical states and outputs HG29.
No simulator state, gains, model properties or hardware are changed here.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


class NativeVelocityLayout:
    def __init__(
        self, model: Any, *, robot_prefix: str, ball_body_name: str, player_count: int = 1
    ) -> None:
        import mujoco

        if (
            type(player_count) is not int
            or not 1 <= player_count <= 8
            or (model.nq, model.nv, model.nu)
            != (36 * player_count + 7, 35 * player_count + 6, 29 * player_count)
            or model.na
            or model.neq
        ):
            raise ValueError("explicit activation-free canonical-sized SIM model required")
        self.model = model
        self.robot_prefix = robot_prefix
        self.pelvis = int(model.body(robot_prefix + "pelvis").id)
        self.ball = int(model.body(ball_body_name).id)
        rj = int(model.body_jntadr[self.pelvis])
        bj = int(model.body_jntadr[self.ball])
        if (
            model.jnt_type[rj] != int(mujoco.mjtJoint.mjJNT_FREE)
            or model.jnt_type[bj] != int(mujoco.mjtJoint.mjJNT_FREE)
            or not np.array_equal(model.body_quat[self.pelvis], [1.0, 0, 0, 0])
        ):
            raise ValueError("actual free robot/ball and identity static root frame required")
        self.rq = int(model.jnt_qposadr[rj])
        self.rv = int(model.jnt_dofadr[rj])
        self.bq = int(model.jnt_qposadr[bj])
        self.bv = int(model.jnt_dofadr[bj])
        self.joints = [int(model.joint(robot_prefix + name).id) for name in G1_DDS_JOINT_NAMES]
        if any(model.jnt_type[j] != int(mujoco.mjtJoint.mjJNT_HINGE) for j in self.joints):
            raise ValueError("29 explicit hinge joints required")
        self.qi = np.array([int(model.jnt_qposadr[j]) for j in self.joints])
        self.vi = np.array([int(model.jnt_dofadr[j]) for j in self.joints])
        qids = [*range(self.rq, self.rq + 7), *self.qi, *range(self.bq, self.bq + 7)]
        vids = [*range(self.rv, self.rv + 6), *self.vi, *range(self.bv, self.bv + 6)]
        if (
            len(set(qids)) != 43
            or len(set(vids)) != 41
            or min(qids) < 0
            or max(qids) >= model.nq
            or min(vids) < 0
            or max(vids) >= model.nv
        ):
            raise ValueError("one-to-one complete canonical state mapping required")
        self.ai = []
        for j in self.joints:
            found = np.flatnonzero(
                (model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
                & (model.actuator_trnid[:, 0] == j)
            )
            if found.shape != (1,):
                raise ValueError("one actuator per canonical joint required")
            self.ai.append(int(found[0]))
        if len(set(self.ai)) != 29:
            raise ValueError("bijection to all actual native actuators required")
        # In a shared model, prefix spelling alone must not assign another
        # player's joints to this body. The shared-world owner checks complete
        # and disjoint coverage separately; legacy one-body coverage is exact.
        for joint in self.joints:
            body = int(model.jnt_bodyid[joint])
            while body not in (0, self.pelvis):
                body = int(model.body_parentid[body])
            if body != self.pelvis:
                raise ValueError("all canonical joints must belong to this player's body")

    def canonical(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        q = np.concatenate(
            (data.qpos[self.rq : self.rq + 7], data.qpos[self.qi], data.qpos[self.bq : self.bq + 7])
        )
        v = np.concatenate(
            (data.qvel[self.rv : self.rv + 6], data.qvel[self.vi], data.qvel[self.bv : self.bv + 6])
        )
        if (
            not np.isfinite(q).all()
            or not np.isfinite(v).all()
            or abs(np.linalg.norm(q[3:7]) - 1) > 1e-5
        ):
            raise ValueError("finite measured canonical state and unit orientation required")
        return q.astype(np.float64), v.astype(np.float64)

    def angular_velocity_body(self, data: Any) -> np.ndarray:
        import mujoco

        # Caller must forward actual data before observation; the independent
        # Jacobian check binds free-joint velocity semantics to the root frame.
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, data, None, jac, self.pelvis)
        omega = data.xmat[self.pelvis].reshape(3, 3).T @ (jac @ data.qvel)
        _, v = self.canonical(data)
        if not np.allclose(omega, v[3:6], rtol=0, atol=1e-10):
            raise ValueError("actual native angular velocity and canonical body frame differ")
        return np.asarray(omega, dtype=np.float64).copy()

    def profile(self) -> dict[str, Any]:
        m = self.model
        bodies = []
        for body in range(m.nbody):
            parent = body
            while parent not in (0, self.pelvis):
                parent = int(m.body_parentid[parent])
            if parent == self.pelvis:
                bodies.append(body)
        return dict(
            nq=m.nq,
            nv=m.nv,
            nu=m.nu,
            robot_qposadr=self.rq,
            robot_dofadr=self.rv,
            ball_qposadr=self.bq,
            ball_dofadr=self.bv,
            actuator_ids_hg=self.ai,
            robot_mass_kg=float(m.body_subtreemass[self.pelvis]),
            body_names=[m.body(b).name for b in bodies],
            body_mass=m.body_mass[bodies].tolist(),
            body_inertia=m.body_inertia[bodies].tolist(),
            body_pos=m.body_pos[bodies].tolist(),
            body_quat=m.body_quat[bodies].tolist(),
            joint_pos=m.jnt_pos[self.joints].tolist(),
            joint_axis=m.jnt_axis[self.joints].tolist(),
            joint_range=m.jnt_range[self.joints].tolist(),
            armature=m.dof_armature[self.vi].tolist(),
            damping=m.dof_damping[self.vi].tolist(),
            frictionloss=m.dof_frictionloss[self.vi].tolist(),
            actuator_gainprm_hg=m.actuator_gainprm[self.ai].tolist(),
            actuator_biasprm_hg=m.actuator_biasprm[self.ai].tolist(),
            actuator_forcerange_hg=m.actuator_forcerange[self.ai].tolist(),
            collision_geoms=[
                dict(
                    body=m.body(int(m.geom_bodyid[g])).name,
                    type=int(m.geom_type[g]),
                    pos=m.geom_pos[g].tolist(),
                    size=m.geom_size[g].tolist(),
                    friction=m.geom_friction[g].tolist(),
                    condim=int(m.geom_condim[g]),
                    contype=int(m.geom_contype[g]),
                    conaffinity=int(m.geom_conaffinity[g]),
                )
                for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])
            ],
            simulator_only=True,
            hardware_authorized=False,
        )
