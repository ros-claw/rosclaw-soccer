"""Same-pose G1 task Jacobians on owned scratch state, not control or physics.

Only canonical actuated hinge columns are exposed to the joint-task solver.
The ball is excluded from the robot COM subtree; root/ball motion cannot become
an optimizer shortcut. A point Jacobian differentiates a held material point,
not the potentially nonsmooth closest-point selection map.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.physics.convex_surface_distance import ConvexSurfaceDistance
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes


@dataclass(frozen=True)
class G1JointTaskKinematics:
    joint_position_rad: np.ndarray[Any, Any]
    joint_limits_rad: np.ndarray[Any, Any]
    robot_subtree_com_world_m: np.ndarray[Any, Any]
    com_jacobian_m_per_rad: np.ndarray[Any, Any]
    foot_origin_world_m: np.ndarray[Any, Any]
    foot_origin_jacobian_m_per_rad: np.ndarray[Any, Any]
    foot_rotation_jacobian: np.ndarray[Any, Any]
    foot_material_point_world_m: np.ndarray[Any, Any]
    foot_material_point_mesh_local_m: np.ndarray[Any, Any]
    foot_material_point_jacobian_m_per_rad: np.ndarray[Any, Any]
    foot_ball_signed_gap_m: np.ndarray[Any, Any]
    torso_rotation_jacobian: np.ndarray[Any, Any]
    ball_center_world_m: np.ndarray[Any, Any]


class G1JointTaskKinematicSampler:
    def __init__(self, model: Any, *, expected_world_hash: str) -> None:
        import mujoco

        if not isinstance(model, mujoco.MjModel) or (model.nq, model.nv, model.nu) != (43, 41, 29):
            raise ValueError("native canonical free-root/free-ball G1 model required")
        binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, binary)
        if hash_bytes(binary.tobytes()) != expected_world_hash:
            raise ValueError("exact declared compiled contact world required")
        from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

        self._model = model
        self._scratch = mujoco.MjData(model)
        self._pelvis = int(model.body("pelvis").id)
        self._torso = int(model.body("torso_link").id)
        self._feet = [int(model.body(s + "_ankle_roll_link").id) for s in ("left", "right")]
        self._geoms = [int(model.geom(f"task_contact_surface_{s}").id) for s in range(2)]
        self._sphere = int(model.geom("ball_geom").id)
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("canonical actuated hinges required")
        self._qi = [int(model.jnt_qposadr[j]) for j in joints]
        self._vi = [int(model.jnt_dofadr[j]) for j in joints]
        if set(self._vi) != set(range(6, 35)):
            raise ValueError("root and ball DOFs must be excluded from all optimization variables")
        self._limits = model.jnt_range[joints].copy()
        self._hulls = []
        for g, body in zip(self._geoms, self._feet, strict=True):
            if (
                model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                or int(model.geom_bodyid[g]) != body
                or not any(
                    {int(model.pair_geom1[i]), int(model.pair_geom2[i])} == {g, self._sphere}
                    for i in range(model.npair)
                )
            ):
                raise ValueError("actual existing per-foot mesh/ball contact pair required")
            mesh = int(model.geom_dataid[g])
            start = int(model.mesh_vertadr[mesh])
            count = int(model.mesh_vertnum[mesh])
            vertices = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
            hull = ConvexHull(vertices)
            self._hulls.append(ConvexSurfaceDistance(vertices[hull.simplices], hull.equations))
        if model.geom_type[self._sphere] != mujoco.mjtGeom.mjGEOM_SPHERE:
            raise ValueError("actual spherical task ball required")

    def sample(self, data: Any) -> G1JointTaskKinematics:
        import mujoco

        model, scratch = self._model, self._scratch
        if (
            not isinstance(data, mujoco.MjData)
            or data.model is not model
            or data is scratch
            or not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
        ):
            raise ValueError("finite caller-owned state from the bound model required")
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = data.qvel
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        center = scratch.geom_xpos[self._sphere].copy()
        com = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, scratch, com, self._pelvis)
        origins, origin_jacs, rot_jacs, points, locals_, point_jacs, gaps = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for g, body, hull in zip(self._geoms, self._feet, self._hulls, strict=True):
            rotation = scratch.geom_xmat[g].reshape(3, 3)
            signed, local = hull.nearest(rotation.T @ (center - scratch.geom_xpos[g]))
            point = scratch.geom_xpos[g] + rotation @ local
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, scratch, jp, jr, body)
            origin_jacs.append(jp[:, self._vi].copy())
            rot_jacs.append(jr[:, self._vi].copy())
            origins.append(scratch.xpos[body].copy())
            points.append(point)
            locals_.append(local.copy())
            mujoco.mj_jac(model, scratch, jp, None, point, body)
            point_jacs.append(jp[:, self._vi].copy())
            gaps.append(signed - float(model.geom_size[self._sphere, 0]))
        torso = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, scratch, None, torso, self._torso)
        result = G1JointTaskKinematics(
            scratch.qpos[self._qi].copy(),
            self._limits.copy(),
            scratch.subtree_com[self._pelvis].copy(),
            com[:, self._vi].copy(),
            np.asarray(origins),
            np.asarray(origin_jacs),
            np.asarray(rot_jacs),
            np.asarray(points),
            np.asarray(locals_),
            np.asarray(point_jacs),
            np.asarray(gaps),
            torso[:, self._vi].copy(),
            center,
        )
        if any(not np.isfinite(value).all() for value in vars(result).values()):
            raise ValueError("finite current actuated-only task measurements required")
        return result
