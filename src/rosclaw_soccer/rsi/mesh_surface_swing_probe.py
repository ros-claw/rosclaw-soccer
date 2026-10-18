"""Scratch-kinematics adapter for an explicitly declared G1 SIM experiment.

Not wired into native execution by importing this module. The caller must
preserve old gating/foot selection, authenticate records and replay physics.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.convex_surface_distance import ConvexSurfaceDistance
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.mesh_surface_swing import surface_strike_joint_delta
from rosclaw_soccer.sim.contracts import hash_bytes

TRACE_WIDTH = 39


def inactive_surface_probe_record(frame: int) -> np.ndarray[Any, Any]:
    if type(frame) is not int or not 0 <= frame < 300:
        raise ValueError("explicit finite-horizon control frame required")
    result = np.zeros(TRACE_WIDTH, dtype=np.float64)
    result[:3] = [frame, -1, 0]
    return result


class G1MeshSurfaceSwingProbe:
    """Own scratch FK only; never refresh or mutate the caller's live MjData."""

    def __init__(self, model: Any, *, expected_world_hash: str) -> None:
        import mujoco

        if (model.nq, model.nv, model.nu) != (43, 41, 29):
            raise ValueError("explicit canonical free-root/free-ball G1 simulation required")
        binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, binary)
        if hash_bytes(binary.tobytes()) != expected_world_hash:
            raise ValueError("probe model differs from declared compiled contact world")
        # Optional diagnostics dependency; untyped upstream hull construction
        # stays at this simulation-only boundary, outside the numeric kernel.
        from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

        self._model = model
        self._scratch = mujoco.MjData(model)
        self._sphere = int(model.geom("ball_geom").id)
        if model.geom_type[self._sphere] != mujoco.mjtGeom.mjGEOM_SPHERE:
            raise ValueError("actual spherical ball required")
        self._feet = [int(model.geom(f"task_contact_surface_{s}").id) for s in range(2)]
        self._hulls = []
        for g, side in zip(self._feet, ("left", "right"), strict=True):
            if (
                model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                or model.geom_bodyid[g] != model.body(f"{side}_ankle_roll_link").id
                or not any(
                    {int(model.pair_geom1[i]), int(model.pair_geom2[i])} == {g, self._sphere}
                    for i in range(model.npair)
                )
            ):
                raise ValueError("explicit existing per-foot mesh/ball pair required")
            mesh = int(model.geom_dataid[g])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
            hull = ConvexHull(vertices)
            self._hulls.append(ConvexSurfaceDistance(vertices[hull.simplices], hull.equations))
        self._dofs = [int(model.jnt_dofadr[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES[:12]]

    def advance(
        self,
        data: Any,
        *,
        frame: int,
        side: int,
        baseline_target: np.ndarray[Any, Any],
        joint_limits: np.ndarray[Any, Any],
        forward_cap_m: float,
        lateral_cap_m: float,
        vertical_cap_m: float,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        import mujoco

        record = inactive_surface_probe_record(frame)
        if type(side) is not int or side not in (0, 1) or data is self._scratch:
            raise ValueError("explicit swing foot and distinct caller-owned state required")
        scratch, model = self._scratch, self._model
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = data.qvel
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        geom = self._feet[side]
        center = scratch.geom_xpos[self._sphere].copy()
        rotation = scratch.geom_xmat[geom].reshape(3, 3)
        _, local = self._hulls[side].nearest(rotation.T @ (center - scratch.geom_xpos[geom]))
        point = scratch.geom_xpos[geom] + rotation @ local
        jacobian = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, scratch, jacobian, None, point, int(model.geom_bodyid[geom]))
        leg_jacobian = jacobian[:, self._dofs[6 * side : 6 * side + 6]]
        delta = surface_strike_joint_delta(
            point,
            center,
            leg_jacobian,
            baseline_target,
            joint_limits,
            np.array([1.0, 0.0, 0.0]),
            ball_radius_m=float(model.geom_size[self._sphere, 0]),
            forward_cap_m=forward_cap_m,
            lateral_cap_m=lateral_cap_m,
            vertical_cap_m=vertical_cap_m,
        )
        record[:3] = [frame, side, 1]
        record[3:] = np.concatenate((point, center, leg_jacobian.ravel(), baseline_target, delta))
        return delta, record
