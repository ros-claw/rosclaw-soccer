"""Read-only current base kinematics on private data, not stale solver caches."""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes


class G1VelocityObservation:
    def __init__(self, model: Any, *, expected_world_hash: str) -> None:
        if not isinstance(model, mujoco.MjModel) or (model.nq, model.nv, model.nu) != (43, 41, 29):
            raise ValueError("canonical current G1 simulation geometry required")
        binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, binary)
        if hash_bytes(binary.tobytes()) != expected_world_hash:
            raise ValueError("exact declared world geometry required")
        self.model = model
        self.private = mujoco.MjData(model)
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        self.qi = [int(model.jnt_qposadr[j]) for j in joints]
        self.vi = [int(model.jnt_dofadr[j]) for j in joints]
        self.pelvis = int(model.body("pelvis").id)
        ball = int(model.body("ball").id)
        bj = int(model.body_jntadr[ball])
        self.bq, self.bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
        if (
            set(self.qi) != set(range(7, 36))
            or set(self.vi) != set(range(6, 35))
            or self.bq != 36
            or self.bv != 35
            or not np.array_equal(model.body_quat[self.pelvis], [1.0, 0.0, 0.0, 0.0])
        ):
            raise ValueError("canonical current-state mapping and identity root frame required")

    def angular_velocity_body(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        for a, shape in ((q, (43,)), (v, (41,))):
            if (
                type(a) is not np.ndarray
                or a.shape != shape
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("finite canonical current-state observations required")
        if abs(np.linalg.norm(q[3:7]) - 1) > 1e-5:
            raise ValueError("unit actual root orientation required")
        d = self.private
        d.qpos[:7] = q[:7]
        d.qpos[self.qi] = q[7:36]
        d.qpos[self.bq :] = q[36:]
        d.qvel[:6] = v[:6]
        d.qvel[self.vi] = v[6:35]
        d.qvel[self.bv :] = v[35:]
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        jac = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, d, None, jac, self.pelvis)
        result = d.xmat[self.pelvis].reshape(3, 3).T @ (jac @ d.qvel)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError("finite measured current body angular velocity required")
        return np.asarray(result, dtype=np.float64).copy()


def world_command_to_body(q: np.ndarray, command: tuple[float, float, float]) -> np.ndarray:
    if (
        type(q) is not np.ndarray
        or q.shape != (43,)
        or q.dtype != np.float64
        or not np.isfinite(q).all()
        or abs(np.linalg.norm(q[3:7]) - 1) > 1e-5
        or type(command) is not tuple
        or len(command) != 3
        or any(type(x) is not float for x in command)
        or not np.isfinite(command).all()
    ):
        raise ValueError("finite current root and immutable world command required")
    w, x, y, z = q[3:7]
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [c * command[0] + s * command[1], -s * command[0] + c * command[1], command[2]],
        dtype=np.float32,
    )
