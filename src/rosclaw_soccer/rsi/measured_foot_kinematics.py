"""Current-state foot/ball geometry on private kinematic data, never a forecast.

No mj_step, dynamics, contact forces, future reference, action, or live MjData
enters this extractor. Velocity vectors are instantaneous world-relative linear
velocities expressed in the current root frame, NOT rotating-coordinate time
derivatives. This is a G1 experiment adapter, not a generic Core body contract.
"""

import math
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes

FOOT_FEATURE_WIDTH = 26


class MeasuredFootKinematics:
    def __init__(self, model: Any, *, expected_world_hash: str) -> None:
        if not isinstance(model, mujoco.MjModel) or (model.nq, model.nv, model.nu) != (43, 41, 29):
            raise ValueError("canonical G1 simulation body geometry required")
        binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, binary)
        if hash_bytes(binary.tobytes()) != expected_world_hash:
            raise ValueError("explicit compiled body geometry commitment required")
        self.model = model
        self.private = mujoco.MjData(model)
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        self.qi = [int(model.jnt_qposadr[j]) for j in joints]
        self.vi = [int(model.jnt_dofadr[j]) for j in joints]
        self.ball = int(model.body("ball").id)
        self.bg = int(model.geom("ball_geom").id)
        bj = int(model.body_jntadr[self.ball])
        self.bq, self.bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
        if (
            set(self.qi) != set(range(7, 36))
            or set(self.vi) != set(range(6, 35))
            or self.bq != 36
            or self.bv != 35
            or int(model.geom_type[self.bg]) != int(mujoco.mjtGeom.mjGEOM_SPHERE)
        ):
            raise ValueError("canonical actuated layout and spherical ball required")
        self.feet, self.geoms = [], []
        for side in ("left", "right"):
            body = int(model.body(side + "_ankle_roll_link").id)
            pitch = int(model.body(side + "_ankle_pitch_link").id)
            geoms = [
                g
                for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) in (body, pitch)
                and (
                    (int(model.geom_contype[g]) & int(model.geom_conaffinity[self.bg]))
                    or (int(model.geom_contype[self.bg]) & int(model.geom_conaffinity[g]))
                )
            ]
            if not geoms:
                raise ValueError("solid foot/ball collision geometry required")
            self.feet.append(body)
            self.geoms.append(geoms)

    def features(
        self, qpos: np.ndarray[Any, Any], qvel: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        for x, shape in ((qpos, (43,)), (qvel, (41,))):
            if (
                type(x) is not np.ndarray
                or x.shape != shape
                or x.dtype != np.float64
                or not np.isfinite(x).all()
            ):
                raise ValueError("owned finite float64 current measured canonical state required")
        if abs(np.linalg.norm(qpos[3:7]) - 1) > 1e-6:
            raise ValueError("unit current root orientation required")
        d = self.private
        d.qpos[:7], d.qpos[self.qi], d.qpos[self.bq : self.bq + 7] = qpos[:7], qpos[7:36], qpos[36:]
        d.qvel[:6], d.qvel[self.vi], d.qvel[self.bv : self.bv + 6] = qvel[:6], qvel[6:35], qvel[35:]
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        w, x, y, z = qpos[3:7]
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        rows: list[np.ndarray[Any, Any]] = []
        for foot, geoms in zip(self.feet, self.geoms, strict=True):
            jac = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, d, jac, None, foot)
            p, v = d.xpos[foot].copy(), jac @ d.qvel
            gap = min(
                float(mujoco.mj_geomDistance(self.model, d, self.bg, g, 3.0, None)) for g in geoms
            )
            if not math.isfinite(gap):
                raise ValueError("nonfinite current geometric distance")
            rows.extend(
                (
                    rotation.T @ (p - qpos[:3]),
                    rotation.T @ (v - qvel[:3]),
                    rotation.T @ (qpos[36:39] - p),
                    rotation.T @ (qvel[35:38] - v),
                    np.asarray([max(0.0, gap)]),
                )
            )
        result = np.concatenate(rows).astype(np.float32)
        if result.shape != (FOOT_FEATURE_WIDTH,) or not np.isfinite(result).all():
            raise ValueError("finite current26 foot/ball kinematics required")
        return result

    def from_student_features(self, feature: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Reconstruct only kinematically relevant state from existing108 inputs.

        RootXY is gauge-fixed to zero; spherical ball orientation/spin is not
        observed by this input and is not used for these geometric features.
        Rounded measured root quaternion is normalized for FK, not physical state.
        """
        if (
            type(feature) is not np.ndarray
            or feature.shape != (108,)
            or feature.dtype != np.float32
            or not np.isfinite(feature).all()
        ):
            raise ValueError("one finite current float32 student108 vector required")
        f = feature.astype(np.float64)
        norm = np.linalg.norm(f[2:6])
        if abs(norm - 1) > 1e-4:
            raise ValueError("rounded unit measured root quaternion required")
        q, v = np.zeros(43), np.zeros(41)
        q[2], q[3:7], q[7:36] = f[1], f[2:6] / norm, f[12:41]
        v[:6], v[6:35] = f[6:12], f[41:70]
        w, x, y, z = q[3:7]
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        q[36:39], q[39] = q[:3] + rotation @ f[70:73], 1.0
        v[35:38] = v[:3] + rotation @ f[73:76]
        return self.features(q, v)
