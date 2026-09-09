"""Read-only shared-pitch projection into the historical single-keeper frame.

The canonical keeper faces -X at (4.52, 0). This is a rigid rotation, never a
sagittal reflection: joint identities and free-joint LOCAL angular velocities
are unchanged. No simulator state, controller target or torque is written.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


@dataclass(frozen=True)
class KeeperSnapshot:
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    time: float

    def __post_init__(self) -> None:
        for name, size in (("qpos", 43), ("qvel", 41)):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (size,) or not np.all(np.isfinite(array)):
                raise ValueError("keeper snapshot has invalid finite state")
            object.__setattr__(self, name, np.frombuffer(array.tobytes(), dtype=np.float64))
        if not math.isfinite(self.time) or self.time < 0:
            raise ValueError("keeper snapshot time is invalid")


@dataclass(frozen=True)
class KeeperFrame:
    origin_xy: tuple[float, float]
    yaw_rad: float

    def __post_init__(self) -> None:
        if len(self.origin_xy) != 2 or not all(
            math.isfinite(v) for v in (*self.origin_xy, self.yaw_rad)
        ):
            raise ValueError("keeper frame requires a finite anchor and yaw")
        object.__setattr__(self, "origin_xy", tuple(float(v) for v in self.origin_xy))

    @property
    def rotation(self) -> NDArray[np.float64]:
        angle = math.pi - self.yaw_rad
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)

    def point(self, world: NDArray[np.float64]) -> NDArray[np.float64]:
        if world.shape != (3,) or not np.all(np.isfinite(world)):
            raise ValueError("keeper frame point must be finite XYZ")
        return np.asarray(
            self.rotation @ (world - np.asarray((*self.origin_xy, 0.0))) + (4.52, 0, 0),
            dtype=np.float64,
        )

    def world_vector(self, canonical: NDArray[np.float64]) -> NDArray[np.float64]:
        if canonical.shape != (3,) or not np.all(np.isfinite(canonical)):
            raise ValueError("keeper frame vector must be finite XYZ")
        return self.rotation.T @ canonical

    def _pose(self, pose: NDArray[np.float64]) -> NDArray[np.float64]:
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("keeper pose must be finite")
        w, x, y, z = pose[3:]
        if not math.isclose(float(np.linalg.norm(pose[3:])), 1.0, abs_tol=1e-6):
            raise ValueError("keeper projection requires a unit quaternion")
        angle = (math.pi - self.yaw_rad) / 2
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray(
            (*self.point(pose[:3]), c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w),
            dtype=np.float64,
        )

    def project(self, model: Any, data: Any, *, prefix: str) -> KeeperSnapshot:
        if prefix and re.fullmatch(r"[a-z][a-z0-9_]{0,62}_", prefix) is None:
            raise ValueError("invalid keeper body prefix")

        def addresses(name: str, kind: int) -> tuple[int, int]:
            try:
                joint = model.joint(name)
            except KeyError as error:
                raise ValueError("keeper body is missing a required joint") from error
            if int(model.jnt_type[joint.id]) != kind:
                raise ValueError("keeper joint type differs from its state contract")
            return int(model.jnt_qposadr[joint.id]), int(model.jnt_dofadr[joint.id])

        # MuJoCo enum: free=0, hinge=3. Resolve names, never assume shared offsets.
        root_q, root_v = addresses(prefix + "floating_base_joint", 0)
        ball_q, ball_v = addresses("ball_free", 0)
        joints = [addresses(prefix + name, 3) for name in G1_DDS_JOINT_NAMES]
        qpos = np.r_[
            self._pose(np.asarray(data.qpos[root_q : root_q + 7])),
            [data.qpos[q] for q, _ in joints],
            self._pose(np.asarray(data.qpos[ball_q : ball_q + 7])),
        ]
        qvel = np.r_[
            self.rotation @ data.qvel[root_v : root_v + 3],
            data.qvel[root_v + 3 : root_v + 6],
            [data.qvel[v] for _, v in joints],
            self.rotation @ data.qvel[ball_v : ball_v + 3],
            data.qvel[ball_v + 3 : ball_v + 6],
        ]
        return KeeperSnapshot(qpos, qvel, float(data.time))
