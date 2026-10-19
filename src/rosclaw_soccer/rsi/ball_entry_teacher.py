"""SIM research teacher: measured ball -> historical skill entry affordance.

No new neural learning is claimed. A frozen prior from PAST own-body motion is
not a future observation. It is an uncertain geometry estimate, not a contact
certificate; different entry histories can change the resulting kick motion.
"""

import re
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class SkillEntryAffordance:
    ball_offset_xy_m: tuple[float, float]
    historical_frame: int
    historical_forward_speed_mps: float
    historical_artifact_hash: str

    def __post_init__(self) -> None:
        if (
            type(self.ball_offset_xy_m) is not tuple
            or len(self.ball_offset_xy_m) != 2
            or not np.isfinite(self.ball_offset_xy_m).all()
            or max(abs(x) for x in self.ball_offset_xy_m) > 1
            or type(self.historical_frame) is not int
            or not 50 <= self.historical_frame < 250
            or not np.isfinite(self.historical_forward_speed_mps)
            or self.historical_forward_speed_mps < 0.3
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.historical_artifact_hash) is None
        ):
            raise ValueError("bounded provenance-bound past-motion affordance required")

    @property
    def contract_hash(self) -> str:
        return hash_json(
            dict(
                schema="soccer.rsi.recorded_skill_entry_affordance.v1",
                **self.__dict__,
                prior_kind="historical_front_support_midpoint_sphere_contact_estimate",
                contact_qualified=False,
                neural_training_updates=0,
                activation_ceiling="SIM_ONLY",
                hardware_authorized=False,
            )
        )


def derive_entry_affordance(
    model: Any,
    canonical_qpos: np.ndarray,
    *,
    artifact_hash: str,
    foot: str,
) -> SkillEntryAffordance:
    """Read-only FK of a complete historical50Hz own-body record, zero mj_step.

    Uses the front pair of original5mm support spheres as an approximate toe
    contact landmark. This does NOT certify the mesh surface or predict impact
    forces. Source frames50..249 exclude entry padding and terminal holding.
    Ground ball center is the existing floor plus its unmodified sphere radius.
    """
    q = canonical_qpos
    if (
        type(q) is not np.ndarray
        or q.shape != (300, 43)
        or q.dtype != np.float64
        or not np.isfinite(q).all()
        or foot not in ("left", "right")
        or (model.nq, model.nv, model.nu) != (43, 41, 29)
    ):
        raise ValueError("complete actual canonical G1 H300 historical pose record required")
    body = int(model.body(foot + "_ankle_roll_link").id)
    spheres = np.flatnonzero(
        (model.geom_bodyid == body)
        & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
        & (model.geom_contype != 0)
    )
    floor = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if (
        len(spheres) != 4
        or not np.all(model.geom_size[spheres, 0] == 0.005)
        or len(floor) != 1
        or model.geom_bodyid[floor[0]] != 0
        or not np.array_equal(model.geom_quat[floor[0]], [1, 0, 0, 0])
    ):
        raise ValueError("original toe support landmarks and single unrotated floor required")
    toes = spheres[model.geom_pos[spheres, 0] == model.geom_pos[spheres, 0].max()]
    if len(toes) != 2:
        raise ValueError("two original front support spheres required")
    ball = model.geom("ball_geom")
    if int(model.geom_type[ball.id]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
        raise ValueError("unchanged sphere ball required")
    ball_radius = float(ball.size[0])
    center_z = float(model.geom_pos[floor[0], 2]) + ball_radius
    reach_radius = ball_radius + 0.005
    qi = model.jnt_qposadr[[model.joint(n).id for n in G1_DDS_JOINT_NAMES]]
    data = mujoco.MjData(model)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, q[0, 3:7])
    points = []
    for row in q:
        data.qpos[:7] = row[:7]
        data.qpos[qi] = row[7:36]
        mujoco.mj_kinematics(model, data)
        points.append(data.geom_xpos[toes].mean(axis=0))
    world = np.asarray(points)
    local = (world - q[0, :3]) @ rotation.reshape(3, 3)
    speed = np.diff(local[:, 0]) * 50
    dz = world[:-1, 2] - center_z
    valid = (np.arange(299) >= 50) & (np.arange(299) < 250) & (abs(dz) < reach_radius)
    index = int(np.argmax(np.where(valid, speed, -np.inf)))
    if not valid[index] or speed[index] < 0.3:
        raise ValueError("historical motion has no qualified forward ground-ball landmark cue")
    offset = local[index, :2].copy()
    offset[0] += np.sqrt(reach_radius**2 - dz[index] ** 2)
    return SkillEntryAffordance(
        (float(offset[0]), float(offset[1])), index, float(speed[index]), artifact_hash
    )


@dataclass(frozen=True)
class BallEntryDecision:
    command: tuple[float, float, float]
    target_root_xy_m: tuple[float, float]
    horizontal_error_m: float
    ready: bool
    timed_out: bool


def ball_entry_decision(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    frame: int,
    affordance: SkillEntryAffordance,
) -> BallEntryDecision:
    """Current measured ball/root state only, bounded teacher world command.

    Shot heading is explicitly world+X for this pilot. This entry teacher is
    not an adaptive contact controller after handoff. No standstill is required;
    actual moving-entry success must be established by subsequent physics.
    """
    if (
        qpos.shape != (43,)
        or qvel.shape != (41,)
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or type(frame) is not int
        or not 0 <= frame < 300
        or abs(np.linalg.norm(qpos[3:7]) - 1) > 1e-4
    ):
        raise ValueError("finite measured native body/ball observation and current frame required")
    affordance.__post_init__()
    target = qpos[36:38] + 0.1 * qvel[35:37] - np.asarray(affordance.ball_offset_xy_m)
    error = target - qpos[:2]
    distance = float(np.linalg.norm(error))
    velocity = 1.5 * error
    velocity *= min(1.0, 0.6 / max(float(np.linalg.norm(velocity)), 1e-12))
    w, x, y, z = qpos[3:7]
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    upright = 1 - 2 * (x * x + y * y)
    timed_out = frame > 90
    ready = bool(
        30 <= frame <= 90
        and distance <= 0.04
        and abs(yaw) <= 0.1
        and qpos[2] >= 0.65
        and upright >= 0.9
    )
    command = (
        (0.0, 0.0, 0.0)
        if timed_out
        else (
            float(velocity[0]),
            float(velocity[1]),
            float(np.clip(-2 * yaw, -1, 1)),
        )
    )
    return BallEntryDecision(
        command, (float(target[0]), float(target[1])), distance, ready, timed_out
    )
