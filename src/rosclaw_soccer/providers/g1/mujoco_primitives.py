"""Minimum G1 MuJoCo primitives required by the Soccer free-kick runner.

These helpers qualify and adapt an external SIM-only policy.  They expose no
ROS, DDS, vendor SDK, driver, actuator transport, or hardware authority.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import G1AssetQualification
from rosclaw_soccer.sim.contracts import ShotParameters

_MOTION_REL = Path("policy/robonaldo/model/freekick_motion.npz")
_RECOVERY_REGIME_COMMITMENT = (
    "sha256:c97343bd33b38b0e6dd40cbc1e8871164d209e71c07f7e63b70ec60e07fefc8a"
)


@dataclass(frozen=True)
class ModelIds:
    pelvis: int
    torso: int
    left_ankle: int
    right_ankle: int
    ball: int
    ball_geom: int
    ball_qpos: int
    ball_qvel: int

    @classmethod
    def from_model(cls, model: Any) -> ModelIds:
        import mujoco

        def body(name: str) -> int:
            identifier = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
            if identifier < 0:
                raise ValueError(f"qualified G1 model is missing body {name}")
            return identifier

        ball = body("ball")
        joint = int(model.body_jntadr[ball])
        ball_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"))
        if ball_geom < 0:
            raise ValueError("qualified G1 model is missing ball_geom")
        return cls(
            pelvis=body("pelvis"),
            torso=body("torso_link"),
            left_ankle=body("left_ankle_roll_link"),
            right_ankle=body("right_ankle_roll_link"),
            ball=ball,
            ball_geom=ball_geom,
            ball_qpos=int(model.jnt_qposadr[joint]),
            ball_qvel=int(model.jnt_dofadr[joint]),
        )


@dataclass(frozen=True)
class Contacts:
    left_floor: bool
    right_floor: bool
    ball_any: bool
    ball_left: bool
    ball_right: bool
    ball_force_n: float
    ball_contact_point: tuple[float, float, float]
    ball_contact_normal_xyz: tuple[float, float, float]
    ball_contact_force_world_xyz_n: tuple[float, float, float]
    left_ground_force_n: float
    right_ground_force_n: float


def load_robonaldo(root: Path) -> tuple[Any, Any, Any, np.ndarray]:
    """Load a previously qualified external policy and verify its import root."""

    resolved = root.expanduser().resolve()
    root_text = str(resolved)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        ctrl = importlib.import_module("common.ctrlcomp")
        module = importlib.import_module("policy.robonaldo.FreeKick")
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise RuntimeError("RoboNaldo module does not expose an import path")
        loaded = Path(module_file).resolve()
        if resolved not in loaded.parents:
            raise RuntimeError(f"RoboNaldo module resolved outside qualified root: {loaded}")
        return (
            ctrl.StateAndCmd,
            ctrl.PolicyOutput,
            module.FreeKick,
            np.asarray(module.MUJOCO_TO_ISAAC, dtype=np.int64),
        )
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(root_text)


def fill_policy_state(state: Any, model: Any, data: Any, ids: ModelIds) -> None:
    state.q = data.qpos[7:36].copy()
    state.dq = data.qvel[6:35].copy()
    state.tau_est = data.ctrl.copy()
    state.root_lin_vel_b = data.qvel[0:3].copy()
    state.root_ang_vel_b = data.qvel[3:6].copy()
    state.torso_pos_w = data.xpos[ids.torso].copy()
    state.torso_quat_w = data.xquat[ids.torso].copy()
    state.pelvis_pos_w = data.qpos[0:3].copy()
    state.pelvis_quat_w = data.qpos[3:7].copy()
    state.ball_pos_w = data.xpos[ids.ball].copy()
    state.ball_vel_w = data.qvel[ids.ball_qvel : ids.ball_qvel + 3].copy()
    state.ball_valid = True


def policy_repeat_count(speed_scale: float, policy_frame: int, simulation_frame: int) -> int:
    if not 185 <= policy_frame <= 430:
        return 1
    if speed_scale < 1.0:
        hold_period = max(2, int(round(1.0 / (1.0 - speed_scale))))
        return 0 if simulation_frame % hold_period == 0 else 1
    if speed_scale == 1.0:
        return 1
    extra_period = max(2, int(round(1.0 / (speed_scale - 1.0))))
    return 2 if policy_frame % extra_period == 0 else 1


def adapt_shot_target(
    *,
    target: np.ndarray,
    default: np.ndarray,
    parameters: ShotParameters,
    policy_frame: int,
) -> np.ndarray:
    adapted = target.copy()
    if 185 <= policy_frame <= 335:
        leg = slice(6, 12)
        adapted[leg] = default[leg] + parameters.swing_amplitude * (adapted[leg] - default[leg])
        adapted[1] += parameters.com_shift_y
        adapted[7] += parameters.com_shift_y * 0.5
        adapted[8] += parameters.foot_yaw_offset
        adapted[11] += parameters.foot_yaw_offset * 0.25
        adapted[10] += parameters.foot_pitch_offset
        adapted[6] -= parameters.loft_synergy
        adapted[8] -= parameters.loft_synergy * 0.60
        adapted[9] += parameters.loft_synergy
        adapted[10] -= parameters.loft_synergy * 0.25
        adapted[12] += parameters.pelvis_yaw_offset
    if 335 < policy_frame <= 430:
        adapted[6] -= parameters.recovery_step_length * 0.4
        adapted[8] += parameters.recovery_step_yaw
    return adapted


def contact_observation(model: Any, data: Any, ids: ModelIds) -> Contacts:
    import mujoco

    left_floor = False
    right_floor = False
    ball_any = False
    ball_left = False
    ball_right = False
    ball_force = 0.0
    ball_contact_point = (0.0, 0.0, 0.0)
    ball_contact_normal = (0.0, 0.0, 0.0)
    ball_contact_force_world = (0.0, 0.0, 0.0)
    left_ground_force = 0.0
    right_ground_force = 0.0
    force = np.zeros(6, dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        names = (name1, name2)
        if "floor" in names:
            other = name2 if name1 == "floor" else name1
            left_floor = left_floor or other.startswith("left_foot")
            right_floor = right_floor or other.startswith("right_foot")
            mujoco.mj_contactForce(model, data, index, force)
            contact_force = float(np.linalg.norm(force[:3]))
            if other.startswith("left_foot"):
                left_ground_force = max(left_ground_force, contact_force)
            if other.startswith("right_foot"):
                right_ground_force = max(right_ground_force, contact_force)
        if ids.ball_geom not in {geom1, geom2}:
            continue
        ball_any = True
        other = name2 if geom1 == ids.ball_geom else name1
        ball_left = ball_left or other.startswith("left_foot")
        ball_right = ball_right or other.startswith("right_foot")
        mujoco.mj_contactForce(model, data, index, force)
        contact_force = float(np.linalg.norm(force[:3]))
        if contact_force >= ball_force:
            ball_force = contact_force
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            direction = 1.0 if geom2 == ids.ball_geom else -1.0
            normal = direction * frame[0]
            force_world = direction * (frame.T @ force[:3])
            ball_contact_point = (
                float(contact.pos[0]),
                float(contact.pos[1]),
                float(contact.pos[2]),
            )
            ball_contact_normal = (float(normal[0]), float(normal[1]), float(normal[2]))
            ball_contact_force_world = (
                float(force_world[0]),
                float(force_world[1]),
                float(force_world[2]),
            )
    return Contacts(
        left_floor=left_floor,
        right_floor=right_floor,
        ball_any=ball_any,
        ball_left=ball_left,
        ball_right=ball_right,
        ball_force_n=ball_force,
        ball_contact_point=ball_contact_point,
        ball_contact_normal_xyz=ball_contact_normal,
        ball_contact_force_world_xyz_n=ball_contact_force_world,
        left_ground_force_n=left_ground_force,
        right_ground_force_n=right_ground_force,
    )


def roll_pitch(quaternion_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = map(float, quaternion_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def build_shared_recovery_controller(qualification: G1AssetQualification) -> Any:
    """Bind Core's still-generic G1 recovery controller to Soccer assets.

    This narrow bridge is the remaining embodiment dependency after S2.  It
    deliberately imports no football task, world, scenario, or backend.
    """

    from rosclaw.simforge.g1_cerebellar_recovery import (
        G1CerebellarRecoveryConfig,
        G1CerebellarRecoveryController,
    )

    _, _, _, mujoco_to_isaac = load_robonaldo(qualification.asset_root)
    with np.load(qualification.asset_root / _MOTION_REL, allow_pickle=False) as motion:
        standing_pose = np.asarray(motion["joint_pos"][0][mujoco_to_isaac], dtype=np.float64)
    return G1CerebellarRecoveryController(
        body_hash=qualification.body_hash,
        motion_hash=qualification.motion_hash,
        regime_commitment=_RECOVERY_REGIME_COMMITMENT,
        regime_eligible=True,
        regime_reasons=(),
        standing_pose=standing_pose,
        # Bind the frozen shared post-impact contract explicitly.  Older Core
        # revisions exposed this as ``shared_post_impact_recovery_config``;
        # newer revisions keep only the generic config/controller API.  The
        # explicit values preserve trajectory compatibility across both APIs.
        config=G1CerebellarRecoveryConfig(
            start_policy_frame=280,
            blend_frames=80,
            standing_pose_blend=0.02,
            roll_posture_bias_rad=0.0,
            target_smoothing_alpha=0.60,
            target_smoothing_start_policy_frame=280,
            target_smoothing_joint_group="upper_body",
        ),
    )


__all__ = [
    "Contacts",
    "ModelIds",
    "adapt_shot_target",
    "build_shared_recovery_controller",
    "contact_observation",
    "fill_policy_state",
    "load_robonaldo",
    "policy_repeat_count",
    "roll_pitch",
]
