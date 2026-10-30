"""Construct a SIM-only incoming-ball bank at episode boundaries.

The source robot states and motion references remain kick-derived. This is an
experimental reset bank, not an independently qualified receiving curriculum.
"""

from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.velocity_policy import target29
from rosclaw_soccer.rsi.native_reference_observation import native_reference_observation177
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse, receiving_ball_launch


def heading_launch(
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    root: tuple[float, float, float],
    quaternion: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Rotate a declared launch by the measured initial pelvis heading only."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-5:
        raise ValueError("finite unit pelvis quaternion required")
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, q)
    forward = matrix.reshape(3, 3)[:2, 0]
    length = float(np.linalg.norm(forward))
    if length < 1e-6:
        raise ValueError("nonvertical initial pelvis heading required")
    c, s = forward / length
    rotation = np.array([[c, -s], [s, c]])
    offset = rotation @ (np.asarray(position[:2]) - np.asarray(root[:2]))
    speed = rotation @ np.asarray(velocity[:2])
    return (
        (float(root[0] + offset[0]), float(root[1] + offset[1]), position[2]),
        (float(speed[0]), float(speed[1]), velocity[2]),
    )


def scale_launch_distance(position, root, distance_m):
    """Change only declared longitudinal reset distance, never live physics."""
    if distance_m not in (0.9, 1.8):
        raise ValueError("predeclared launch distance required")
    return (root[0] + (position[0] - root[0]) * distance_m / 0.9, position[1], position[2])


def heading_front_launch(position, velocity, root, quaternion, *, team_sign):
    """Canonicalize the team-side launch before applying body heading.

    The existing heading mode deliberately stays unchanged for compatibility;
    this explicit new course puts both team sides in front of their own body.
    This is initialization-only research, never a live ball correction.
    """
    if team_sign not in (-1.0, 1.0):
        raise ValueError("explicit team-side sign required")
    canonical_position = (
        root[0] + team_sign * (position[0] - root[0]),
        root[1] + team_sign * (position[1] - root[1]),
        position[2],
    )
    canonical_velocity = (team_sign * velocity[0], team_sign * velocity[1], velocity[2])
    return heading_launch(canonical_position, canonical_velocity, root, quaternion)


def build_incoming_bank(
    *,
    model: Any,
    params: Any,
    bank: dict[str, NDArray[Any]],
    plans: list[dict[str, NDArray[Any]]],
    courses: tuple[ReceivingCourse, ...],
    launch_frame: str = "world",
    launch_distance_m: float = 0.9,
) -> tuple[dict[str, NDArray[Any]], str]:
    """Build all 256 resets after preflight; never mutate a live simulation.

    The legacy world frame remains default. The explicit heading mode rotates
    the same declared distribution into each initial pelvis heading; it defines
    a different course bank, not the same initial physical conditions.
    A no-slip initial ball spin is set at reset; ground friction
    subsequently determines its actual motion. Reject the entire bank when
    any launched ball initially intersects a robot collider.
    """
    if (
        len(courses) != 256
        or len(plans) != 8
        or launch_frame not in ("world", "heading", "heading_front")
        or launch_distance_m not in (0.9, 1.8)
    ):
        raise ValueError("entire 256-state source and course bank required")
    binding = preflight_receiving_courses(courses)
    if launch_frame == "heading":
        binding = str(hash_json(dict(schema="soccer.heading_receiving.v1", base=binding)))
    if launch_frame == "heading_front":
        binding = str(hash_json(dict(schema="soccer.heading_front_receiving.v1", base=binding)))
    if launch_distance_m != 0.9:
        binding = str(
            hash_json(
                dict(
                    schema="soccer.receiving_launch_distance.v1",
                    base=binding,
                    distance_m=launch_distance_m,
                )
            )
        )
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    geometry = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == layout.ball]
    if len(geometry) != 1 or model.geom_type[geometry[0]] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise ValueError("one explicit spherical ball geometry required")
    geom = geometry[0]
    radius = float(model.geom_size[geom, 0])
    if not 0.03 <= radius <= 0.15 or not np.array_equal(model.geom_pos[geom], np.zeros(3)):
        raise ValueError("bounded centered native ball required")
    result = {key: value.copy() for key, value in bank.items()}
    required = {
        "qpos": (256, model.nq),
        "qvel": (256, model.nv),
        "qacc_warmstart": (256, model.nv),
        "observation": (256, 177),
    }
    for key, shape in required.items():
        if result[key].shape != shape or not np.isfinite(result[key]).all():
            raise ValueError("complete finite native reset arrays required")
    torso = int(model.body("robot/torso_link").id)
    for i, course in enumerate(courses):
        data = mujoco.MjData(model)
        data.qpos[:] = result["qpos"][i]
        data.qvel[:] = result["qvel"][i]
        root = tuple(float(x) for x in data.qpos[layout.rq : layout.rq + 3])
        position, velocity = receiving_ball_launch(course, origin=root, radius_m=radius)
        if launch_distance_m != 0.9:
            position = scale_launch_distance(position, root, launch_distance_m)
        if launch_frame == "heading":
            position, velocity = heading_launch(
                position, velocity, root, data.qpos[layout.rq + 3 : layout.rq + 7]
            )
        elif launch_frame == "heading_front":
            position, velocity = heading_front_launch(
                position,
                velocity,
                root,
                data.qpos[layout.rq + 3 : layout.rq + 7],
                team_sign=1.0 if course.agent_id.startswith("red.") else -1.0,
            )
        data.qpos[layout.bq : layout.bq + 7] = (*position, 1, 0, 0, 0)
        data.qvel[layout.bv : layout.bv + 6] = (
            *velocity,
            -velocity[1] / radius,
            velocity[0] / radius,
            0,
        )
        data.ctrl[layout.ai] = target29(result["previous_action"][i], params)
        data.time = float(result["reference_frame"][i]) * model.opt.timestep * 10
        data.qacc_warmstart[:] = 0
        mujoco.mj_forward(model, data)
        for ci in range(data.ncon):
            contact = data.contact[ci]
            bodies = [int(model.geom_bodyid[int(g)]) for g in contact.geom]
            if layout.ball in bodies:
                other = bodies[1] if bodies[0] == layout.ball else bodies[0]
                if (model.body(other).name or "").startswith("robot/") and contact.dist < 0:
                    raise ValueError(f"incoming launch {i} initially intersects robot")
        result["qpos"][i] = data.qpos
        result["qvel"][i] = data.qvel
        result["qacc_warmstart"][i] = 0
        source = int(result["source_id"][i])
        observation, _ = native_reference_observation177(
            data=data,
            layout=layout,
            params=params,
            plan=plans[source],
            frame=int(result["reference_frame"][i]),
            origin=result["origin"][i],
            goal=result["goal_world"][i],
            last_action=result["previous_action"][i],
            phase=np.float32(result["phase_before"][i]),
            torso=torso,
        )
        result["observation"][i] = observation
    return result, binding
