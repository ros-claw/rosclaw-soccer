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
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse, receiving_ball_launch


def build_incoming_bank(
    *,
    model: Any,
    params: Any,
    bank: dict[str, NDArray[Any]],
    plans: list[dict[str, NDArray[Any]]],
    courses: tuple[ReceivingCourse, ...],
) -> tuple[dict[str, NDArray[Any]], str]:
    """Build all 256 resets after preflight; never mutate a live simulation.

    The legacy classroom launch is world-axis aligned, 0.9m from the measured
    pelvis projection. Positive x is not assumed to be the robot's forward
    direction. A no-slip initial ball spin is set at reset; ground friction
    subsequently determines its actual motion. Reject the entire bank when
    any launched ball initially intersects a robot collider.
    """
    if len(courses) != 256 or len(plans) != 8:
        raise ValueError("entire 256-state source and course bank required")
    binding = preflight_receiving_courses(courses)
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
        data.qpos[layout.bq : layout.bq + 7] = (*position, 1, 0, 0, 0)
        data.qvel[layout.bv : layout.bv + 6] = (*velocity, 0, velocity[0] / radius, 0)
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
