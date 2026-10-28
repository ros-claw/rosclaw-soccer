"""Reintegrate recorded actuator torques, never recompute teacher or actors."""

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout


def replay_saved_teacher_torques(
    model,
    initial_state: np.ndarray,
    history: dict,
    *,
    ball_body_name: str = "soccer_ball",
    teacher_forward_each_substep: bool = True,
) -> int:
    """Require exact shared qpos, qvel and grouped force at every native step."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    if (
        not isinstance(initial_state, np.ndarray)
        or initial_state.dtype != np.float64
        or initial_state.shape != (mujoco.mj_stateSize(model, spec),)
        or not np.isfinite(initial_state).all()
        or set(history) != {"qpos", "qvel", "torque", "ball_forces"}
        or type(teacher_forward_each_substep) is not bool
    ):
        raise ValueError("complete native integration snapshot and saved physics required")
    q, v, torques, contacts = (history[k] for k in ("qpos", "qvel", "torque", "ball_forces"))
    n = len(q)
    if (
        n < 1
        or q.shape != (n, 10, model.nq)
        or v.shape != (n, 10, model.nv)
        or torques.shape != (n, 10, model.nu)
        or contacts.shape != (n, 10, 2, 3)
        or not all(np.isfinite(x).all() for x in (q, v, torques, contacts))
        or (contacts < 0).any()
    ):
        raise ValueError("complete finite two-player native histories required")
    layouts = [
        NativeVelocityLayout(model, robot_prefix=p, ball_body_name=ball_body_name, player_count=2)
        for p in ("a/", "b/")
    ]
    owners = {}
    for index, (prefix, layout) in enumerate(zip(("a/", "b/"), layouts, strict=True)):
        feet = [
            int(model.body(prefix + side + "_ankle_roll_link").id) for side in ("left", "right")
        ]
        for body in range(1, model.nbody):
            parent = body
            while parent not in (0, layout.pelvis):
                parent = int(model.body_parentid[parent])
            if parent == layout.pelvis:
                owners[body] = (index, feet.index(body) if body in feet else 2)
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, initial_state, spec)
    mujoco.mj_forward(model, data)
    for control in range(n):
        if control:
            mujoco.mj_forward(model, data)
        for substep in range(10):
            if control and teacher_forward_each_substep:
                # The live teacher evaluates the current Jacobian at each 2ms boundary.
                mujoco.mj_forward(model, data)
            data.ctrl[:] = torques[control, substep]
            mujoco.mj_step(model, data)
            if (
                not np.array_equal(data.qpos, q[control, substep])
                or not np.array_equal(data.qvel, v[control, substep])
                or not np.array_equal(data.actuator_force, torques[control, substep])
            ):
                raise ValueError(f"saved native physics diverged at {control}:{substep}")
            forces = np.zeros((2, 3))
            for i, contact in enumerate(data.contact):
                b1, b2 = (int(model.geom_bodyid[g]) for g in contact.geom)
                ball = layouts[0].ball
                other = b2 if b1 == ball else b1 if b2 == ball else -1
                if other in owners:
                    player, part = owners[other]
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, wrench)
                    forces[player, part] = max(
                        forces[player, part], float(np.linalg.norm(wrench[:3]))
                    )
            if not np.array_equal(forces, contacts[control, substep]):
                raise ValueError(f"saved native contact forces diverged at {control}:{substep}")
    return n * 10
