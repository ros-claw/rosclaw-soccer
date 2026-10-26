"""Verify a recorded SIM torque trial with a separately composed motor law.

Targets are verification inputs only. This never calls the producing torque
policy or layer and does not turn replay into another independent scenario.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan


def replay_torque_trial(
    model: Any,
    data: Any,
    layout: Any,
    raw: dict[str, np.ndarray],
    plan: CFtrackTorqueIntentPlan,
    *,
    kp: np.ndarray,
    kd: np.ndarray,
    effort: np.ndarray,
    controls: int = 160,
) -> dict[str, Any]:
    import mujoco

    if (
        type(model) is not mujoco.MjModel
        or type(data) is not mujoco.MjData
        or layout.model is not model
        or type(plan) is not CFtrackTorqueIntentPlan
        or type(controls) is not int
        or not 1 <= controls <= 160
        or data.time != 0
    ):
        raise ValueError("bounded cold native SIM replay required")
    for value in (kp, kd, effort):
        if (
            type(value) is not np.ndarray
            or value.shape != (29,)
            or value.dtype != np.float64
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite float64 HG29 gains and actual effort required")
    if (kp < 0).any() or (kd < 0).any() or (effort <= 0).any():
        raise ValueError("nonnegative gains and positive effort required")
    n = controls * 10
    for key, shape in (
        ("initial_qpos", (model.nq,)),
        ("initial_qvel", (model.nv,)),
        ("qpos", (n, model.nq)),
        ("qvel", (n, model.nv)),
        ("torque", (n, model.nu)),
        ("parent_torque", (n, model.nu)),
        ("ball_forces", (n, 1, 3)),
        ("joint_targets", (controls, 29)),
    ):
        value = raw[key]
        if type(value) is not np.ndarray or value.shape != shape or not np.isfinite(value).all():
            raise ValueError("complete finite native replay measurement required: " + key)
    if not np.array_equal(data.qpos, raw["initial_qpos"]) or not np.array_equal(
        data.qvel, raw["initial_qvel"]
    ):
        raise ValueError("independently initialized body differs from evidence")
    feet = [
        int(model.body(layout.robot_prefix + side + "_ankle_roll_link").id)
        for side in ("left", "right")
    ]
    parts = {}
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor not in (0, layout.pelvis):
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == layout.pelvis:
            parts[body] = feet.index(body) if body in feet else 2
    offsets = plan.offsets()
    saturated = stopped = 0
    for step in range(n):
        parent = np.clip(
            kp * (raw["joint_targets"][step // 10] - data.qpos[layout.qi])
            - kd * data.qvel[layout.vi],
            -effort,
            effort,
        )
        relative = step / 10 - plan.start_control
        delta = np.zeros(29)
        if 0 <= relative < 48:
            index = int(relative)
            alpha = relative - index
            delta = ((1 - alpha) * offsets[index] + alpha * offsets[index + 1]) * effort
        q = data.qpos[layout.qi]
        limits = model.jnt_range[layout.joints]
        # Mirror the native frame's unlimited-joint convention, not raw zero ranges.
        limited = model.jnt_limited[layout.joints].astype(bool)
        lo = np.where(limited, limits[:, 0], -1e6)
        hi = np.where(limited, limits[:, 1], 1e6)
        outward = ((q <= lo) & (delta < 0)) | ((q >= hi) & (delta > 0))
        stopped += int(outward.any())
        delta[outward] = 0
        proposed = parent.copy()
        changed = delta != 0
        proposed[changed] += delta[changed]
        actual = np.clip(proposed, -effort, effort)
        saturated += int(not np.array_equal(actual, proposed))
        actual[~changed] = parent[~changed]
        if not np.array_equal(parent, raw["parent_torque"][step, layout.ai]):
            raise ValueError(f"independent parent PD mismatch at native step {step}")
        data.ctrl[layout.ai] = actual
        mujoco.mj_step(model, data)
        forces = np.zeros((1, 3))
        for ci in range(data.ncon):
            contact = data.contact[ci]
            b1, b2 = (int(model.geom_bodyid[g]) for g in contact.geom)
            other = b2 if b1 == layout.ball else b1 if b2 == layout.ball else -1
            if other in parts:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, wrench)
                forces[0, parts[other]] = max(
                    forces[0, parts[other]], float(np.linalg.norm(wrench[:3]))
                )
        if (
            any(int(w.number) for w in data.warning)
            or not np.array_equal(data.qpos, raw["qpos"][step])
            or not np.array_equal(data.qvel, raw["qvel"][step])
            or not np.array_equal(data.actuator_force, raw["torque"][step])
            or not np.array_equal(data.actuator_force[layout.ai], actual)
            or not np.array_equal(forces, raw["ball_forces"][step])
        ):
            raise ValueError(f"independent native state/torque/contact mismatch at step {step}")
    return dict(
        native_steps=n,
        saturation_steps=saturated,
        measured_stop_guard_steps=stopped,
        values_exact=True,
        producer_torque_policy_or_layer_used=False,
        plan_envelope_math_shared=True,
        hardware_authorized=False,
    )
