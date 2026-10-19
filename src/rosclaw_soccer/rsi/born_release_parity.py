"""Independent born-lease algebra on recorded current state, zero integration.

Uses the same authenticated kinematic sampler, not a second kinematic engine.
Does not call the candidate control or its action generator. Body/NN causality
and the independent recorded-body replay remain separate prerequisites.
"""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.rsi.navigation_recovery_parity import review_constant_forward_commands
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS

FIELDS = {
    "born_release_requested_offset_nm",
    "born_release_applied_offset_nm",
    "born_release_forecast_steps",
}
KNOWN_ACTIONS = {
    "baseline",
    "forward_plus",
    "forward_minus",
    "lateral_plus",
    "lateral_minus",
    "lift_plus",
    "lift_minus",
    "yaw_plus",
    "yaw_minus",
    "lateral-1_yaw-1",
    "lateral-1_yaw+1",
    "lateral+1_yaw-1",
    "lateral+1_yaw+1",
}


def body_arrays(trace: dict[str, Any]) -> dict[str, Any]:
    if {k for k in trace if k.startswith("born_release_")} != FIELDS:
        raise ValueError("exact whole-episode born capture required")
    base = {k: v for k, v in trace.items() if k not in FIELDS}
    original_arrays(base)
    for key, shape, dtype in (
        ("born_release_requested_offset_nm", (300, 1, 29), np.float64),
        ("born_release_applied_offset_nm", (300, 1, 10, 29), np.float64),
        ("born_release_forecast_steps", (300, 1, 1), np.int64),
    ):
        a = trace[key]
        if (
            type(a) is not np.ndarray
            or a.shape != shape
            or a.dtype != dtype
            or not np.isfinite(a).all()
        ):
            raise ValueError("finite whole born-control captures required")
    if np.any(trace["born_release_forecast_steps"]):
        raise ValueError("external task predictions cannot be disguised as lease-local forecasts")
    return base


def action_map(
    linear: np.ndarray[Any, Any], angular: np.ndarray[Any, Any], side: int
) -> dict[str, np.ndarray[Any, Any]]:
    """Rebuild fixed40N/4Nm wishes and alias removal, not candidate callbacks."""
    if (
        type(side) is not int
        or side not in (0, 1)
        or any(
            type(a) is not np.ndarray
            or a.shape != (3, 29)
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            or np.max(np.abs(a)) > 1e6
            for a in (linear, angular)
        )
    ):
        raise ValueError("finite canonical actuated-only contact Jacobians required")
    forward = linear.T @ np.array([40.0, 0, 0])
    lateral = linear.T @ np.array([0, 40.0, 0])
    lift = linear.T @ np.array([0, 0, 40.0])
    yaw = angular.T @ np.array([0, 0, 4.0])
    wishes = [
        ("baseline", np.zeros(29)),
        ("forward_plus", forward),
        ("forward_minus", -forward),
        ("lateral_plus", lateral),
        ("lateral_minus", -lateral),
        ("lift_plus", lift),
        ("lift_minus", -lift),
        ("yaw_plus", yaw),
        ("yaw_minus", -yaw),
        ("lateral-1_yaw-1", -lateral - yaw),
        ("lateral-1_yaw+1", -lateral + yaw),
        ("lateral+1_yaw-1", lateral - yaw),
        ("lateral+1_yaw+1", lateral + yaw),
    ]
    caps = np.minimum(12.0, 0.2 * np.asarray(G1_HARD_TORQUE_LIMITS))
    result, seen = {}, set()
    ids = slice(side * 6, side * 6 + 6)
    for name, wish in wishes:
        value = np.zeros(29)
        value[ids] = np.clip(wish[ids], -caps[ids], caps[ids])
        value[value == 0] = 0.0
        if value.tobytes() not in seen:
            result[name] = value
            seen.add(value.tobytes())
    return result


def review_born_release(
    trace: dict[str, Any],
    baseline: dict[str, Any],
    *,
    action_name: str,
    model: Any,
    world_hash: str,
) -> dict[str, Any]:
    base = body_arrays(trace)
    body_arrays(baseline)
    review_constant_forward_commands(base)
    outcome = label_first_touch(base).to_dict()
    first = outcome["first_foot_contact_frame"]
    if type(first) is not int or not 30 <= first < 299:
        raise ValueError("whole actual completed-foot event required")
    start = first + 1
    if trace.keys() != baseline.keys():
        raise ValueError("same whole born-capture layout required")
    for k, a in trace.items():
        if a[:start].tobytes() != baseline[k][:start].tobytes():
            raise ValueError("whole pre-birth body/control/NN/capture prefix differs: " + k)
        if action_name == "baseline" and a.tobytes() != baseline[k].tobytes():
            raise ValueError("baseline all46 arrays must match")
    for k in (
        "canonical_qpos",
        "canonical_qvel",
        "joint_target_rad",
        "foundation_neural_encoder_features",
        "foundation_neural_raw_action_isaac",
    ):
        if trace[k][start].tobytes() != baseline[k][start].tobytes():
            raise ValueError("same pre-action body/neural state at birth required: " + k)
    requested = np.zeros((300, 1, 29))
    available = None
    side = None
    cancelled = None
    if action_name != "baseline":
        if type(action_name) is not str or not action_name.startswith("born_release:"):
            raise ValueError("distinct born action name required")
        name = action_name.split(":", 1)[1]
        if name not in KNOWN_ACTIONS:
            raise ValueError("unknown born action")
        unsafe = bool(
            base["root_pose_xyzw_m"][start, 0, 2] < 0.65
            or np.any(base["force_n"][start - 1, 0, 2:] > 1)
        )
        held = np.zeros(29)
        if not unsafe:
            sampler = G1JointTaskKinematicSampler(model, expected_world_hash=world_hash)
            data = mujoco.MjData(model)
            q, v = base["canonical_qpos"][start, 0], base["canonical_qvel"][start, 0]
            ball = int(model.geom_bodyid[model.geom("ball_geom").id])
            joint = int(model.body_jntadr[ball])
            bq, bv = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
            data.qpos[:7], data.qpos[sampler._qi], data.qpos[bq : bq + 7] = q[:7], q[7:36], q[36:]
            data.qvel[:6], data.qvel[sampler._vi], data.qvel[bv : bv + 6] = v[:6], v[6:35], v[35:]
            data.time = start * 0.02
            sample = sampler.sample(data)
            side = int(np.argmax(base["force_n"][start - 1, 0, :2]))
            bank = action_map(
                sample.foot_material_point_jacobian_m_per_rad[side],
                sample.foot_rotation_jacobian[side],
                side,
            )
            available = name in bank
            if available:
                held = bank[name]
        for f in range(start, 300):
            unsafe = unsafe or bool(
                base["root_pose_xyzw_m"][f, 0, 2] < 0.65
                or np.any(base["force_n"][f - 1, 0, 2:] > 1)
            )
            if f < start + 3 and not unsafe and available:
                requested[f, 0] = held
        cancelled = unsafe
    actual = trace["born_release_requested_offset_nm"]
    if actual.tobytes() != requested.tobytes():
        raise ValueError("independent current-Jacobian born request/cancellation differs")
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    pd = base["pd_requested_torque_nm"]
    parent = np.clip(pd, -limits, limits)
    tau = np.where(
        np.any(actual != 0, axis=-1)[:, :, None, None],
        np.clip(pd + actual[:, :, None, :], -limits, limits),
        parent,
    )
    if not np.array_equal(tau, base["torque_nm"]) or not np.array_equal(
        tau - parent, trace["born_release_applied_offset_nm"]
    ):
        raise ValueError("actual born torque attribution differs")
    return dict(
        start_frame=start,
        action_name=action_name,
        action_available=available,
        measured_contact_side=side,
        cancelled=cancelled,
        trigger_frame=start if available is not None else None,
        current_jacobian_requests_rebuilt=True,
        native_torque_attribution_rebuilt=True,
        new_physical_steps=0,
        new_nn_inferences=0,
        teacher_qualified=False,
        hardware_authorized=False,
    )
