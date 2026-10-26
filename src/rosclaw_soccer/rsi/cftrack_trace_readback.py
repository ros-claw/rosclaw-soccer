"""Independent original NN/history/target readback using scratch measurements only.

The caller pins the source and regenerates the initial reference independently.
Saved targets are evidence to compare, never control commands for an actor.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


def verify_cftrack_trace(
    policy: Any,
    data: Any,
    layout: Any,
    raw: dict[str, np.ndarray],
    *,
    controls: int,
    motor_plan: CFtrackMotorResidualPlan | None = None,
    proprioception_audit: Callable[[Any, np.ndarray], None] | None = None,
    initial_cached_qpos: np.ndarray | None = None,
) -> int:
    import mujoco
    import onnxruntime as ort

    if (
        type(controls) is not int
        or not 1 <= controls <= 600
        or type(data) is not mujoco.MjData
        or data.time != 0
        or policy.counter_step != 0
        or policy.async_stage_thread is not None
        or type(policy.ort_session) is not ort.InferenceSession
        or policy.ort_session.get_providers() != ["CPUExecutionProvider"]
    ):
        raise ValueError(
            "bounded cold original CPU policy and never-integrated scratch data required"
        )
    for key, shape in [
        ("qpos", (controls * 10, 43)),
        ("qvel", (controls * 10, 41)),
        ("joint_targets", (controls, 29)),
        ("neural_obs", (controls, 1, 1244)),
        ("neural_actions", (controls, 1, 29)),
    ]:
        value = raw[key]
        if (
            type(value) is not np.ndarray
            or value.shape != shape
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite complete readback trace required: " + key)
    if motor_plan is not None and type(motor_plan) is not CFtrackMotorResidualPlan:
        raise ValueError("explicit motor plan required")
    if "nn_decoded_targets" in raw:
        a = raw["nn_decoded_targets"]
        if a.shape != (controls, 29) or not np.isfinite(a).all():
            raise ValueError("complete decoded NN targets required")
    elif motor_plan is not None:
        raise ValueError("motor trace must separate original NN targets")
    model = layout.model
    if initial_cached_qpos is not None and (
        type(initial_cached_qpos) is not np.ndarray
        or initial_cached_qpos.shape != (model.nq,)
        or initial_cached_qpos.dtype.kind != "f"
        or not np.isfinite(initial_cached_qpos).all()
    ):
        raise ValueError("finite explicit prior native position for initial cached ball required")
    original = policy.ort_session
    calls = 0

    class Verify:
        def run(self, outputs: Any, inputs: Any) -> Any:
            nonlocal calls
            if (
                calls >= controls
                or not np.array_equal(inputs["obs"], raw["neural_obs"][calls])
                or not np.array_equal(inputs["time_step"], np.zeros((1, 1), np.float32))
            ):
                raise ValueError("original proprioception/reference/history input does not rebuild")
            if proprioception_audit is not None:
                proprioception_audit(policy, inputs["obs"])
            result = original.run(outputs, inputs)
            if not np.array_equal(result[0], raw["neural_actions"][calls]):
                raise ValueError("original NN output differs")
            calls += 1
            return result

    view = SimpleNamespace(
        world=SimpleNamespace(data=data, control_steps=0), policy=policy, layout=layout
    )
    policy.ort_session = Verify()
    try:
        for tick in range(controls):
            # Author's mixed-time cache: ball pose is from before last substep,
            # while actual joints/root and their velocities are after it.
            if tick:
                data.qpos[:] = raw["qpos"][tick * 10 - 2]
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                data.qpos[:] = raw["qpos"][tick * 10 - 1]
                data.qvel[:] = raw["qvel"][tick * 10 - 1]
            else:
                current = data.qpos.copy()
                if initial_cached_qpos is not None:
                    data.qpos[:] = initial_cached_qpos
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                data.qpos[:] = current
            view.world.control_steps = tick
            SharedOmniContactPlayer._observe(cast(SharedOmniContactPlayer, view))
            policy.run()
            decoded = policy.policy_output.actions.astype(np.float64)
            if "nn_decoded_targets" in raw and not np.array_equal(
                decoded, raw["nn_decoded_targets"][tick]
            ):
                raise ValueError("original decoded NN target differs")
            applied = (
                motor_plan.apply(
                    decoded,
                    counter=tick,
                    limits_hg=model.jnt_range[layout.joints].astype(np.float64),
                )
                if motor_plan is not None
                else decoded
            )
            if (
                policy.counter_step != tick + 1
                or policy.async_stage_thread is not None
                or not np.array_equal(applied, raw["joint_targets"][tick])
            ):
                raise ValueError("decoded/residual target or policy clock differs")
        if data.time != 0:
            raise ValueError("scratch readback advanced physical time")
    finally:
        policy.ort_session = original
    return calls
