"""Whole-world native replay with own RNN and explicit shared IK recomputation.

No producer integration, shadow state, standing controller or raw-state resets.
Observation and kinematic solver are shared admitted codecs, not independent
implementations; report this boundary rather than claim algorithmic diversity.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch


def equal(actual: Any, expected: Any, label: str) -> None:
    if not np.array_equal(actual, expected):
        raise ValueError("whole-run audit mismatch: " + label)


def private_replay_state(actor: Any, hidden_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Only explicitly admitted motor classes may allocate dual256 history."""
    if type(hidden_width) is not int or hidden_width not in (128, 256):
        raise ValueError("known private recurrent context width required")
    if hidden_width == 256:
        from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
        from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
        from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
        from rosclaw_soccer.rsi.supported_refinement_motor import SupportedRefinementMotor

        if (
            type(actor)
            not in (
                DualRecurrentMotor,
                ConsolidatedMotorAdapter,
                RefinementMotorAdapter,
                SupportedRefinementMotor,
            )
            or actor.training
            or any(p.requires_grad or p.device.type != "cpu" for p in actor.parameters())
        ):
            raise ValueError("independent replay requires explicit frozen CPU dual256 motor")
        admitted: (
            DualRecurrentMotor
            | ConsolidatedMotorAdapter
            | RefinementMotorAdapter
            | SupportedRefinementMotor
        ) = actor
        return admitted.initial_state()
    return torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)


def replay_teacher_world(
    ep: Any,
    actor: Any,
    kick: dict[str, np.ndarray],
    stand: dict[str, np.ndarray] | None,
    *,
    latent: np.ndarray,
    center: int,
    foot: str,
    counts: dict[str, int],
    contact_observer: Callable[[dict[str, Any]], None] | None = None,
    hidden_width: int = 128,
    action_adapter: Callable[[Any, np.ndarray, dict[str, Any]], tuple[np.ndarray, int]]
    | None = None,
    allow_incomplete_healthy_prefix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count actual work even if a later comparison fails; never retry here."""
    import mujoco

    from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
    from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

    if (
        mujoco.__version__ != "3.5.0"
        or mujoco.get_mjcb_control() is not None
        or ep.env.model.opt.timestep != 0.002
        or actor.training
        or any(p.requires_grad or p.device.type != "cpu" for p in actor.parameters())
        or not 1 <= len(kick["qpos"]) <= ep.clip.length
        or (stand is not None and not 1 <= len(stand["qpos"]) <= 100)
    ):
        raise ValueError("bounded native CPU whole-run evidence required")
    for raw in (kick, stand):
        if raw is not None and any(not np.isfinite(v).all() for v in raw.values()):
            raise ValueError("finite raw primitives required")
    model, data = ep.env.model, ep.env.data
    if action_adapter is not None and hidden_width != 256:
        raise ValueError("new action reconstruction cannot reinterpret legacy128 audit controls")
    if allow_incomplete_healthy_prefix and (
        hidden_width != 256
        or action_adapter is None
        or stand is not None
        or len(kick["qpos"]) >= ep.clip.length
    ):
        raise ValueError("explicit new dual-state interrupted-prefix audit only, never success")
    if contact_observer is not None and model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
        raise ValueError("contact clock diagnostic excludes multi-stage RK4 solver geometry")
    intent = FootTaskSpaceIntent(model, foot=foot, center=center, latent=latent)
    state = private_replay_state(actor, hidden_width)
    body_failed = False
    for phase, raw in (("kick", kick), ("stand", stand)):
        if raw is None:
            continue
        if phase == "stand" and (body_failed or ep.frame != ep.clip.length):
            raise ValueError("standing requires uncensored actual reference end")
        for k, v in snapshot(data).items():
            equal(v, raw["initial_" + k], phase + " initial " + k)
        for frame in range(len(raw["qpos"])):
            if body_failed:
                raise ValueError("body failure must terminate at first actual control boundary")
            equal(data.qpos, raw["native_qpos_before"][frame], "q before")
            equal(data.qvel, raw["native_qvel_before"][frame], "v before")
            equal(float(data.time), raw["time_before"][frame], "time before")
            if phase == "kick":
                obs = ep.observation()
            else:
                mujoco.mj_forward(model, data)
                obs = ep.build_observation(
                    ep.env,
                    None,
                    0,
                    ep.last_action,
                    ep.ref_pos.astype(np.float32),
                    np.zeros(29, np.float32),
                    np.zeros(3, np.float32),
                    ep.env.ball_pos,
                    ep.goal,
                    ep.offset,
                    obs_dim=None,
                )
                equal(
                    raw["STAND_reference_pose"][frame], ep.ref_pos.astype(np.float32), "stand pose"
                )
                equal(raw["STAND_control"][frame], frame, "stand cursor")
            equal(obs, raw["NN_obs"][frame], "own observation")
            for tensor, k in zip(state, ("NN_h_before", "NN_c_before"), strict=True):
                equal(tensor.numpy(), raw[k][frame], "own " + k)
            with torch.no_grad():
                if action_adapter is not None:
                    mean, state, evidence = actor.forward_with_evidence(
                        torch.from_numpy(obs).reshape(1, 1, 160), state
                    )
                else:
                    mean, state = actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
            counts["neural"] += 1
            nominal = mean[0, 0].numpy().copy()
            equal(nominal, raw["NN_nominal_action"][frame], "own nominal")
            for tensor, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True):
                equal(tensor.numpy(), raw[k][frame], "own " + k)
            if phase == "kick":
                before = snapshot(data)
                if action_adapter is None:
                    action, plan = intent(ep, nominal.copy())
                    counts["kinematic_intent"] += 1
                    for k, value in plan.items():
                        equal(value, raw["PLAN_" + k][frame], "own IK " + k)
                else:
                    action, planner_calls = action_adapter(
                        ep, nominal.copy(), dict(evidence, actual_observation=obs.copy())
                    )
                    if type(planner_calls) is not int or planner_calls not in (0, 1):
                        raise ValueError("exact bounded actual solver invocation count required")
                    counts["kinematic_intent"] += planner_calls
                for k, value in snapshot(data).items():
                    equal(value, before[k], "no IK physical write " + k)
            else:
                action = nominal
            equal(action, raw["NN_action"][frame], "applied action")
            target = ep.offset + action[ep.order] * ep.scale
            equal(target, raw["target"][frame], "actual target")
            equal(raw["neural_control"][frame], True, "no neural padding")
            bad = False
            for sub in range(10):
                q, v = ep.layout.canonical(data)
                torque = np.minimum(
                    ep.effort, np.maximum(-ep.effort, ep.kp * (target - q[7:36]) - ep.kd * v[6:35])
                )
                equal(torque, raw["torque"][frame, sub], "own PD caps")
                data.ctrl[ep.layout.ai] = torque
                clock = float(data.time)
                mujoco.mj_step(model, data)
                counts["native"] += 1
                if (
                    abs(data.time - clock - 0.002) > 1e-10
                    or tuple(int(w.number) for w in data.warning) != ep.warning
                ):
                    raise ValueError("native timing/warning differs")
                q, v = ep.layout.canonical(data)
                equal(q, raw["qpos"][frame, sub], "whole physical q")
                equal(v, raw["qvel"][frame, sub], "whole physical v")
                equal(data.actuator_force[ep.layout.ai], torque, "actual effort")
                forces = np.zeros(3)
                for j, contact in enumerate(data.contact):
                    b1, b2 = [int(model.geom_bodyid[g]) for g in contact.geom]
                    other = b2 if b1 == ep.layout.ball else b1 if b2 == ep.layout.ball else -1
                    if other in ep.robot:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(model, data, j, wrench)
                        role = ep.feet.index(other) if other in ep.feet else 2
                        norm = float(np.linalg.norm(wrench[:3]))
                        forces[role] = max(forces[role], norm)
                        if contact_observer is not None and norm > (1.0 if role == 2 else 10.0):
                            contact_observer(
                                dict(
                                    solver_time_s=clock,
                                    end_step_time_s=float(data.time),
                                    reference_frame=ep.frame,
                                    phase=phase,
                                    role=role,
                                    robot_body=mujoco.mj_id2name(
                                        model, mujoco.mjtObj.mjOBJ_BODY, other
                                    ),
                                    geom_names=[
                                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                                        for g in contact.geom
                                    ],
                                    force_norm_N=norm,
                                    contact_position_world=contact.pos.tolist(),
                                    ball_solver_center_world=data.xpos[ep.layout.ball].tolist(),
                                    ball_position_after_step=q[36:39].tolist(),
                                    ball_velocity_after_step=v[35:38].tolist(),
                                )
                            )
                equal(forces, raw["forces"][frame, sub], "native contacts")
                bad |= bool(q[2] < 0.65 or 1 - 2 * (q[4] ** 2 + q[5] ** 2) < 0.9)
            equal(bad, raw["body_failed"][frame], "failure boundary")
            body_failed = bad
            ep.last_action = action.copy()
            if phase == "kick":
                ep.ref_pos = ep.clip.joint_pos[ep.frame].copy()
                ep.ref_vel = ep.clip.joint_vel[ep.frame].copy()
                ep.ref_ang = ep.clip.body_ang_vel_w[ep.frame, ep.torso_index].copy()
                ep.frame += 1
                ep.ready = False
        for k, v in snapshot(data).items():
            equal(v, raw["terminal_" + k], phase + " terminal " + k)
    if (
        not body_failed
        and not allow_incomplete_healthy_prefix
        and (ep.frame != ep.clip.length or stand is None or len(stand["qpos"]) != 100)
    ):
        raise ValueError("healthy world cannot silently omit reference/standing suffix")
    return state[0], state[1]
