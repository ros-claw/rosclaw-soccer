"""Explicit SIM handoff history: warm R2 on actual velocity-controlled motion.

Shadow R2 proposals are NEVER labeled as executed controls. The observation
history uses converted ACTUAL previous joint targets. This module has no reset
API and never changes a body pose, velocity, clock, or model parameter.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_causal_context import CausalMotorContext, sealed_float32
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot


def policy_action_from_hg_target(
    target: np.ndarray, offset: np.ndarray, scale: np.ndarray, order: np.ndarray
) -> np.ndarray:
    if (
        any(v.shape != (29,) or not np.isfinite(v).all() for v in (target, offset, scale))
        or target.dtype != np.float64
        or not (scale > 0).all()
        or order.shape != (29,)
        or order.dtype.kind not in "iu"
        or sorted(order.tolist()) != list(range(29))
    ):
        raise ValueError("finite actual HG29 target and bijective original R2 units required")
    result = np.empty(29, np.float32)
    result[order] = ((target - offset) / scale).astype(np.float32)
    if not np.isfinite(result).all() or not np.allclose(
        offset + result[order] * scale, target, atol=3e-7, rtol=0
    ):
        raise ValueError("actual target cannot be represented in original float32 policy units")
    return result


def _sealed(value: np.ndarray) -> np.ndarray:
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class AppliedVelocityControl:
    episode_key: int
    native_before: int
    time_before: float
    target_hg: np.ndarray
    terminal: dict[str, np.ndarray]
    row: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            self.target_hg.shape != (29,)
            or self.target_hg.dtype != np.float64
            or not np.isfinite(self.target_hg).all()
        ):
            raise ValueError("finite actual applied velocity target required")
        object.__setattr__(self, "target_hg", _sealed(self.target_hg))
        object.__setattr__(self, "terminal", {k: _sealed(v) for k, v in self.terminal.items()})

    def assert_current(self, ep: Any) -> None:
        if (
            id(ep) != self.episode_key
            or ep.work.native_steps != self.native_before + 10
            or abs(float(ep.env.data.time) - self.time_before - 0.02) > 1e-10
        ):
            raise ValueError("same actual one-control integration boundary required")
        actual = snapshot(ep.env.data)
        if set(actual) != set(self.terminal) or any(
            not np.array_equal(v, self.terminal[k]) for k, v in actual.items()
        ):
            raise ValueError("stale or physically modified velocity handoff rejected")


def integrate_velocity_target(
    ep: Any, target: np.ndarray, kp: np.ndarray, kd: np.ndarray
) -> AppliedVelocityControl:
    """One bounded native SIM control, without changing the R2 gain profile."""
    import mujoco

    from rosclaw_soccer.rsi.velocity_torque_transfer import (
        decode_pd_torque,
        require_torque_encoding,
    )

    if ep.done or mujoco.get_mjcb_control() is not None or ep.env.model.opt.timestep != 0.002:
        raise ValueError("live callback-free native SIM episode required")
    m, d = ep.env.model, ep.env.data
    effort = require_torque_encoding(m, ep.layout)
    if not np.array_equal(effort, ep.effort):
        raise ValueError("same compiled joint effort limits required")
    before_count, before_time = ep.work.native_steps, float(d.time)
    row = dict(
        native_qpos_before=d.qpos.copy(),
        native_qvel_before=d.qvel.copy(),
        target=target.copy(),
        time_before=before_time,
    )
    samples = []
    for _ in range(10):
        q, v = ep.layout.canonical(d)
        torque = decode_pd_torque(target, q[7:36], v[6:35], kp, kd, effort)
        d.ctrl[ep.layout.ai] = torque
        clock = float(d.time)
        mujoco.mj_step(m, d)
        ep.work.native_steps += 1
        if (
            abs(d.time - clock - 0.002) > 1e-10
            or tuple(int(w.number) for w in d.warning) != ep.warning
            or not np.array_equal(d.actuator_force[ep.layout.ai], torque)
        ):
            raise ValueError("external velocity native effort/timing/warning changed")
        q, v = ep.layout.canonical(d)
        samples.append(dict(qpos=q, qvel=v, forces=ep.forces(), torque=torque.copy()))
    row.update({k: np.stack([s[k] for s in samples]) for k in samples[0]})
    q = row["qpos"]
    row["body_failed"] = bool(
        ((q[:, 2] < 0.65) | (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2) < 0.9)).any()
    )
    if row["body_failed"]:
        ep.done = True
        ep.termination_reason = "BODY_FAILURE"
    return AppliedVelocityControl(id(ep), before_count, before_time, target, snapshot(d), row)


class OwnVelocityShadow:
    """One player's explicit R2 private history; no cross-player or teacher state."""

    def __init__(self, ep: Any, actor: Any) -> None:
        if (
            ep.done
            or ep.frame != 0
            or ep.ready
            or actor.training
            or any(p.requires_grad or p.device.type != "cpu" for p in actor.parameters())
        ):
            raise ValueError("fresh reference boundary and frozen own CPU R2 actor required")
        self.ep, self.actor = ep, actor
        self._state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
        self._pending: tuple[CausalMotorContext, np.ndarray, int, float] | None = None
        self.records: list[dict[str, Any]] = []

    def observe(self) -> tuple[CausalMotorContext, np.ndarray]:
        ep = self.ep
        if (
            self._pending is not None
            or ep.done
            or ep.frame != len(self.records)
            or not 0 <= ep.frame < 80
        ):
            raise ValueError("one current uncommitted shadow observation per real control required")
        obs = ep.observation()
        before_physics = snapshot(ep.env.data)
        before = [s.numpy().copy() for s in self._state]
        with torch.no_grad():
            proposal, after = self.actor(torch.from_numpy(obs).reshape(1, 1, 160), self._state)
        if any(not np.array_equal(v, snapshot(ep.env.data)[k]) for k, v in before_physics.items()):
            raise ValueError("neural shadow proposal changed physical state")
        context = CausalMotorContext(
            id(ep), ep.frame, obs, before[0], before[1], after[0].numpy(), after[1].numpy()
        )
        action = sealed_float32(proposal[0, 0].numpy(), (29,))
        self._pending = (context, action, ep.work.native_steps, float(ep.env.data.time))
        return context, action

    def adopt_before_shadow(self, control: AppliedVelocityControl) -> None:
        if self.records or self._pending is not None or self.ep.frame != 0 or self.ep.ready:
            raise ValueError(
                "only pre-reference controls may adopt history without shadow inference"
            )
        control.assert_current(self.ep)
        self.ep.last_action = policy_action_from_hg_target(
            control.target_hg, self.ep.offset, self.ep.scale, self.ep.order
        )

    def commit_velocity(self, control: AppliedVelocityControl) -> None:
        if self._pending is None or self.ep.done or not self.ep.ready:
            raise ValueError("own observed shadow and healthy actual velocity motion required")
        context, proposal, native, clock = self._pending
        context.assert_current(self.ep)
        control.assert_current(self.ep)
        if native != control.native_before or clock != control.time_before:
            raise ValueError("shadow query must precede exactly this actual control")
        ep = self.ep
        applied = policy_action_from_hg_target(control.target_hg, ep.offset, ep.scale, ep.order)
        self.records.append(
            dict(
                observation=context.observation,
                R2_shadow_proposal=proposal,
                h_before=context.h_before,
                c_before=context.c_before,
                h_after=context.h_after,
                c_after=context.c_after,
                actual_target_hg=control.target_hg,
                actual_previous_action_next=applied.copy(),
                proposal_executed=False,
                reference_frame=ep.frame,
            )
        )
        self._state = (
            torch.from_numpy(context.h_after.copy()),
            torch.from_numpy(context.c_after.copy()),
        )
        ep.last_action = applied
        ep.ref_pos = ep.clip.joint_pos[ep.frame].copy()
        ep.ref_vel = ep.clip.joint_vel[ep.frame].copy()
        ep.ref_ang = ep.clip.body_ang_vel_w[ep.frame, ep.torso_index].copy()
        ep.frame += 1
        ep.ready = False
        self._pending = None

    def suffix_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._pending is not None
            or len(self.records) != 80
            or self.ep.frame != 80
            or self.ep.ready
            or self.ep.done
        ):
            raise ValueError("all80 own causal shadow controls must commit before suffix")
        return self._state[0].detach().clone(), self._state[1].detach().clone()
