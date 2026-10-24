"""Explicit SIM static-standing task using an already learned motor network.

This is NOT an extension/padding of a football motion reference. A new static
joint-pose goal and zero reference velocities are declared; actual body and
previous-action feedback continue every20ms under original every2ms PD caps.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.motor_option_state import MotorOptionState
from rosclaw_soccer.rsi.paid_balance_recovery import body_feedback
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
from rosclaw_soccer.rsi.private_motor_history import (
    MotorHistory,
    clone_history,
    history_trace,
    recurrent_history,
)


class NeuralStandingTask:
    """Private per-player RNN context, at most 150 newly declared controls."""

    def __init__(
        self,
        ep: Any,
        actor: Any,
        *,
        state: MotorHistory,
        memory_mode: str,
        count_inference: Callable[[], None],
        hidden_width: int = 128,
    ) -> None:
        if (
            memory_mode not in ("carry", "reset")
            or type(hidden_width) is not int
            or hidden_width not in (128, 256)
            or not ep.done
            or ep.termination_reason != "RECOVERY_ADMITTED"
            or ep.frame != ep.clip.length
            or ep.ref_pos.shape != (29,)
            or not np.isfinite(ep.ref_pos).all()
        ):
            raise ValueError(
                "actual reference-end admission and private finite motor context required"
            )
        recurrent_history(state, hidden_width=hidden_width)
        if hidden_width == 256:
            from rosclaw_soccer.rsi.committed_neural_motor import CommittedNeuralMotor
            from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
            from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
            from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
            from rosclaw_soccer.rsi.supported_refinement_motor import SupportedRefinementMotor

            if type(actor) not in (
                DualRecurrentMotor,
                ConsolidatedMotorAdapter,
                RefinementMotorAdapter,
                SupportedRefinementMotor,
                CommittedNeuralMotor,
            ):
                raise ValueError("new dual256 state requires the explicit research motor contract")
            if (type(actor) is CommittedNeuralMotor) != (type(state) is MotorOptionState):
                raise ValueError("standing requires actor-bound private history representation")
            if type(state) is MotorOptionState and state.generation_hash != actor.generation_hash:
                raise ValueError("standing history belongs to another model generation")
        self.hidden_width = hidden_width
        self.ep, self.actor, self.memory_mode, self.count_inference = (
            ep,
            actor,
            memory_mode,
            count_inference,
        )
        self.state: MotorHistory = (
            clone_history(state, hidden_width=hidden_width)
            if memory_mode == "carry"
            else actor.initial_state()
            if hidden_width == 256
            else (
                torch.zeros(2, 1, hidden_width, dtype=torch.float32, device="cpu"),
                torch.zeros(2, 1, hidden_width, dtype=torch.float32, device="cpu"),
            )
        )
        self.pose = ep.ref_pos.astype(np.float32, copy=True)
        self.goal = ep.goal.copy()
        self.expected_action = ep.last_action.copy()
        self.expected_time = float(ep.env.data.time)
        self.origin_clip, self.origin_frame = ep.clip, ep.frame
        self.controls = 0

    def step(self) -> dict[str, Any]:
        import mujoco

        ep = self.ep
        if (
            self.controls >= 150
            or not ep.done
            or ep.clip is not self.origin_clip
            or ep.frame != self.origin_frame
            or ep.termination_reason not in ("RECOVERY_ADMITTED", "REFERENCE_COMPLETE")
            or not np.isclose(float(ep.env.data.time), self.expected_time, rtol=0, atol=1e-12)
            or not np.array_equal(ep.last_action, self.expected_action)
            or not np.array_equal(ep.goal, self.goal)
            or not np.array_equal(ep.env.data.mocap_pos[ep.env.goal_mocap_id], self.goal)
        ):
            raise ValueError("standing context expired, body failed or bounded task completed")
        mujoco.mj_forward(ep.env.model, ep.env.data)
        # Pinned observer explicitly ignores motion/time. This static task has
        # no motion array or replay cursor; do not call ep.observation past end.
        obs = ep.build_observation(
            ep.env,
            None,
            0,
            ep.last_action,
            self.pose,
            np.zeros(29, np.float32),
            np.zeros(3, np.float32),
            ep.env.ball_pos,
            self.goal,
            ep.offset,
            obs_dim=None,
        )
        if obs.shape != (160,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
            raise ValueError("exact measured160 static-standing observation required")
        before = clone_history(self.state, hidden_width=self.hidden_width)
        with torch.no_grad():
            mean, next_state = self.actor(torch.from_numpy(obs).reshape(1, 1, 160), self.state)
        self.count_inference()
        if (
            mean.shape != (1, 1, 29)
            or mean.dtype != torch.float32
            or mean.device.type != "cpu"
            or not torch.isfinite(mean).all()
        ):
            raise ValueError("finite own CPU29 mean and next private memory required")
        private_trace = history_trace(before, next_state, hidden_width=self.hidden_width)
        action = mean[0, 0].numpy().copy()
        ep.target = ep.offset + action[ep.order] * ep.scale
        ep.last_action = action.copy()
        row: dict[str, Any] = ep._integrate_target(True)
        row.update(
            NN_obs=obs.copy(),
            NN_nominal_action=action.copy(),
            NN_action=action.copy(),
            **private_trace,
            STAND_reference_pose=self.pose.copy(),
            STAND_control=self.controls,
        )
        self.state = clone_history(next_state, hidden_width=self.hidden_width)
        self.expected_action = ep.last_action.copy()
        self.expected_time = float(ep.env.data.time)
        self.controls += 1
        if row["body_failed"]:
            ep.termination_reason = "BODY_FAILURE"
        elif self.controls >= 100:
            ep.termination_reason = "REFERENCE_COMPLETE"  # NEW stand task, not old football tail.
        return row

    def collect(
        self, controls: int, *, until_ball_settled: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if type(controls) is not int or controls < 1 or self.controls + controls > 150:
            raise ValueError("bounded positive standing collection required")
        initial = snapshot(self.ep.env.data)
        rows = []
        for _ in range(controls):
            row = self.step()
            rows.append(row)
            q, v = self.ep.layout.canonical(self.ep.env.data)
            if row["body_failed"] or (until_ball_settled and q[38] <= 0.12 and abs(v[37]) <= 0.1):
                break
        raw = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        raw.update({"initial_" + k: v for k, v in initial.items()})
        raw.update({"terminal_" + k: v for k, v in snapshot(self.ep.env.data).items()})
        q, v = raw["qpos"][-1, -1], raw["qvel"][-1, -1]
        feature = body_feedback(q, v)
        healthy = not bool(raw["body_failed"].any())
        upright = float(1 - 2 * (q[4] ** 2 + q[5] ** 2))
        return raw, dict(
            task="static_neural_stand",
            memory_mode=self.memory_mode,
            controls=len(rows),
            native_steps=len(rows) * 10,
            neural_rows=len(rows),
            healthy=healthy,
            stability_observed=bool(
                healthy
                and q[2] >= 0.72
                and upright >= 0.98
                and np.linalg.norm(feature[4:6]) <= 0.2
                and np.linalg.norm(v[3:6]) <= 0.3
            ),
            terminal_feature=feature.tolist(),
            terminal_upright=upright,
            ball_settled=bool(q[38] <= 0.12 and abs(v[37]) <= 0.1),
            no_motion_padding=True,
            old_exam_result=False,
            optimizer_updates=0,
            hardware_authorized=False,
        )
