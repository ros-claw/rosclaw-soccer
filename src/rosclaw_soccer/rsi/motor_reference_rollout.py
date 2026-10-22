"""Fresh actual frozen-parent reference/standing sequences for rehearsal.

No learner hidden-state labels, teacher forcing during physics, state resets,
synthetic tail, hardware path or inference from a rendered video.
"""

from typing import Any

import numpy as np
import torch


def collect_parent_reference(
    ep: Any, actor: Any, *, foot: str, hidden_width: int = 128
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, dict[str, Any] | None, int]:
    from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
    from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
    from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

    if (
        ep.frame != 0
        or ep.done
        or ep.ready
        or actor.training
        or any(p.requires_grad or p.device.type != "cpu" for p in actor.parameters())
    ):
        raise ValueError("fresh declared native world and frozen parent motor required")
    initial = snapshot(ep.env.data)
    if hidden_width == 256:
        from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor

        if type(actor) is not DualRecurrentMotor:
            raise ValueError("explicit typed dual256 motor required")
        state = actor.initial_state()
    elif hidden_width == 128:
        state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    else:
        raise ValueError("explicit known private motor history contract required")
    identity = FootTaskSpaceIntent(
        ep.env.model, foot=foot, center=105, latent=np.zeros((2, 6), np.float64)
    )
    rows = []
    calls = 0
    while not ep.done and ep.frame < ep.clip.length:
        obs = ep.observation()
        before = [t.numpy().copy() for t in state]
        with torch.no_grad():
            mean, state = actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
        calls += 1
        action = mean[0, 0].numpy().copy()
        target, plan = identity(ep, action.copy())
        if not np.array_equal(action, target) or plan["ik_iterations"] != 0:
            raise ValueError("identity record cannot change parent motor control")
        row = ep.step(action)
        row.update(
            NN_obs=obs,
            NN_nominal_action=action,
            NN_action=action.copy(),
            NN_h_before=before[0],
            NN_c_before=before[1],
            NN_h_after=state[0].numpy().copy(),
            NN_c_after=state[1].numpy().copy(),
        )
        row.update({"PLAN_" + k: v for k, v in plan.items()})
        rows.append(row)
        if ep.frame == ep.clip.length and not ep.done:
            ep.admit_reference_recovery()
    raw = {k: np.array([v[k] for v in rows]) for k in rows[0]}
    raw.update({"initial_" + k: v for k, v in initial.items()})
    raw.update({"terminal_" + k: v for k, v in snapshot(ep.env.data).items()})
    stand = summary = None
    if ep.termination_reason == "RECOVERY_ADMITTED":

        def count() -> None:
            nonlocal calls
            calls += 1

        stand, summary = NeuralStandingTask(
            ep,
            actor,
            state=state,
            memory_mode="carry",
            count_inference=count,
            hidden_width=hidden_width,
        ).collect(100)
    return raw, stand, summary, calls
