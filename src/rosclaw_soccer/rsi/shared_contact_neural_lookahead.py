"""SIM-only shared-world branch with causal receiver neural reinference."""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import observation98, target29
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld


def neural_lookahead(
    model: Any,
    measured: Any,
    *,
    players: dict[str, NativePDParameters],
    ball_body_name: str,
    receiver: str,
    passer_target: np.ndarray,
    policy: Any,
    last_action: np.ndarray,
    phase: np.float32,
    command: np.ndarray,
    corrections: np.ndarray,
    horizon_controls: int,
) -> list[dict[str, Any]]:
    """Roll out frozen receiver actor on branch-local measured state.

    The passer target and receiver body-frame command remain held. Previous raw
    NN actions, not corrected targets, feed the actor's history. This is not a
    resumed passer policy, a teacher certificate or a physical match result.
    """
    if (
        type(players) is not dict
        or len(players) != 2
        or receiver not in players
        or type(passer_target) is not np.ndarray
        or passer_target.shape != (29,)
        or not np.isfinite(passer_target).all()
        or type(last_action) is not np.ndarray
        or last_action.shape != (29,)
        or last_action.dtype != np.float32
        or not np.isfinite(last_action).all()
        or type(phase) is not np.float32
        or not 0 <= phase < 1
        or type(command) is not np.ndarray
        or command.shape != (3,)
        or command.dtype != np.float32
        or not np.isfinite(command).all()
        or type(corrections) is not np.ndarray
        or corrections.ndim != 2
        or corrections.shape[1] != 29
        or not 1 <= len(corrections) <= 32
        or not np.isfinite(corrections).all()
        or np.max(abs(corrections)) > 0.12
        or np.any(corrections[:, 12:] != 0)
        or type(horizon_controls) is not int
        or not 1 <= horizon_controls <= 50
    ):
        raise ValueError("bounded shared state, private actor history and targets required")
    passer = next(p for p in players if p != receiver)
    results = []
    for candidate, delta in enumerate(corrections):
        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, measured)
        world = SharedNativeMotorWorld(
            model, scratch, players=players, ball_body_name=ball_body_name
        )
        layout = world.layouts[receiver]
        index = list(players).index(receiver)
        action, gait = last_action.copy(), phase
        foot_first = nonfoot_first = None
        failure: list[str] = []
        for tick in range(horizon_controls):
            mujoco.mj_forward(model, scratch)
            q, velocity = layout.canonical(scratch)
            obs, gait = observation98(
                q,
                velocity,
                layout.angular_velocity_body(scratch),
                command,
                action,
                gait,
                policy.params,
            )
            action = policy.session.run(["actions"], {"obs": obs[None]})[0][0].copy()
            target = target29(action, policy.params) + delta
            sample = world.advance({passer: passer_target, receiver: target})
            forces = sample["ball_forces"][:, index, :]
            if foot_first is None and np.any(forces[:, :2] > 1):
                foot_first = tick * 10 + int(np.flatnonzero(np.any(forces[:, :2] > 1, axis=1))[0])
            if nonfoot_first is None and np.any(forces[:, 2] > 1):
                nonfoot_first = tick * 10 + int(np.flatnonzero(forces[:, 2] > 1)[0])
            if world.failed:
                failure = [p for p, failed in sample["body_failed"].items() if failed]
                break
        q, v = layout.canonical(scratch)
        results.append(
            dict(
                candidate=candidate,
                native_steps=world.native_steps,
                neural_calls=world.control_steps,
                receiver_foot_first=foot_first,
                receiver_nonfoot_first=nonfoot_first,
                failed_players=failure,
                final_ball_relative_xy_m=(q[36:38] - q[:2]).tolist(),
                final_ball_relative_velocity_xy_mps=(v[35:37] - v[:2]).tolist(),
                passer_neural_resume=False,
                physically_qualified=False,
            )
        )
    return results
