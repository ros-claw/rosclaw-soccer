"""Bounded shared-world held-target physical lookahead; not a soccer policy.

Plans are scratch simulations from one current integration state. Held targets
are explicitly frozen and cannot stand in for neural closed-loop outcomes.
"""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld


def lookahead_candidates(
    model: Any,
    measured: Any,
    *,
    players: dict[str, NativePDParameters],
    ball_body_name: str,
    targets: dict[str, np.ndarray],
    receiver: str,
    corrections: np.ndarray,
    horizon_controls: int,
) -> list[dict[str, Any]]:
    """Evaluate bounded candidate offsets in cloned native multi-body physics.

    The complete targets for every player are held fixed for the horizon;
    there is no replay of future controls and no neural calls. Initial world is
    never mutated. Results are diagnostics, not labels or promotion evidence.
    """
    if (
        type(players) is not dict
        or len(players) != 2
        or receiver not in players
        or set(targets) != set(players)
        or any(
            not isinstance(v, np.ndarray) or v.shape != (29,) or not np.isfinite(v).all()
            for v in targets.values()
        )
        or type(corrections) is not np.ndarray
        or corrections.ndim != 2
        or corrections.shape[1] != 29
        or not 1 <= len(corrections) <= 32
        or not np.isfinite(corrections).all()
        or np.max(np.abs(corrections)) > 0.12
        or np.any(corrections[:, 12:] != 0)
        or type(horizon_controls) is not int
        or not 1 <= horizon_controls <= 50
    ):
        raise ValueError("bounded measured two-player lookahead inputs required")
    rows = []
    for candidate, delta in enumerate(corrections):
        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, measured)
        world = SharedNativeMotorWorld(
            model, scratch, players=players, ball_body_name=ball_body_name
        )
        proposed = {p: v.astype(np.float64, copy=True) for p, v in targets.items()}
        proposed[receiver] += delta
        player_index = list(players).index(receiver)
        foot_first = None
        nonfoot_first = None
        peak_nonfoot = 0.0
        for control in range(horizon_controls):
            step = world.advance(proposed)
            forces = step["ball_forces"][:, player_index, :]
            if foot_first is None and np.any(forces[:, :2] > 1):
                foot_first = control * 10 + int(
                    np.flatnonzero(np.any(forces[:, :2] > 1, axis=1))[0]
                )
            if nonfoot_first is None and np.any(forces[:, 2] > 1):
                nonfoot_first = control * 10 + int(np.flatnonzero(forces[:, 2] > 1)[0])
            peak_nonfoot = max(peak_nonfoot, float(forces[:, 2].max()))
            if world.failed:
                break
        layout = world.layouts[receiver]
        q, v = layout.canonical(scratch)
        rows.append(
            dict(
                candidate=candidate,
                native_steps=world.native_steps,
                receiver_foot_first=foot_first,
                receiver_nonfoot_first=nonfoot_first,
                maximum_nonfoot_force_n=peak_nonfoot,
                body_failed=bool(world.failed),
                final_ball_relative_xy_m=(q[36:38] - q[:2]).tolist(),
                final_ball_relative_velocity_xy_mps=(v[35:37] - v[:2]).tolist(),
                held_targets_only=True,
                neural_calls=0,
                physically_qualified=False,
            )
        )
    return rows
