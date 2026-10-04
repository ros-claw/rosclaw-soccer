"""Causal measured-state SONIC decoder inputs, not a policy or motor executor.

Supplied tokens are external prior commands, not reconstructed planner evidence.
Offline replay of stored targets is not a substitute for this body feedback.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_runup import (
    MUJOCO_TO_ISAACLAB,
    G1SonicRunupController,
)


class SonicDecoderHistory:
    """One independent instance per batch/episode; no implicit reset or world IO."""

    def __init__(self, lanes: int) -> None:
        if type(lanes) is not int or not 1 <= lanes <= 4096:
            raise ValueError("bounded integer lane count required")
        self._lanes = lanes
        self._next_frame = 0
        self._history: list[tuple[np.ndarray[Any, Any], ...]] = []
        self._default = np.asarray(G1SonicRunupController.default_angles, dtype=np.float32)

    def observe(
        self,
        frame: int,
        qpos: Any,
        qvel: Any,
        prior_token: Any,
        previous_raw_action_isaac: Any,
    ) -> np.ndarray[Any, Any]:
        """Append current measured state and previous action, then own 994 inputs.

        Caller must bind tokens and actions to their original source. This helper
        does not validate weights, predict events, learn or authorize execution.
        """
        if type(frame) is not int or frame != self._next_frame or not 0 <= frame < 20000:
            raise ValueError("consecutive measured decoder frames required")
        q, v, token, action = [
            np.array(x, dtype=np.float32, copy=True)
            for x in (qpos, qvel, prior_token, previous_raw_action_isaac)
        ]
        if (
            (q.shape, v.shape, token.shape, action.shape)
            != ((self._lanes, 43), (self._lanes, 41), (self._lanes, 64), (self._lanes, 29))
            or not all(np.isfinite(x).all() for x in (q, v, token, action))
            or not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-4, rtol=0)
            or (frame == 0 and np.any(action != 0))
        ):
            raise ValueError("finite aligned body/token/action and clean episode start required")
        w, x, y, z = (q[:, i] for i in range(3, 7))
        gravity = np.stack(
            (2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1), axis=1
        )
        entry = (
            v[:, 3:6].copy(),
            (q[:, 7:36] - self._default)[:, MUJOCO_TO_ISAACLAB],
            v[:, 6:35][:, MUJOCO_TO_ISAACLAB].copy(),
            action,
            gravity,
        )
        history = self._history + [entry] if self._history else [entry] * 10
        history = history[-10:]
        result: np.ndarray[Any, Any] = np.concatenate(
            (
                token,
                *[
                    np.stack([e[i] for e in history], axis=1).reshape(self._lanes, -1)
                    for i in range(5)
                ],
            ),
            axis=1,
        )
        if result.shape != (self._lanes, 994) or not np.isfinite(result).all():
            raise ValueError("invalid measured decoder input")
        # Publish state only after complete input validation; failures don't eat frames.
        self._history = history
        self._next_frame += 1
        result.flags.writeable = False
        return result
