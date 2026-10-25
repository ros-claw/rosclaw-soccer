"""Own-policy kick exploration with physical PD-target commitment, SIM_ONLY.

The stochastic policy emits RAW action coordinates. The original deterministic
clip/map/PD codec is part of the environment; densities refer to the raw draws,
never to a clipped action mislabeled as a Gaussian sample. No optimizer here.
"""

import importlib
import math
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.correlated_motor_exploration import sample_action
from rosclaw_soccer.rsi.kick_prior_actor import KickPriorActor, TorchKickSession
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer


class OwnKickSession(TorchKickSession):
    def __init__(self, actor: KickPriorActor, *, seed: int, std: float = 0.015) -> None:
        if (
            type(seed) is not int
            or not 0 <= seed < 2**31
            or not math.isfinite(std)
            or (not 0.005 <= std <= 0.05)
        ):
            raise ValueError("explicit bounded seed and small raw-coordinate exploration required")
        super().__init__(actor)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.log_std = torch.full((29,), math.log(std), dtype=torch.float32)
        self.rho = 0.9
        self.previous_action: torch.Tensor | None = None
        self.previous_mean: torch.Tensor | None = None
        self.pending: dict[str, Any] | None = None
        self.samples: list[dict[str, Any]] = []
        self.faulted = False

    def run(self, outputs: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.faulted or self.pending is not None:
            raise ValueError("one proposal then actual physical commitment required")
        try:
            mean = torch.from_numpy(super().run(outputs, feed)[0][0])
            action, logp = sample_action(
                mean,
                self.log_std,
                rho=self.rho,
                previous_action=self.previous_action,
                previous_mean=self.previous_mean,
                generator=self.generator,
            )
            self.pending = dict(
                observation=feed["obs"][0].copy(),
                mean=mean.numpy().copy(),
                raw_action=action.numpy().copy(),
                raw_logp=float(logp),
            )
            return [action.numpy()[None].copy()]
        except BaseException:
            self.faulted = True
            raise

    def commit_executed(self, policy: Any, target: np.ndarray, *, frame: int) -> None:
        try:
            if (
                self.faulted
                or self.pending is None
                or type(frame) is not int
                or frame < 0
                or (self.samples and frame != self.samples[-1]["frame"] + 1)
            ):
                raise ValueError("one consecutive actual physical commitment required")
            module = importlib.import_module(type(policy).__module__)
            action = self.pending["raw_action"]
            clipped = np.clip(action, policy.action_clip_lo_il, policy.action_clip_hi_il)
            expected = (
                policy.default_q_mj + policy.action_scale_mj * clipped[module.MUJOCO_TO_ISAAC]
            )
            if target.shape != (29,) or not np.array_equal(expected.astype(np.float64), target):
                raise ValueError("raw draw did not produce the exact actually executed PD target")
            self.samples.append(
                dict(
                    **self.pending,
                    frame=frame,
                    executed_pd_target=target.copy(),
                    clipped_raw_action=clipped.copy(),
                    clipped_components=int(np.count_nonzero(clipped != action)),
                )
            )
            self.previous_action = torch.from_numpy(action.copy())
            self.previous_mean = torch.from_numpy(self.pending["mean"].copy())
            self.pending = None
        except BaseException:
            self.faulted = True
            raise


class OwnKickPlayer(SharedRoboNaldoPlayer):
    """Commit AR memory only after the shared world proves target execution."""

    def commit(self) -> None:
        try:
            super().commit()
            session = self.policy.ort_session
            if type(session) is not OwnKickSession:
                raise ValueError("explicit own-policy stochastic session required")
            session.commit_executed(
                self.policy,
                self.world.last_applied_target(self.prefix),
                frame=self.world.control_steps - 1,
            )
        except BaseException:
            self.failed = True
            raise
