"""Reproducible stochastic intention sampling and native-only reward observation.

Adapters reuse the already audited whole H300 rollout/replay. Prefix sampling
warms memory without consuming RNG; independent replay regenerates all noise.
"""

import numpy as np

from rosclaw_soccer.rsi.native_foot_reward import NativeFootRewardObserver, NearFootReward
from scripts.rsi_motor_student_rollout import exact_array


class SamplingPolicy:
    def __init__(self, policy, *, seed, stochastic):
        import torch

        self.policy = policy
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.stochastic = stochastic
        self.frame = 0
        self.rows = []

    def __call__(self, x, hidden):
        import torch

        if self.frame >= 300 or x.shape != (1, 1, 108):
            raise ValueError("one sequential owned H300 sampling stream required")
        with torch.no_grad():
            row = self.policy.sample(
                x,
                hidden,
                generator=self.generator,
                stochastic=self.stochastic and self.frame >= 20,
            )
        self.rows.append(
            {
                k: row[k][0, 0].cpu().numpy().copy()
                for k in ("mean", "std", "noise", "raw", "log_prob", "entropy")
            }
        )
        self.frame += 1
        return row["token"], row["value"], row["hidden"]

    def recording(self):
        if self.frame != 300:
            raise ValueError("complete sampling record required")
        return {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}

    def network_call_counts(self):
        if self.frame != 300:
            raise ValueError("complete policy sampling stream required")
        method = getattr(self.policy, "network_call_counts", None)
        return method() if callable(method) else dict(intention_network_calls=300)


class RewardedBody:
    def __init__(self, body):
        self.body = body
        self.observer = NativeFootRewardObserver(body)
        self.reward = None
        self.rewards = np.zeros(300, dtype=np.float32)
        self.terms = []

    def __getattr__(self, key):
        return getattr(self.body, key)

    def step(self, frame, target):
        if frame < 20:
            return self.body.step(frame, target)
        if self.reward is None:
            self.reward = NearFootReward(initial_ball_y_m=float(self.body.canonical()[0][37]))
        gap_before = self.observer.gap(self.body)
        row = self.body.step(frame, target)
        evidence = self.observer.evidence(
            row, frame=frame, gap_before=gap_before, gap_after=self.observer.gap(self.body)
        )
        reward, terms = self.reward.step(evidence)
        self.rewards[frame] = reward
        self.terms.append(terms)
        return row

    def finish(self, score, diagnostics):
        if self.reward is None:
            raise ValueError("native suffix reward evidence required")
        self.rewards[-1] += self.reward.terminal(
            score,
            causal_first_touch_possible=diagnostics["post_boundary_control_can_cause_first_touch"],
        )
        return self.rewards.copy()


def checked_private_gap(observer, body):
    """Preflight proves the distance query does not mutate live solver state."""
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mujoco.mj_stateSize(body.model, spec))
    after = np.empty_like(before)
    mujoco.mj_getState(body.model, body.data, before, spec)
    gap = observer.gap(body)
    mujoco.mj_getState(body.model, body.data, after, spec)
    exact_array(before, after)
    return gap
