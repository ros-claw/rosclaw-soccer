"""Private bounded stochastic motor stream; no physics or model update.

Every motion action is actually sampled from the existing normalized truncated
AR law. Legacy v1 recovery is deterministic. Explicit v2 also samples recovery,
continuing the SAME private AR history and RNG without a phase-boundary reset.
Seed and consecutive own observations suffice to regenerate all draws; saved
actions/hidden states are not inputs. One episode per instance, no reset API.
"""

import hashlib
import math
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.motor_exploration_profile import exploration_sigma
from rosclaw_soccer.rsi.motor_sampling_law import draw_motor_action


class SharedMotorSampling:
    def __init__(
        self, scale_policy: np.ndarray, *, seed: int, profile: str, include_recovery: bool = False
    ) -> None:
        if (
            type(seed) is not int
            or type(include_recovery) is not bool
            or not 0 <= seed < 2**31
            or scale_policy.shape != (29,)
            or scale_policy.dtype != np.float32
            or not np.isfinite(scale_policy).all()
            or not (scale_policy > 0).all()
        ):
            raise ValueError("explicit seed and unchanged positive29 policy scales required")
        self.sigma = exploration_sigma(profile, "box-truncated-ar1")
        self.seed, self.profile = seed, profile
        self.include_recovery = include_recovery
        self.scale = torch.from_numpy(scale_policy.copy())
        self.log_std = torch.full((29,), math.log(self.sigma), dtype=torch.float32)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.previous_action: torch.Tensor | None = None
        self.previous_mean: torch.Tensor | None = None
        self.next_control = self.stochastic_controls = 0
        self.recovery_started = False
        self.motion_controls = self.recovery_controls = 0

    def _rng_hash(self) -> np.ndarray:
        digest = hashlib.sha256(self.generator.get_state().numpy().tobytes()).digest()
        return np.frombuffer(digest, dtype=np.uint8).copy()

    def propose(
        self, mean: np.ndarray, reference: np.ndarray, *, control: int, standing: bool
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if (
            type(control) is not int
            or control != self.next_control
            or not 0 <= control < 400
            or type(standing) is not bool
            or (not standing and (self.recovery_started or control >= 300))
            or (standing and (self.motion_controls == 0 or self.recovery_controls >= 100))
            or any(
                x.shape != (29,) or x.dtype != np.float32 or not np.isfinite(x).all()
                for x in (mean, reference)
            )
        ):
            raise ValueError("one consecutive actual own motion/recovery observation required")
        before = self._rng_hash()
        active = not standing or self.include_recovery
        if standing:
            self.recovery_started = True
            self.recovery_controls += 1
        else:
            self.motion_controls += 1
        if not active:
            action, logp = mean.copy(), np.array(0.0, dtype=np.float32)
        else:
            m, r = torch.from_numpy(mean.copy()), torch.from_numpy(reference.copy())
            with torch.no_grad():
                sampled, density = draw_motor_action(
                    "box-truncated-ar1",
                    m,
                    r,
                    self.scale,
                    self.log_std,
                    rho=0.9,
                    previous_action=self.previous_action,
                    previous_mean=self.previous_mean,
                    generator=self.generator,
                )
            action, logp = sampled.numpy().copy(), density.numpy().copy()
            self.previous_action, self.previous_mean = sampled.clone(), m.clone()
            self.stochastic_controls += 1
        self.next_control += 1
        return action, dict(
            active=np.array(active),
            logp=logp,
            rng_before=before,
            rng_after=self._rng_hash(),
        )

    def metadata(self, *, reference_controls: int, recovery_controls: int) -> dict[str, Any]:
        """Producer provenance, NOT admission to an optimizer."""
        if (
            type(reference_controls) is not int
            or not self.motion_controls <= reference_controls <= 300
            or type(recovery_controls) is not int
            or not 0 <= recovery_controls <= 100
            or recovery_controls != self.recovery_controls
            or self.next_control != self.motion_controls + recovery_controls
            or (recovery_controls and self.motion_controls != reference_controls)
            or self.stochastic_controls
            != self.motion_controls + (recovery_controls if self.include_recovery else 0)
        ):
            raise ValueError("actual declared motion/recovery counts required")
        return dict(
            controller_kind=(
                "pure-refinement-full-recovery-stochastic-native-v2"
                if self.include_recovery
                else "pure-refinement-stochastic-native-v1"
            ),
            sampling_law="box-truncated-ar1",
            sampling_profile=self.profile,
            sampling_seed=self.seed,
            stochastic_controls=self.stochastic_controls,
            motion_reference_controls=reference_controls,
            observed_recovery_controls=recovery_controls,
            total_observed_controls=self.next_control,
            deterministic_recovery_excluded_from_actor_rows=not self.include_recovery,
            PPO_actor_rows=0,
            requires_whole_native_and_law_audit=True,
        )
