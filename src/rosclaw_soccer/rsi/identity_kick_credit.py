"""SIM training credit from identity-bound foot forces, never nearest-foot guesses.

Inputs are chronological physics-substep force magnitudes for LEFT/RIGHT ball
contacts. The sensor binding must be qualified independently. Reward is not a
skill certificate; no contact, wrong-foot-first or ambiguous first contact earns
no positive kick credit. Episode resets are explicit.
"""

import torch


class IdentityKickCredit:
    def __init__(self, worlds: int, device: str):
        if type(worlds) is not int or not 1 <= worlds <= 512:
            raise ValueError("bounded explicit world count required")
        self.worlds = worlds
        self.seen = torch.zeros(worlds, dtype=torch.bool, device=device)
        self.remaining = torch.zeros(worlds, dtype=torch.long, device=device)
        self.last_step: int | None = None
        self.cached: dict[str, torch.Tensor] | None = None
        self.forbidden = torch.zeros(worlds, dtype=torch.bool, device=device)
        self._uses_forbidden: bool | None = None

    def reset(self, ids: torch.Tensor) -> None:
        if ids.dtype != torch.long or ids.ndim != 1 or ((ids < 0) | (ids >= self.worlds)).any():
            raise ValueError("explicit valid reset world indices required")
        self.seen[ids] = False
        self.remaining[ids] = 0
        self.forbidden[ids] = False

    def step(
        self,
        *,
        step: int,
        forces: torch.Tensor,
        expected: torch.Tensor,
        ball_velocity: torch.Tensor,
        goal_delta: torch.Tensor,
        forbidden_forces: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if type(step) is not int or step < 0:
            raise ValueError("monotonic explicit control step required")
        if self.last_step == step:
            if self._uses_forbidden != (forbidden_forces is not None):
                raise ValueError("contact intent cannot change within a reward step")
            assert self.cached is not None
            return self.cached
        if self.last_step is not None and step != self.last_step + 1:
            raise ValueError("no skipped or replayed reward controls")
        b = self.worlds
        if (
            forces.shape != (b, 10, 2)
            or not forces.is_floating_point()
            or expected.shape != (b,)
            or expected.dtype not in (torch.int8, torch.int32, torch.int64)
            or ((expected < 0) | (expected > 1)).any()
            or ball_velocity.shape != (b, 3)
            or goal_delta.shape != (b, 3)
            or any(not torch.isfinite(v).all() for v in (forces, ball_velocity, goal_delta))
            or (forces < 0).any()
            or any(
                v.device != self.seen.device for v in (forces, expected, ball_velocity, goal_delta)
            )
        ):
            raise ValueError("finite aligned chronological actual two-foot force inputs required")
        uses_forbidden = forbidden_forces is not None
        if self._uses_forbidden is not None and self._uses_forbidden != uses_forbidden:
            raise ValueError("contact intent cannot change within an environment lifetime")
        if forbidden_forces is not None and (
            forbidden_forces.shape != (b, 10)
            or not forbidden_forces.is_floating_point()
            or forbidden_forces.device != self.seen.device
            or not torch.isfinite(forbidden_forces).all()
            or (forbidden_forces < 0).any()
        ):
            raise ValueError("finite aligned actual forbidden-body contact history required")
        self._uses_forbidden = uses_forbidden
        newly_forbidden = torch.zeros_like(self.seen)
        if forbidden_forces is not None:
            bad = (forbidden_forces > 1.0).any(dim=1)
            newly_forbidden = bad & ~self.forbidden
            self.forbidden |= bad
        hit = forces > 10.0
        times = hit.any(dim=2)
        first = times.to(torch.int64).argmax(dim=1)
        at_first = hit[torch.arange(b, device=forces.device), first]
        new = times.any(dim=1) & ~self.seen
        single = at_first.sum(dim=1) == 1
        identity_correct = new & single & at_first.gather(1, expected.long()[:, None]).squeeze(1)
        correct = identity_correct & ~self.forbidden
        wrong = new & ~identity_correct
        self.seen |= new
        self.remaining[correct] = 5
        self.remaining[self.forbidden] = 0
        speed = ball_velocity.norm(dim=1)
        distance = goal_delta.norm(dim=1)
        active = (self.remaining > 0) & (speed >= 0.5) & (distance > 1e-6)
        cosine = (ball_velocity * goal_delta).sum(dim=1) / (speed * distance).clamp_min(1e-6)
        self.cached = dict(
            contact=correct.float(),
            wrong_contact=wrong.float(),
            alignment=torch.where(active, cosine.clamp(0, 1), 0),
            speed=torch.where(active, 1 - torch.exp(-speed.square() / 1.2**2), 0),
        )
        if uses_forbidden:
            self.cached["forbidden_contact"] = newly_forbidden.float()
        self.remaining = (self.remaining - 1).clamp_min(0)
        self.last_step = step
        return self.cached
