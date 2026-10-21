"""Bounded temporal TWO-network teacher; never a trained single-policy claim."""

from typing import Any

import numpy as np
import torch
from torch import nn


def contact_weight(step: int, cutover: int) -> float:
    if (
        type(step) is not int
        or step < 0
        or type(cutover) is not int
        or cutover not in (140, 160, 190)
    ):
        raise ValueError("nonnegative causal step and predeclared cutover required")
    u = min(1.0, max(0.0, (step - (cutover - 12)) / 24))
    return 1.0 - (10 * u**3 - 15 * u**4 + 6 * u**5)


class RecoveryMixtureTeacher(nn.Module):
    """Smoothly blend residual proposals on SAME actual observation and memory.

    Reference-phase access is explicit teacher authority. No case ID, world
    reset, future observation, contact-force oracle, or repeated reference is
    used. Both actors are executed and counted even at weight 0/1. This is only
    a teacher feasibility experiment until independent physical audit passes.
    """

    composite_teacher = True

    def __init__(self, contact: Any, recovery: Any, *, cutover: int) -> None:
        super().__init__()
        contact_weight(0, cutover)
        a, b = contact.base.state_dict(), recovery.base.state_dict()
        if (
            set(a) != set(b)
            or any(not torch.equal(a[k], b[k]) for k in a)
            or not torch.equal(contact.action_scale, recovery.action_scale)
        ):
            raise ValueError("exact shared frozen base, normalization and action scale required")
        self.contact, self.recovery, self.cutover = contact, recovery, cutover
        self.requires_grad_(False)
        self.eval()
        self.begin_skill()

    def begin_skill(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def forward(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if (
            obs.shape != (1, 1, 160)
            or obs.device.type != "cpu"
            or obs.dtype != torch.float32
            or not torch.isfinite(obs).all()
        ):
            raise ValueError("one actual CPU160 causal observation required")
        new, ns = self.contact(obs, tuple(t.clone() for t in state))
        old, os = self.recovery(obs, tuple(t.clone() for t in state))
        if (
            any(not torch.equal(x, y) for x, y in zip(ns, os, strict=True))
            or new.shape != (1, 1, 29)
            or old.shape != new.shape
            or not torch.isfinite(new).all()
            or not torch.isfinite(old).all()
        ):
            raise ValueError("finite shared-memory own-network proposals required")
        step = len(self.rows)
        weight = contact_weight(step, self.cutover)
        mixed = new if weight == 1 else old if weight == 0 else old + weight * (new - old)
        self.rows.append(
            dict(
                step=step,
                contact_weight=weight,
                contact_proposal=new[0, 0].detach().numpy().copy(),
                recovery_proposal=old[0, 0].detach().numpy().copy(),
            )
        )
        return mixed, ns

    def trace(self) -> dict[str, np.ndarray]:
        if not self.rows:
            raise ValueError("teacher must execute before recording evidence")
        return {"TEACHER_" + k: np.array([r[k] for r in self.rows]) for k in self.rows[0]}
