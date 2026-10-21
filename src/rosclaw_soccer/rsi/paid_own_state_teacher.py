"""Explicit SIM research handover and counterfactual IK labels on OWN states.

The student always computes its own recurrent proposal. The collection host
provides current context; a separately declared training teacher can replace
the executed action. No hidden actor latch, borrowed history, or reset exists.
"""

from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_causal_context import CausalMotorContext
from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
from rosclaw_soccer.rsi.paid_trainable_actor import PaidTrainableActor


def handoff_controls(center: int, horizon: int) -> dict[str, int]:
    if (
        type(center) is not int
        or not 43 <= center <= 200
        or type(horizon) is not int
        or not center + 38 < horizon <= 300
    ):
        raise ValueError("original teacher phase and actual bounded horizon required")
    return dict(
        teacher=0, phase_start=center - 42, middle=center - 24, late=center - 8, student=horizon
    )


class OwnStateTeacherHandoff:
    def __init__(
        self,
        episode: Any,
        base: PaidTrainableActor,
        *,
        intent: FootTaskSpaceIntent,
        switch_control: int,
    ) -> None:
        if (
            type(base) is not PaidTrainableActor
            or base.training
            or any(p.requires_grad or p.device.type != "cpu" for p in base.parameters())
            or intent.model is not episode.env.model
            or type(switch_control) is not int
            or not 0 <= switch_control <= episode.clip.length <= 300
        ):
            raise ValueError("frozen own CPU R2, same SIM model and declared boundary required")
        self.episode, self.base, self.intent, self.switch_control = (
            episode,
            base,
            intent,
            switch_control,
        )
        # These are transparent compute counters, never inputs to action choice.
        self.r2_head_queries = self.ik_queries = self.teacher_rows = self.student_rows = 0

    def __call__(
        self, ep: Any, nominal: np.ndarray, context: CausalMotorContext
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if ep is not self.episode or type(context) is not CausalMotorContext:
            raise ValueError("same player's own host context required")
        context.assert_current(ep)
        if nominal.shape != (29,) or nominal.dtype != np.float32 or not np.isfinite(nominal).all():
            raise ValueError("finite actual student proposal required")
        with torch.no_grad():
            hidden = torch.from_numpy(context.h_after[-1, 0].copy()).reshape(1, 1, 128)
            r2_nominal = self.base.actor(hidden)[0, 0].numpy().copy()
        self.r2_head_queries += 1
        student_delta = (nominal[ep.order] - r2_nominal[ep.order]) * ep.scale
        if not np.isfinite(student_delta).all() or np.abs(student_delta).max() > 0.20002:
            raise ValueError("student exceeded original R2-relative authority")
        teacher, plan = self.intent(ep, r2_nominal)
        self.ik_queries += 1
        teacher_delta = (teacher[ep.order] - r2_nominal[ep.order]) * ep.scale
        if not np.isfinite(teacher_delta).all() or np.abs(teacher_delta).max() > 0.20002:
            raise ValueError("expert exceeded original R2-relative authority")
        applied = context.frame >= self.switch_control
        self.teacher_rows += int(applied)
        self.student_rows += int(not applied)
        return (teacher.copy() if applied else nominal.copy()), dict(
            **plan,
            R2_nominal_action=r2_nominal,
            teacher_action=teacher.copy(),
            teacher_applied=applied,
            switch_control=self.switch_control,
            context_frame=context.frame,
        )
