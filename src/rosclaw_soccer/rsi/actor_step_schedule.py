"""Explicit post-snapshot actor LR schedule; never resets weights or Adam state."""

import math
from typing import Any


def apply_actor_step_schedule(optimizer: Any, profile: str) -> dict[str, Any]:
    groups = optimizer.param_groups
    if len(groups) != 2 or profile not in ("inherit", "recover-base-lr"):
        raise ValueError("explicit reviewed two-group actor step schedule required")
    before = groups[0]["lr"]
    if not math.isfinite(before) or not 0 < before <= 3e-5:
        raise ValueError("bounded inherited actor rate required")
    after = 3e-5 if profile == "recover-base-lr" else before
    groups[0]["lr"] = after
    return dict(
        profile=profile,
        initial_actor_lr=before,
        scheduled_actor_lr=after,
        applied_after_initial_checkpoint=True,
        optimizer_state_reset=False,
    )


def validate_actor_step_schedule(
    initial: dict[str, Any],
    final: dict[str, Any],
    epochs: list[dict[str, Any]],
    schedule: dict[str, Any],
    profile: str,
) -> None:
    if profile not in ("inherit", "recover-base-lr") or len(epochs) != 4:
        raise ValueError("four bounded scheduled actor-critic epochs required")
    rates = [g["lr"] for g in initial["param_groups"]]
    if (
        len(rates) != 2
        or any(type(r) not in (int, float) or not math.isfinite(r) for r in rates)
        or not 0 < rates[0] <= 3e-5
        or not 0 < rates[1] <= 0.001
    ):
        raise ValueError("bounded actual inherited optimizer rates required")
    target = 3e-5 if profile == "recover-base-lr" else rates[0]
    if schedule != dict(
        profile=profile,
        initial_actor_lr=rates[0],
        scheduled_actor_lr=target,
        applied_after_initial_checkpoint=True,
        optimizer_state_reset=False,
    ):
        raise ValueError("exact post-initial actor-only schedule commitment required")
    rates[0] = target
    for epoch in epochs:
        if epoch["learning_rates_before"] != rates:
            raise ValueError("learning rates must continue from the previous accepted step")
        decision = epoch["decision"]
        attempts = decision["attempts"]
        if not 1 <= len(attempts) <= 8 or type(decision["applied"]) is not bool:
            raise ValueError("bounded actual trust-region attempts required")
        for index, attempt in enumerate(attempts):
            accepted = attempt["accepted"]
            kl = attempt["kl"]
            valid_kl = type(kl) in (float, int) and math.isfinite(kl) and 0 <= kl <= 0.015
            if (
                attempt["scale"] != 0.5**index
                or type(accepted) is not bool
                or accepted != valid_kl
                or accepted != (decision["applied"] and index == len(attempts) - 1)
                or (
                    accepted
                    and (
                        type(kl) not in (float, int)
                        or not math.isfinite(kl)
                        or not 0 <= kl <= 0.015
                    )
                )
            ):
                raise ValueError("scheduled rates cannot relax the existing KL trust region")
        if not decision["applied"] and len(attempts) != 8:
            raise ValueError("a rejected step must exhaust its original bounded attempts")
        if decision["applied"]:
            rates = [r * attempts[-1]["scale"] for r in rates]
        if epoch["learning_rates_after"] != rates:
            raise ValueError("actual accepted backtracking rate accounting differs")
    if [g["lr"] for g in final["param_groups"]] != rates:
        raise ValueError("final optimizer must preserve the last actual effective rates")
