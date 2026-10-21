"""Reusable SIM three-skill trial with an explicit reference-end recovery.

No task/role/case winner lookup. Every actor has private h0 per invocation;
all body dynamics and actual previous actions carry across boundaries.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_balance_recovery import collect_balance_recovery
from rosclaw_soccer.rsi.paid_practice_chain import advance_practice_boundary
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot


def run_recovery_trial(
    bench: Any,
    case: dict[str, Any],
    entry_actor: Any,
    successor_actor: Any,
    latent: np.ndarray,
    *,
    store: Callable[[str, dict[str, np.ndarray]], dict[str, Any]],
) -> dict[str, Any]:
    """First complete -> second reference -> recovery -> third complete.

    Caller storage MUST check qualified first/second prefix arrays before new
    recovery physics. Ordinary recovery/continuation failures are retained.
    """
    if latent.shape != (9, 4) or not np.isfinite(latent).all() or np.abs(latent).max() > 1:
        raise ValueError("one finite bounded common feedback model required")
    source = case["source"]
    if ("left" in source) == ("right" in source):
        raise ValueError("one admitted reference foot required")
    ep = bench.make(case)
    start_native, start_neural = bench.native_steps, bench.neural_rows
    skills, boundaries = [], []
    recovery = None
    status = "STARTED"
    foot = "left" if "left" in source else "right"

    def skill(index: int, actor: Any, *, prefix: bool = False) -> dict[str, Any]:
        row: dict[str, Any]
        raw, row = bench.collect_current(ep, actor, stop_at_reference=prefix)
        if any(k.startswith(("PLAN_", "TEACHER_")) for k in raw) or not np.array_equal(
            raw["NN_action"], raw["NN_nominal_action"]
        ):
            raise ValueError("kick motion must come from the declared own learned network")
        result = dict(row, **store(f"skill{index}", raw))
        terminal = store(f"terminal{index}", snapshot(ep.env.data))
        skills.append(dict(index=index, result=result, terminal=terminal))
        return row

    first = skill(0, entry_actor)
    if first["body_failed"]:
        status = "FIRST_BODY_FAILURE"
    else:
        boundary = advance_practice_boundary(
            ep, ep.clip, foot=foot, store=store, prefix="between0-1"
        )
        boundaries.append(boundary)
        if boundary["status"] != "READY":
            status = "FIRST_BOUNDARY_" + boundary["status"]
        else:
            second = skill(1, successor_actor, prefix=True)
            if second["body_failed"]:
                status = "SECOND_REFERENCE_BODY_FAILURE"
            else:
                before = snapshot(ep.env.data)
                raw, row = collect_balance_recovery(ep, latent)
                raw.update({"initial_" + k: v for k, v in before.items()})
                raw.update({"terminal_" + k: v for k, v in snapshot(ep.env.data).items()})
                recovery = dict(row, **store("recovery", raw))
                if not row["healthy"]:
                    status = "RECOVERY_BODY_FAILURE"
                else:
                    boundary = advance_practice_boundary(
                        ep, ep.clip, foot=foot, store=store, prefix="post-recovery"
                    )
                    boundaries.append(boundary)
                    if boundary["status"] != "READY":
                        status = "RECOVERY_CONTINUATION_" + boundary["status"]
                    else:
                        third = skill(2, successor_actor)
                        status = (
                            "THIRD_BODY_FAILURE"
                            if third["body_failed"]
                            else "THREE_SKILLS_WITH_RECOVERY_EXECUTED"
                        )
    if ep.native_steps != bench.native_steps - start_native:
        raise ValueError("single-world native work differs from shared bench accounting")
    if not np.isclose(float(ep.env.data.time), ep.native_steps * 0.002, rtol=0, atol=1e-10):
        raise ValueError("continuous-world simulation clock differs from actual native work")
    return dict(
        status=status,
        skills=skills,
        boundaries=boundaries,
        recovery=recovery,
        stable=status == "THREE_SKILLS_WITH_RECOVERY_EXECUTED",
        native_steps=ep.native_steps,
        neural_rows=bench.neural_rows - start_neural,
        final_sim_time=float(ep.env.data.time),
        body_initializations=1,
        body_resets=0,
        actual_previous_action_carried=True,
        private_h0_each_skill=True,
        new_recovery_phase_not_old_exam=True,
        hardware_authorized=False,
        team_game=False,
    )
