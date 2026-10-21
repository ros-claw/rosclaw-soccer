"""SIM lifecycle: reference-prefix kicks and continuously observed idle tasks."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_balance_recovery import collect_balance_recovery
from rosclaw_soccer.rsi.paid_motor_schedule import lifecycle_motor_schedule
from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
from rosclaw_soccer.rsi.paid_practice_chain import require_identical_snapshot
from rosclaw_soccer.rsi.paid_skill_handoff import practice_ball_drop, snapshot


def run_standing_lifecycle(
    bench: Any,
    case: dict[str, Any],
    actor: Any,
    *,
    variant: int,
    skills: int,
    store: Callable[[str, dict[str, np.ndarray]], dict[str, Any]],
    record_initial: bool = False,
    motor_schedule: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Variants: fixed-target/R2, neural-carry/R2, neural-carry/R6 successor.

    First NN kick prefix is caller-verified before any new standing physics.
    After that, worlds contain genuinely NEW successive states, not purported
    exact reproductions of an old second-kick proof.
    """
    if (
        type(variant) is not int
        or variant not in (0, 1, 2)
        or type(skills) is not int
        or skills not in (1, 3)
        or type(record_initial) is not bool
    ):
        raise ValueError("predeclared lifecycle variant and one/three skills required")
    schedule = lifecycle_motor_schedule(variant, skills, motor_schedule)
    source = case["source"]
    if ("left" in source) == ("right" in source):
        raise ValueError("one admitted reference foot required")
    ep = bench.make(case)
    initial = store("world-initial", snapshot(ep.env.data)) if record_initial else None
    start_native, start_neural = bench.native_steps, bench.neural_rows
    kicks: list[dict[str, Any]] = []
    stands: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    status = "STARTED"
    foot = "left" if "left" in source else "right"

    def inference() -> None:
        with bench.lock:
            bench.neural_rows += 1

    for k in range(skills):
        if k:
            before = snapshot(ep.env.data)
            previous = ep.last_action.copy()
            try:
                ep.begin_current_state(ep.clip, goal=ep.goal.copy())
            except ValueError as error:
                if str(error) != "initial ball/body overlap; no resampling":
                    raise
                status = "HANDOFF_CONTACT_BLOCKED"
                require_identical_snapshot(before, snapshot(ep.env.data))
                break
            after = snapshot(ep.env.data)
            require_identical_snapshot(before, after)
            if not np.array_equal(ep.last_action, previous):
                raise ValueError("lifecycle erased actual preceding standing action")
            boundary = store(
                f"between{k - 1}-{k}-boundary",
                {
                    **{"before_" + key: value for key, value in before.items()},
                    **{"after_" + key: value for key, value in after.items()},
                    "previous_actual_action": previous,
                },
            )
            boundaries[-1]["boundary"] = boundary
        motor = actor if schedule[k] == "student" else actor.base
        raw, row = bench.collect_current(ep, motor, stop_at_reference=True)
        if not np.array_equal(raw["NN_action"], raw["NN_nominal_action"]):
            raise ValueError("kick requires declared own neural actor, no teacher actions")
        kicks.append(
            dict(
                index=k,
                result=dict(row, **store(f"kick{k}", raw)),
                terminal=store(f"kick-terminal{k}", snapshot(ep.env.data)),
            )
        )
        if row["body_failed"]:
            status = "KICK_BODY_FAILURE"
            break
        stand_task = None
        if variant:
            stand_task = NeuralStandingTask(
                ep,
                actor.base,
                state=(
                    torch.from_numpy(raw["NN_h_after"][-1].copy()),
                    torch.from_numpy(raw["NN_c_after"][-1].copy()),
                ),
                memory_mode="carry",
                count_inference=inference,
            )
            stand_raw, stand_row = stand_task.collect(100)
        else:
            before = snapshot(ep.env.data)
            stand_raw, stand_row = collect_balance_recovery(ep, np.zeros((9, 4)))
            stand_raw.update({"initial_" + key: value for key, value in before.items()})
            stand_raw.update(
                {"terminal_" + key: value for key, value in snapshot(ep.env.data).items()}
            )
            stand_row.update(
                task="fixed_target_stand",
                neural_rows=0,
                stability_observed=stand_row["recovery_stability_observed"],
            )
        stands.append(dict(index=k, result=dict(stand_row, **store(f"stand{k}", stand_raw))))
        if not stand_row["healthy"]:
            status = "STAND_BODY_FAILURE"
            break
        if k == skills - 1:
            status = "ALL_KICKS_AND_STANDS_EXECUTED"
            break
        try:
            drop = practice_ball_drop(ep, foot=foot)
        except ValueError as error:
            if str(error) != "declared practice drop overlaps body; retain failure, no resample":
                raise
            status = "FIXTURE_REJECTED"
            break
        rec = dict(drop=store(f"between{k}-{k + 1}-drop", drop), settle=None, boundary=None)
        boundaries.append(rec)
        if bool(drop["performed"]):
            if stand_task is not None:
                settle, settle_row = stand_task.collect(50, until_ball_settled=True)
                healthy, settled = settle_row["healthy"], settle_row["ball_settled"]
            else:
                before = snapshot(ep.env.data)
                rows = []
                for _ in range(50):
                    part = ep.advance_hold()
                    rows.append(part)
                    q, v = ep.layout.canonical(ep.env.data)
                    if part["body_failed"] or (q[38] <= 0.12 and abs(v[37]) <= 0.1):
                        break
                settle = {key: np.array([x[key] for x in rows]) for key in rows[0]}
                settle.update({"initial_" + key: value for key, value in before.items()})
                settle.update(
                    {"terminal_" + key: value for key, value in snapshot(ep.env.data).items()}
                )
                healthy, settled = (
                    not bool(settle["body_failed"].any()),
                    bool(q[38] <= 0.12 and abs(v[37]) <= 0.1),
                )
                settle_row = dict(
                    task="fixed_target_settle",
                    healthy=healthy,
                    ball_settled=settled,
                    native_steps=len(rows) * 10,
                    neural_rows=0,
                    controls=len(rows),
                )
            rec["settle"] = dict(settle_row, **store(f"between{k}-{k + 1}-settle", settle))
            if not healthy:
                status = "SETTLE_BODY_FAILURE"
                break
            if not settled:
                status = "BALL_NOT_SETTLED"
                break
    if ep.native_steps != bench.native_steps - start_native or not np.isclose(
        ep.env.data.time, ep.native_steps * 0.002, rtol=0, atol=1e-10
    ):
        raise ValueError("one continuous body/clock/native accounting required")
    return dict(
        **(dict(initial=initial) if record_initial else {}),
        **(dict(motor_schedule=list(schedule)) if motor_schedule is not None else {}),
        status=status,
        kicks=kicks,
        stands=stands,
        boundaries=boundaries,
        stable=status == "ALL_KICKS_AND_STANDS_EXECUTED",
        native_steps=ep.native_steps,
        neural_rows=bench.neural_rows - start_neural,
        final_sim_time=float(ep.env.data.time),
        body_initializations=1,
        body_resets=0,
        requested_skills=skills,
        variant=variant,
        first_only_qualified_reference_prefix=True,
        old_exam_result=False,
        optimizer_updates=0,
        hardware_authorized=False,
        team_game=False,
    )
