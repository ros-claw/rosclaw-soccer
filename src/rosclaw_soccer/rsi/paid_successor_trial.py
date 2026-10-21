"""One explicitly restored SIM training branch, kick then neural standing."""

from typing import Any

import torch

from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
from rosclaw_soccer.rsi.paid_successor_branch import begin_successor_branch


def run_successor_trial(
    bench: Any,
    case: dict[str, Any],
    actor: Any,
    *,
    entry: Any,
    previous: Any,
    transform: Any,
    store: Any,
) -> dict[str, Any]:
    ep = begin_successor_branch(bench, case, entry, previous)
    raw, row = bench.collect_current(ep, actor, transform=transform, stop_at_reference=True)
    result = dict(
        kick=dict(row, **store("kick", raw)),
        kick_terminal=store("kick-terminal", snapshot(ep.env.data)),
        stand=None,
    )
    if not row["body_failed"]:

        def inference() -> None:
            with bench.lock:
                bench.neural_rows += 1

        task = NeuralStandingTask(
            ep,
            actor,
            state=(
                torch.from_numpy(raw["NN_h_after"][-1].copy()),
                torch.from_numpy(raw["NN_c_after"][-1].copy()),
            ),
            memory_mode="carry",
            count_inference=inference,
        )
        stand, metric = task.collect(100)
        result["stand"] = dict(metric, **store("stand", stand))
    return result
