"""Actual variable-head SIM recovery collection, not a learner or promotion.

No hindsight observations: measured future is used only for terminal credit.
The original CPU foundation uses single-row inference to retain native rounding.
"""

from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.contact_option_credit import recovery_terminal_reward
from rosclaw_soccer.rsi.native_approach_contact import NativeApproachRecoveryTail
from rosclaw_soccer.rsi.native_contact_window import native_contact_window


def finish_approach_option(
    env: Any,
    head: dict[str, np.ndarray],
    foundation: Any,
    *,
    counters: dict[str, int],
    trace: list[dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if trace or next(foundation.parameters()).device.type != "cpu":
        raise ValueError("unused trace and original CPU foundation required")
    tail = NativeApproachRecoveryTail(env)
    for _ in range(60):
        ids = tail.active_ids
        if not len(ids):
            break
        action = np.zeros((env.worlds, 29), dtype=np.float32)
        obs = tail.observation
        with torch.no_grad():
            for i in ids:
                action[i] = foundation(torch.from_numpy(obs[i : i + 1])).numpy()[0]
        counters["recovery_foundation_rows"] = counters.get("recovery_foundation_rows", 0) + len(
            ids
        )
        trace.append(tail.step(action))
    if trace:
        raw = {k: np.stack([row[k] for row in trace]) for k in trace[0]}
    else:
        raw = {
            k: np.zeros((1, env.worlds), dtype=bool) for k in ("valid", "terminated", "completed")
        }
    terminal = raw["terminated"] | raw["completed"]
    reward = np.zeros(raw["valid"].shape, dtype=np.float32)
    layout = env.layout
    qi = [*range(layout.rq, layout.rq + 7), *layout.qi, *range(layout.bq, layout.bq + 7)]
    vi = [*range(layout.rv, layout.rv + 6), *layout.vi, *range(layout.bv, layout.bv + 6)]
    outcomes = []
    for w, index in enumerate(env.index):
        parts: dict[str, list[np.ndarray]] = {k: [] for k in ("qpos", "qvel", "forces")}
        for part in (head, raw):
            mask = part["valid"][:, w]
            if mask.any():
                for key, width in (("qpos", 43), ("qvel", 41), ("forces", 3)):
                    parts[key].append(part[key][mask, w].reshape(-1, width))
        q, v, f = (np.concatenate(parts[k]) for k in ("qpos", "qvel", "forces"))
        q, v = q[:, qi], v[:, vi]
        result = dict(
            world=w,
            bank_index=int(index),
            postcontact=native_contact_window(
                force=f[:, :2],
                canonical_qpos=q,
                canonical_qvel=v,
                goal=env.bank["goal_world"][index],
                intended_foot=int(env.bank["expected_foot"][index]),
            ),
            contact_failed=bool(env.head_failed[w]),
            recovery_failed=bool(tail.failed[w] and not env.head_failed[w]),
            reference_controls=int(env.elapsed[w]),
            recovery_controls=int(tail.elapsed[w]),
            forbidden=bool((f[:, 2] > 1).any()),
            stable=bool(
                q[:, 2].min() >= 0.65 and (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).min() >= 0.9
            ),
        )
        last = np.flatnonzero(raw["valid"][:, w])
        if len(last):
            if not np.array_equal(np.flatnonzero(terminal[:, w]), [last[-1]]):
                raise ValueError("recovery must end once at actual final sample")
            reward[last[-1], w] = recovery_terminal_reward(result)
        result["recovery_reward"] = float(reward[:, w].sum())
        outcomes.append(result)
    raw["reward"], raw["terminal"] = reward, terminal
    return raw, outcomes
