"""Real SIM recovery rollout and measured reward; no model/optimizer creation."""

from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.contact_option_credit import recovery_terminal_reward
from rosclaw_soccer.rsi.native_contact_window import native_contact_window
from rosclaw_soccer.rsi.native_recovery_tail import NativeRecoveryTail


def finish_contact_option(
    env: Any,
    head: dict[str, np.ndarray],
    foundation: Any,
    *,
    counters: dict[str, int] | None = None,
    trace: list[dict[str, np.ndarray]] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], int]:
    """Same live batch, immutable original foundation, valid inference only.

    Returns actual tail arrays/rewards, whole-option measurements, actual NN rows.
    All-failed head has a single all-invalid padding row, no invented q/v sample.
    """
    failed = np.asarray(head["terminated"].any(axis=0), dtype=bool)
    tail = NativeRecoveryTail(env, contact_failed=failed)
    device = next(foundation.parameters()).device
    rows = trace if trace is not None else []
    if rows:
        raise ValueError("unused recovery trace required")
    nn_rows = 0
    for _ in range(60):
        ids = tail.active_ids
        if not len(ids):
            break
        action = np.zeros((env.worlds, 29), dtype=np.float32)
        with torch.no_grad():
            action[ids] = (
                foundation(torch.from_numpy(tail.observation[ids]).to(device)).cpu().numpy()
            )
        nn_rows += len(ids)
        if counters is not None:
            counters["recovery_foundation_rows"] = counters.get(
                "recovery_foundation_rows", 0
            ) + len(ids)
        rows.append(tail.step(action))
    if rows:
        raw = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    else:
        raw = {
            k: np.zeros((1, env.worlds), dtype=bool) for k in ("valid", "terminated", "completed")
        }
    reward = np.zeros(raw["valid"].shape, dtype=np.float32)
    terminal = raw["terminated"] | raw["completed"]
    layout = env.layout
    qi = [*range(layout.rq, layout.rq + 7), *layout.qi, *range(layout.bq, layout.bq + 7)]
    vi = [*range(layout.rv, layout.rv + 6), *layout.vi, *range(layout.bv, layout.bv + 6)]
    outcomes = []
    for world, index in enumerate(env.index):
        qs, vs, forces = [], [], []
        for part in (head, raw):
            mask = part["valid"][:, world]
            if mask.any():
                qs.append(part["qpos"][mask, world].reshape(-1, 43)[:, qi])
                vs.append(part["qvel"][mask, world].reshape(-1, 41)[:, vi])
                forces.append(part["forces"][mask, world].reshape(-1, 3))
        q, v, f = (np.concatenate(x) for x in (qs, vs, forces))
        post = native_contact_window(
            force=f[:, :2],
            canonical_qpos=q,
            canonical_qvel=v,
            goal=env.bank["goal_world"][index],
            intended_foot=int(env.bank["expected_foot"][index]),
        )
        result = dict(
            world=world,
            bank_index=int(index),
            postcontact=post,
            contact_failed=bool(failed[world]),
            recovery_failed=bool(tail.failed[world] and not failed[world]),
            recovery_controls=int(tail.elapsed[world]),
            forbidden=bool((f[:, 2] > 1).any()),
            stable=bool(
                q[:, 2].min() >= 0.65 and (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).min() >= 0.9
            ),
        )
        last = np.flatnonzero(raw["valid"][:, world])
        if len(last):
            if not terminal[last[-1], world] or terminal[:, world].sum() != 1:
                raise ValueError("measured recovery must reach exactly one finite terminal")
            reward[last[-1], world] = recovery_terminal_reward(result)
        result["recovery_reward"] = float(reward[:, world].sum())
        outcomes.append(result)
    raw["reward"], raw["terminal"] = reward, terminal
    return raw, outcomes, nn_rows
