"""Recreate finite CEM proposals/selection, not a substitute for native replay."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.stroke_search import checked_latent, sample_generation, update_elites
from rosclaw_soccer.rsi.targeted_pass_precision import precision_rank


def verify_targeted_search(
    report: dict[str, Any], *, context: int, center: int, targets: list[list[float]], foot: str
) -> None:
    records: list[dict[str, Any]] = report["records"]
    if (
        len(records) != 72
        or len(report["generations"]) != 4
        or len(report["selections"]) != 2
        or len(targets) != 2
        or context not in (1, 4, 20, 21)
        or not 98 <= center <= 180
        or foot not in ("left", "right")
    ):
        raise ValueError("complete bounded72-world targeted-pass search required")
    cursor = 0

    def row(task: int, tag: str, z: np.ndarray) -> dict[str, Any]:
        nonlocal cursor
        r = records[cursor]
        cursor += 1
        if (
            r["task"] != task
            or r["tag"] != tag
            or r["context"] != context
            or r["center"] != center
            or r["stroke_foot"] != foot
            or r["target"] != targets[task]
            or not np.array_equal(checked_latent(r["latent"]), z)
            or r["precision_rank"] != list(precision_rank(r["passing_result"]))
            or r["assisted"] is not True
            or r["PPO_actor_rows"] != 0
        ):
            raise ValueError("every ordered actual teacher proposal must match declaration")
        return r

    selections = []
    for task in range(2):
        rng = np.random.default_rng(66023 + context * 101 + task)
        mean, std = np.zeros((2, 6)), np.full((2, 6), 0.4)
        incumbent = row(task, f"task{task}-baseline", np.zeros((2, 6)))
        repeat = row(task, f"task{task}-baseline-repeat", np.zeros((2, 6)))
        if repeat["passing_result"] != incumbent["passing_result"]:
            raise ValueError("zero teacher repeat outcome changed")
        if {k: v["hash"] for k, v in repeat["artifacts"].items()} != {
            k: v["hash"] for k, v in incumbent["artifacts"].items()
        }:
            raise ValueError("zero teacher whole raw stream repeat differs")
        for generation in range(2):
            samples, draws = sample_generation(rng, mean, std)
            values = [incumbent] + [
                row(task, f"task{task}-g{generation}-c{k}", z) for k, z in enumerate(samples)
            ]
            ranks = [precision_rank(r["passing_result"]) for r in values]
            updated, spread, elite = update_elites([np.array(r["latent"]) for r in values], ranks)
            best = max(range(17), key=lambda k: (ranks[k], -k))
            expected = dict(
                task=task,
                generation=generation,
                draws=draws,
                mean_before=mean.tolist(),
                std_before=std.tolist(),
                tags=[r["tag"] for r in values],
                ranks=[list(r) for r in ranks],
                elite=elite,
                mean_after=updated.tolist(),
                std_after=spread.tolist(),
                selected=values[best]["tag"],
            )
            if report["generations"][task * 2 + generation] != expected:
                raise ValueError("own CEM draws/elite/selection differ")
            incumbent = values[best]
            mean, std = updated, spread
        for k in range(2):
            repeat = row(task, f"task{task}-selected-repeat{k}", np.array(incumbent["latent"]))
            if repeat["passing_result"] != incumbent["passing_result"]:
                raise ValueError("selected teacher actual repeat differs")
            if {k: v["hash"] for k, v in repeat["artifacts"].items()} != {
                k: v["hash"] for k, v in incumbent["artifacts"].items()
            }:
                raise ValueError("selected teacher whole raw stream repeat differs")
        selections.append(
            dict(
                task=task,
                selected_tag=incumbent["tag"],
                latent=incumbent["latent"],
                passing_result=incumbent["passing_result"],
            )
        )
    if (
        cursor != 72
        or selections != report["selections"]
        or report["counts"]
        != dict(
            native_steps=sum(r["native_steps"] for r in records),
            actor_calls=sum(r["actor_decisions"] for r in records),
        )
    ):
        raise ValueError("whole72 selected teacher/accounting differs")
