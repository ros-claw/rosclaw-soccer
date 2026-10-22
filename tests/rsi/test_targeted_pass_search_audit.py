"""Synthetic CEM custody, NOT native pass evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.stroke_search import sample_generation, update_elites
from rosclaw_soccer.rsi.targeted_pass_precision import precision_rank
from rosclaw_soccer.rsi.targeted_pass_search_audit import verify_targeted_search


def report():
    rows, generations, selections = [], [], []
    targets = [[3.75, 0.0, 0.11], [3.75, 0.2, 0.11]]

    def world(task, tag, z):
        result = dict(
            whole_body_healthy=True,
            recovery_observed=True,
            clean_foot=True,
            first_foot_contact_substep=1000,
            arrival_speed_mps=2.0,
            minimum_goal_error_3d_m=float(0.2 + 0.1 * np.linalg.norm(z)),
        )
        row = dict(
            context=4,
            task=task,
            tag=tag,
            center=110,
            stroke_foot="left",
            target=targets[task],
            latent=z.tolist(),
            passing_result=result,
            precision_rank=list(precision_rank(result)),
            assisted=True,
            PPO_actor_rows=0,
            native_steps=4000,
            actor_decisions=400,
            artifacts={"kick": dict(hash="synthetic-fixture-not-physics")},
        )
        rows.append(row)
        return row

    for task in range(2):
        rng = np.random.default_rng(66023 + 4 * 101 + task)
        mean, std = np.zeros((2, 6)), np.full((2, 6), 0.4)
        incumbent = world(task, f"task{task}-baseline", np.zeros((2, 6)))
        world(task, f"task{task}-baseline-repeat", np.zeros((2, 6)))
        for g in range(2):
            samples, draws = sample_generation(rng, mean, std)
            values = [incumbent] + [
                world(task, f"task{task}-g{g}-c{k}", z) for k, z in enumerate(samples)
            ]
            ranks = [precision_rank(r["passing_result"]) for r in values]
            updated, spread, elite = update_elites([np.array(r["latent"]) for r in values], ranks)
            best = max(range(17), key=lambda k: (ranks[k], -k))
            generations.append(
                dict(
                    task=task,
                    generation=g,
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
            )
            incumbent = values[best]
            mean, std = updated, spread
        for k in range(2):
            world(task, f"task{task}-selected-repeat{k}", np.array(incumbent["latent"]))
        selections.append(
            dict(
                task=task,
                selected_tag=incumbent["tag"],
                latent=incumbent["latent"],
                passing_result=incumbent["passing_result"],
            )
        )
    return dict(
        records=rows,
        generations=generations,
        selections=selections,
        counts=dict(native_steps=288000, actor_calls=28800),
    ), targets


def test_complete72_negative_worlds_are_valid_custody_not_reported_as_native_success():
    r, targets = report()
    verify_targeted_search(r, context=4, center=110, targets=targets, foot="left")


@pytest.mark.parametrize("change", ["latent", "elite", "order", "count", "foot"])
def test_tampering_or_dropped_work_rejected(change):
    r, targets = report()
    r = copy.deepcopy(r)
    if change == "latent":
        r["records"][2]["latent"][0][0] += 0.01
    elif change == "elite":
        r["generations"][0]["elite"] = []
    elif change == "order":
        r["records"][-1], r["records"][-2] = r["records"][-2], r["records"][-1]
    elif change == "count":
        r["counts"]["native_steps"] += 10
    else:
        r["records"][0]["stroke_foot"] = "right"
    with pytest.raises(ValueError):
        verify_targeted_search(r, context=4, center=110, targets=targets, foot="left")
