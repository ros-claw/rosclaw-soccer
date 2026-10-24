"""Finite privileged TRAIN search for precise AND moderate-speed delivery.

Ranking differs from the old functional-receive-first search: controlled receive
cannot hide a persistent precision/speed violation. No neural updates here.
"""

import math
from typing import Any

from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label

PILOTS = {25: (115025, 1), 26: (115023, 0)}
WORLDS = 38


def joint_delivery_rank(row: dict[str, Any]) -> tuple[int, int, int, float]:
    o = row["outcome"]
    controlled = controlled_pass_label(o)
    if type(o["combined_precision_and_functional_receive"]) is not bool:
        raise ValueError("typed original precision/receive outcome required")
    e = o["incoming"]["incoming_target_error_3d_before_B_contact_m"]
    v = o["incoming_arrival_speed_mps"]
    if any(
        x is not None and (type(x) not in (int, float) or not math.isfinite(x) or x < 0)
        for x in (e, v)
    ):
        raise ValueError("finite actual delivery error/speed or explicit missingness required")
    safe = bool(
        o["whole_body_healthy"]
        and o["passer_clean_foot"]
        and o["passer_full100NNrecovery"]
        and not o["B_excluded_contact"]
        and o["incoming"]["A_first_foot_strike_substep"] is not None
    )
    observed = e is not None and v is not None
    joint = bool(observed and e <= 0.1 and v <= 3.0)
    if o["combined_precision_and_functional_receive"] and not joint:
        raise ValueError("strict receipt cannot contradict actual precision/speed measurements")
    strict = bool(controlled and o["combined_precision_and_functional_receive"])
    # Training-only margins; official scoring remains error<=.1 and speed<=3.
    score = (
        -1e6
        if not observed
        else -((max(e - 0.08, 0.0) / 0.1) ** 2) - max(v - 2.8, 0.0) ** 2 + 0.05 * controlled
    )
    return int(safe), int(strict), int(joint), float(score)


def pilot_config(state: int) -> dict[str, Any]:
    if type(state) is not int or state not in PILOTS:
        raise ValueError("two preregistered near-precision PUBLIC TRAIN pilots only")
    seed, task = PILOTS[state]
    return dict(
        state_id=state,
        source_expert_seed=seed,
        task=task,
        rng_seed=176023 + 101 * state,
        generations=2,
        proposals=16,
        selected_repeats=2,
        unsearched_TRAIN_offsets_forward_m=[-0.005, 0.005],
        repeats_per_TRAIN_offset=2,
        maximum_worlds=WORLDS,
        maximum_native_steps=WORLDS * 4000,
        maximum_actor_calls=WORLDS * 800,
        formal_controls=300,
        A_neural_recovery_controls=100,
        original_total_authority_rad=0.2,
        new_optimizer_updates=0,
        Fresh_cases_read=0,
    )


def verify_pilot_search(
    report: dict[str, Any],
    baseline: dict[str, Any],
    validation_courses: list[dict[str, Any]],
) -> None:
    """Recreate every proposal, rank and selection; NOT a physical verifier.

    The caller must authenticate the inherited baseline and independently
    regenerate/preflight the two untouched TRAIN perturbations before calling.
    """
    import numpy as np

    from rosclaw_soccer.rsi.stroke_search import sample_generation, update_elites

    cfg = pilot_config(report["config"]["state_id"])
    rows = report["records"]
    course = baseline["course"]
    if (
        report["config"] != cfg
        or len(rows) != WORLDS
        or len(report["generations"]) != 2
        or len(validation_courses) != 2
        or course["state_id"] != cfg["state_id"]
        or course["task"] != cfg["task"]
        or course["assisted"] is not False
    ):
        raise ValueError("complete fixed two-generation38-world TRAIN pilot required")
    cursor = 0

    def take(tag: str, target: dict[str, Any], latent: np.ndarray) -> dict[str, Any]:
        nonlocal cursor
        row = rows[cursor]
        if (
            row["index"] != cursor
            or row["tag"] != tag
            or row["course"] != target | dict(assisted=True, latent=latent.tolist())
            or row["neural_actor_optimizer_updates"] != 0
            or (
                cursor < 34
                and (
                    type(row["prefix_identity_fields"]) is not int
                    or row["prefix_identity_fields"] <= 0
                )
            )
            or (cursor >= 34 and row["prefix_identity_fields"] is not None)
        ):
            raise ValueError("ordered own-state proposal/selection/validation accounting required")
        cursor += 1
        return dict(tag=tag, latent=latent, row=row)

    incumbent: dict[str, Any] = dict(tag="inherited", latent=np.zeros((2, 6)), row=baseline)
    mean, std = np.zeros((2, 6)), np.full((2, 6), 0.4)
    rng = np.random.default_rng(cfg["rng_seed"])
    for g in range(2):
        samples, draws = sample_generation(rng, mean, std)
        values = [incumbent] + [take(f"g{g}-c{k}", course, z) for k, z in enumerate(samples)]
        ranks = [joint_delivery_rank(v["row"]) for v in values]
        updated, spread, elite = update_elites([v["latent"] for v in values], ranks)
        best = max(range(17), key=lambda k: (ranks[k], -k))
        incumbent = values[best]
        expected = dict(
            generation=g,
            mean_before=mean.tolist(),
            std_before=std.tolist(),
            draws=draws,
            tags=[v["tag"] for v in values],
            ranks=[list(r) for r in ranks],
            elite=elite,
            mean_after=updated.tolist(),
            std_after=spread.tolist(),
            selected=incumbent["tag"],
        )
        if report["generations"][g] != expected:
            raise ValueError("preregistered random draws, joint ranks, elites or winner differ")
        mean, std = updated, spread
    for repeat in range(2):
        v = take(f"selected-repeat{repeat}", course, incumbent["latent"])
        if v["row"]["outcome"] != incumbent["row"]["outcome"]:
            raise ValueError("selected teacher failed exact physical outcome repetition")
    for index, variant in enumerate(validation_courses):
        pair = [
            take(f"validation{index}-repeat{r}", variant, incumbent["latent"]) for r in range(2)
        ]
        if pair[0]["row"]["outcome"] != pair[1]["row"]["outcome"]:
            raise ValueError("unsearched TRAIN validation repetition differs")
    selected = dict(
        tag=incumbent["tag"],
        latent=incumbent["latent"].tolist(),
        rank=list(joint_delivery_rank(incumbent["row"])),
        outcome=incumbent["row"]["outcome"],
    )
    if cursor != WORLDS or report["selected"] != selected:
        raise ValueError("complete38-world final selection commitment required")
