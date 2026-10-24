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
