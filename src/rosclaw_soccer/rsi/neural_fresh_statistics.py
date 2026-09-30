"""Frozen paired high-quality metric with whole-seed cluster uncertainty."""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_cluster_score(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    seeds, lanes = protocol["seeds"], protocol["lanes_per_seed"]
    expected = {(seed, lane) for seed in seeds for lane in lanes}
    if len(rows) != len(expected) or {(r["seed"], r["lane"]) for r in rows} != expected:
        raise ValueError("complete unique preregistered fresh course pool required")
    if len(seeds) != 10 or len(set(seeds)) != 10 or len(lanes) != 4 or len(set(lanes)) != 4:
        raise ValueError("ten whole-seed clusters of four courses required")
    measurements = ("minimum_pelvis_z_m", "maximum_lateral_excursion_m")
    for row in rows:
        for arm in ("parent", "candidate"):
            outcome = row[arm]
            if any(
                type(outcome[k]) is not bool for k in ("high_quality", "clean_foot_only")
            ) or any(
                type(outcome[k]) not in (float, int) or not np.isfinite(outcome[k])
                for k in measurements
            ):
                raise ValueError("finite physical measurements and boolean outcomes required")
    cluster_delta = np.asarray(
        [
            sum(
                int(r["candidate"]["high_quality"]) - int(r["parent"]["high_quality"])
                for r in rows
                if r["seed"] == seed
            )
            / len(lanes)
            for seed in seeds
        ]
    )
    samples = np.random.default_rng(20261001310).integers(0, len(seeds), size=(20000, len(seeds)))
    low, high = np.quantile(cluster_delta[samples].mean(axis=1), [0.025, 0.975])
    improvement = float(cluster_delta.mean())
    clean_loss = sum(
        r["parent"]["clean_foot_only"] and not r["candidate"]["clean_foot_only"] for r in rows
    )
    quality_loss = sum(
        r["parent"]["high_quality"] and not r["candidate"]["high_quality"] for r in rows
    )
    new_out = sum(
        r["parent"]["maximum_lateral_excursion_m"]
        <= 4
        < r["candidate"]["maximum_lateral_excursion_m"]
        for r in rows
    )
    pelvis = all(r["candidate"]["minimum_pelvis_z_m"] >= 0.65 for r in rows)
    return dict(
        parent_high_quality_count=sum(r["parent"]["high_quality"] for r in rows),
        candidate_high_quality_count=sum(r["candidate"]["high_quality"] for r in rows),
        absolute_success_gain=improvement,
        paired_cluster_ci95=[float(low), float(high)],
        bootstrap_resamples=20000,
        independent_seed_cluster_count=10,
        physical_course_count=40,
        parent_clean_foot_loss=clean_loss,
        parent_high_quality_loss=quality_loss,
        new_out_of_play=new_out,
        safe_pelvis_guardrail=pelvis,
        fresh_effect_gate_passed=bool(
            improvement + 1e-12 >= protocol["minimum_absolute_success_gain"] and low > 0
        ),
        fresh_safety_gate_passed=pelvis and clean_loss == 0 and new_out == 0,
        promotion_authorized=False,
    )
