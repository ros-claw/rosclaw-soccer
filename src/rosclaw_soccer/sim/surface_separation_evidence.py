"""Numeric separation evidence; independent of force thresholds and skill scores.

The caller supplies signed surface distances from a declared physical clock and
geometry binding. A hash label does not authenticate that source. This module
does not call a simulator, change contacts, authorize motion or promote policies.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from rosclaw_soccer.sim.contracts import hash_json


class SurfaceSeparationTracker:
    """Account for every declared surface pair on every physics sample.

    Negative distance denotes overlap. Zero clearance is permitted only when the
    declared minimum is zero. Missing pairs, invalid clocks and nonfinite values
    latch a fault; later good samples cannot repair the episode. This predicate
    must supplement, never replace, body safety and task/contact-force exams.
    """

    def __init__(
        self,
        *,
        geometry_contract_hash: str,
        pair_ids: tuple[str, ...],
        physics_dt_sec: float,
        duration_sec: float,
        minimum_clearance_m: float = 0.0,
    ) -> None:
        values = (physics_dt_sec, duration_sec, minimum_clearance_m)
        if (
            not isinstance(geometry_contract_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", geometry_contract_hash) is None
            or type(pair_ids) is not tuple
            or not 1 <= len(pair_ids) <= 256
            or any(
                not isinstance(p, str)
                or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}", p) is None
                for p in pair_ids
            )
            or len(set(pair_ids)) != len(pair_ids)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not 1e-6 <= physics_dt_sec <= 0.01
            or not physics_dt_sec <= duration_sec <= 60
            or not 0 <= minimum_clearance_m <= 1
            or not math.isclose(
                duration_sec / physics_dt_sec,
                round(duration_sec / physics_dt_sec),
                rel_tol=0,
                abs_tol=1e-8,
            )
        ):
            raise ValueError("bound pairs and bounded integral physical clock required")
        self._pairs = tuple(sorted(pair_ids))
        self._dt = physics_dt_sec
        self._expected = round(duration_sec / physics_dt_sec)
        self._required = minimum_clearance_m
        self._hash = str(
            hash_json(
                dict(
                    schema="soccer.surface_separation.v1",
                    geometry_contract_hash=geometry_contract_hash,
                    pair_ids=self._pairs,
                    physics_dt_sec=physics_dt_sec,
                    duration_sec=duration_sec,
                    minimum_clearance_m=minimum_clearance_m,
                    distance_convention="signed_surface_distance_negative_overlap",
                )
            )
        )
        self._samples = 0
        self._faulted = False
        self._minima: dict[str, float | None] = dict.fromkeys(self._pairs)
        self._first_violation: float | None = None

    def observe(self, *, elapsed_sec: float, signed_distances_m: Mapping[str, float]) -> None:
        if self._faulted:
            raise ValueError("surface separation evidence is fault-latched")
        if not isinstance(signed_distances_m, Mapping):
            self._faulted = True
            raise ValueError("every bound pair requires a signed surface distance")
        distances = dict(signed_distances_m)
        if (
            type(elapsed_sec) not in (int, float)
            or not math.isfinite(elapsed_sec)
            or not math.isclose(
                elapsed_sec, (self._samples + 1) * self._dt, rel_tol=0, abs_tol=1e-8
            )
            or self._samples >= self._expected
            or set(distances) != set(self._pairs)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in distances.values())
        ):
            self._faulted = True
            raise ValueError("complete consecutive finite surface samples required")
        self._samples += 1
        for pair, distance in distances.items():
            previous = self._minima[pair]
            self._minima[pair] = float(distance) if previous is None else min(previous, distance)
            if distance < self._required and self._first_violation is None:
                self._first_violation = elapsed_sec

    def result(self) -> dict[str, object]:
        complete = self._samples == self._expected
        return dict(
            schema="soccer.surface_separation.v1",
            contract_hash=self._hash,
            samples=self._samples,
            expected_samples=self._expected,
            complete=complete,
            faulted=self._faulted,
            minimum_signed_distance_m=dict(self._minima),
            required_clearance_m=self._required,
            first_clearance_violation_sec=self._first_violation,
            separated_entire_episode=bool(
                complete and not self._faulted and self._first_violation is None
            ),
            activation_ceiling="SIM_ONLY",
            promotion_eligible=False,
        )
