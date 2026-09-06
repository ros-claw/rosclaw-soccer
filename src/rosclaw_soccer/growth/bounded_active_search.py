"""Deterministic, content-bound active-search plans for SIM-only growth loops."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import cast

from rosclaw_soccer.sim.contracts import hash_json

_NAME = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19)


@dataclass(frozen=True)
class BoundedSearchDimension:
    """One finite high-level parameter interval."""

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            not _NAME.fullmatch(self.name)
            or not math.isfinite(self.lower)
            or not math.isfinite(self.upper)
            or self.lower >= self.upper
        ):
            raise ValueError("bounded search dimension is invalid")


@dataclass(frozen=True)
class BoundedSearchCandidate:
    """One immutable proposal; evaluation authority remains with the caller."""

    candidate_index: int
    stage: str
    values: tuple[float, ...]
    plan_hash: str
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_index, bool)
            or self.candidate_index < 0
            or self.stage not in {"GLOBAL", "LOCAL", "WARM_START"}
            or not self.values
            or not all(math.isfinite(value) for value in self.values)
            or not self.plan_hash.startswith("sha256:")
            or len(self.plan_hash) != 71
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.direct_joint_torque_output
        ):
            raise ValueError("bounded search candidate is invalid")

    @property
    def candidate_hash(self) -> str:
        return cast(str, hash_json(asdict(self)))


@dataclass(frozen=True)
class BoundedActiveSearchPlan:
    """Plan deterministic global and trust-region proposals without evaluation."""

    dimensions: tuple[BoundedSearchDimension, ...]
    global_candidate_count: int = 64
    local_candidate_count: int = 64
    sequence_skip: int = 32
    local_radius_fraction: float = 0.125
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.bounded_active_search_plan.v1"

    def __post_init__(self) -> None:
        if (
            not 1 <= len(self.dimensions) <= len(_PRIMES)
            or len({dimension.name for dimension in self.dimensions}) != len(self.dimensions)
            or isinstance(self.global_candidate_count, bool)
            or isinstance(self.local_candidate_count, bool)
            or not 4 <= self.global_candidate_count <= 4096
            or not 4 <= self.local_candidate_count <= 4096
            or isinstance(self.sequence_skip, bool)
            or not 0 <= self.sequence_skip <= 1_000_000
            or not math.isfinite(self.local_radius_fraction)
            or not 0.01 <= self.local_radius_fraction <= 0.50
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.schema_version != "rosclaw_soccer.bounded_active_search_plan.v1"
        ):
            raise ValueError("bounded active search plan is invalid")

    @property
    def plan_hash(self) -> str:
        return cast(str, hash_json(asdict(self)))

    def global_candidates(self) -> tuple[BoundedSearchCandidate, ...]:
        return self._candidates(
            stage="GLOBAL",
            count=self.global_candidate_count,
            lower=tuple(dimension.lower for dimension in self.dimensions),
            upper=tuple(dimension.upper for dimension in self.dimensions),
            index_offset=0,
        )

    def local_candidates(self, center: tuple[float, ...]) -> tuple[BoundedSearchCandidate, ...]:
        if len(center) != len(self.dimensions) or not all(math.isfinite(value) for value in center):
            raise ValueError("local search center is invalid")
        for value, dimension in zip(center, self.dimensions, strict=True):
            if not dimension.lower <= value <= dimension.upper:
                raise ValueError("local search center exceeds the global envelope")
        lower = tuple(
            max(
                dimension.lower,
                value - self.local_radius_fraction * (dimension.upper - dimension.lower),
            )
            for value, dimension in zip(center, self.dimensions, strict=True)
        )
        upper = tuple(
            min(
                dimension.upper,
                value + self.local_radius_fraction * (dimension.upper - dimension.lower),
            )
            for value, dimension in zip(center, self.dimensions, strict=True)
        )
        warm_start = BoundedSearchCandidate(
            candidate_index=0,
            stage="WARM_START",
            values=center,
            plan_hash=self.plan_hash,
        )
        local = self._candidates(
            stage="LOCAL",
            count=self.local_candidate_count,
            lower=lower,
            upper=upper,
            index_offset=1,
        )
        return (warm_start, *local)

    def _candidates(
        self,
        *,
        stage: str,
        count: int,
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        index_offset: int,
    ) -> tuple[BoundedSearchCandidate, ...]:
        candidates = []
        for offset in range(count):
            sequence_index = self.sequence_skip + offset + 1
            unit = tuple(
                _radical_inverse(sequence_index, _PRIMES[index])
                for index in range(len(self.dimensions))
            )
            values = tuple(
                low + coordinate * (high - low)
                for coordinate, low, high in zip(unit, lower, upper, strict=True)
            )
            candidates.append(
                BoundedSearchCandidate(
                    candidate_index=index_offset + offset,
                    stage=stage,
                    values=values,
                    plan_hash=self.plan_hash,
                )
            )
        return tuple(candidates)


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    value = index
    while value:
        result += factor * (value % base)
        value //= base
        factor /= base
    return result


__all__ = [
    "BoundedActiveSearchPlan",
    "BoundedSearchCandidate",
    "BoundedSearchDimension",
]
