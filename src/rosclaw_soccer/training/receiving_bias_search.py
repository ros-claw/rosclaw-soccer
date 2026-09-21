"""Bounded counterfactual receiving search, never motor or promotion authority.

Intermediate beam nodes may contain forbidden contacts. They are private search
seeds, not executable proposals. The caller must re-check complete candidates
and run the unchanged physical exam; a clean prediction is not physical success.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReceivingBiasForecast:
    unsafe: bool
    foot_contact_samples: int
    nonfoot_contact_samples: int
    minimum_shin_gap_m: float
    terminal_ball_speed: float

    def __post_init__(self) -> None:
        if type(self.unsafe) is not bool or any(
            type(value) is not int or not 0 <= value <= 100_000
            for value in (self.foot_contact_samples, self.nonfoot_contact_samples)
        ):
            raise ValueError("explicit predicted safety and bounded contact counts required")
        for value in (self.minimum_shin_gap_m, self.terminal_ball_speed):
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e4:
                raise ValueError("finite bounded predicted geometry and speed required")
        if self.terminal_ball_speed < 0:
            raise ValueError("predicted speed must be nonnegative")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ReceivingBiasForecast":
        if not isinstance(row, Mapping):
            raise ValueError("named counterfactual diagnostics required")
        try:
            return cls(
                row["unsafe"],
                row["foot_contact_samples"],
                row["nonfoot_contact_samples"],
                row["minimum_shin_gap_m"],
                row["terminal_ball_speed"],
            )
        except KeyError as error:
            raise ValueError("incomplete counterfactual diagnostics") from error

    @property
    def clean(self) -> bool:
        """Local search filter only, not the receiving exam or a safety certificate."""
        self.__post_init__()
        return (
            not self.unsafe
            and self.nonfoot_contact_samples == 0
            and self.foot_contact_samples > 0
            and self.terminal_ball_speed <= 0.35
        )


def receiving_bias_candidates(
    bias: Sequence[float] | NDArray[np.float64],
) -> list[tuple[str, NDArray[np.float64]]]:
    """Original 24 right-leg increments; each returned vector is caller-owned.

    The action interface is A0 leg12. Bias stays within 0.06 rad; the caller must
    still clip the entire reference-plus-bias proposal to its original 0.1 rad
    limit and leave filtering, rate limits and torque guards with the executor.
    """
    raw = np.asarray(bias)
    if (
        raw.shape != (12,)
        or raw.dtype.kind not in "fiu"
        or any(isinstance(value, (bool, np.bool_)) for value in bias)
    ):
        raise ValueError("twelve real numeric bias values required")
    current = np.asarray(raw, dtype=np.float64).copy()
    if not np.isfinite(current).all() or float(np.max(np.abs(current))) > 0.060000000001:
        raise ValueError("finite bias bounded by 0.06 rad required")
    candidates: list[tuple[str, NDArray[np.float64]]] = []
    for joint in range(6, 12):
        for magnitude in (0.02, 0.06):
            for sign in (-1, 1):
                proposed = current.copy()
                proposed[joint] = np.clip(proposed[joint] + sign * magnitude, -0.06, 0.06)
                candidates.append((f"joint_{joint}_{sign}_{magnitude}", proposed))
    return candidates


def receiving_bias_is_clean(diagnostics: Mapping[str, Any]) -> bool:
    """Validate and apply the research prediction filter, not the physical exam."""
    return ReceivingBiasForecast.from_mapping(diagnostics).clean


def receiving_bias_beam_seeds(
    rows: Sequence[Mapping[str, Any]],
    *,
    width: int = 2,
) -> tuple[int, ...]:
    """Rank private partial fixes, rejecting malformed batches instead of skipping.

    Prefer fewer nonfoot contact samples, more shin clearance, lower speed, then
    smaller action change. Stable input order breaks ties. Only predicted body-safe
    nodes with foot contact can seed another search depth; they are not thereby
    authorized for execution.
    """
    if type(width) is not int or not 1 <= width <= 8 or len(rows) > 256:
        raise ValueError("bounded explicit beam width and batch required")
    checked: list[tuple[ReceivingBiasForecast, float]] = []
    for row in rows:
        if not isinstance(row, Mapping) or "diagnostics" not in row or "change_squared" not in row:
            raise ValueError("bound candidate diagnostics and change required")
        diagnostics = ReceivingBiasForecast.from_mapping(row["diagnostics"])
        change = row["change_squared"]
        if type(change) not in (int, float) or not math.isfinite(change) or not 0 <= change <= 1e4:
            raise ValueError("finite nonnegative candidate change required")
        checked.append((diagnostics, float(change)))
    eligible = [
        i
        for i, (diagnostics, _) in enumerate(checked)
        if not diagnostics.unsafe and diagnostics.foot_contact_samples > 0
    ]
    return tuple(
        sorted(
            eligible,
            key=lambda i: (
                checked[i][0].nonfoot_contact_samples,
                -checked[i][0].minimum_shin_gap_m,
                checked[i][0].terminal_ball_speed,
                checked[i][1],
                i,
            ),
        )[:width]
    )
