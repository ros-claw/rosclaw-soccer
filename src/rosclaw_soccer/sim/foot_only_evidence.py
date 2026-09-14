"""Whole-episode robot contact evidence, separate from initial impulse credit.

Callers must bind actual geometry groups and report EVERY physics sample. Ball
contacts with the floor/net/posts are not robot-body contacts. No motion, score
inflation, policy activation or physical provenance follows from hash labels.
"""

from __future__ import annotations

import math
import re

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.sim.strike_quality import StrikeQualityConfig


class FootOnlyContactTracker:
    """Reject prior, simultaneous AND later non-foot robot contact.

    Unlike StrikeQualityTracker's initial impulse attribution, this requires the
    entire declared episode to remain clean. Forces use >= the declared threshold.
    Body safety and completed accounting are necessary, but not a goal/skill exam.
    """

    def __init__(self, *, geometry_contract_hash: str, config: StrikeQualityConfig) -> None:
        if (
            not isinstance(geometry_contract_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", geometry_contract_hash) is None
            or not isinstance(config, StrikeQualityConfig)
        ):
            raise ValueError("bound geometry groups and physical clock required")
        self._config = config
        self._hash = str(
            hash_json(
                dict(
                    schema="soccer.foot_only_contact.v1",
                    geometry_contract_hash=geometry_contract_hash,
                    physics_dt_sec=config.physics_dt_sec,
                    duration_sec=config.duration_sec,
                    force_threshold_n=config.force_threshold_n,
                    force_comparator="greater_or_equal",
                    environment_excluded_from_nonfoot_robot_force=True,
                )
            )
        )
        self._samples = 0
        self._faulted = False
        self._safe = True
        self._first_foot: float | None = None
        self._first_nonfoot: float | None = None

    def observe(
        self,
        *,
        elapsed_sec: float,
        body_safe: bool,
        foot_ball_normal_force_n: float,
        nonfoot_robot_ball_normal_force_n: float,
    ) -> None:
        if self._faulted:
            raise ValueError("foot-only evidence is fault-latched")
        expected = (self._samples + 1) * self._config.physics_dt_sec
        values = (elapsed_sec, foot_ball_normal_force_n, nonfoot_robot_ball_normal_force_n)
        if (
            type(body_safe) is not bool
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or min(foot_ball_normal_force_n, nonfoot_robot_ball_normal_force_n) < 0
            or not math.isclose(elapsed_sec, expected, rel_tol=0, abs_tol=1e-8)
            or elapsed_sec > self._config.duration_sec + 1e-8
        ):
            self._faulted = True
            raise ValueError("complete consecutive finite physical contact samples required")
        self._samples += 1
        self._safe &= body_safe
        if foot_ball_normal_force_n >= self._config.force_threshold_n and self._first_foot is None:
            self._first_foot = elapsed_sec
        if (
            nonfoot_robot_ball_normal_force_n >= self._config.force_threshold_n
            and self._first_nonfoot is None
        ):
            self._first_nonfoot = elapsed_sec

    def result(self) -> dict[str, object]:
        expected = round(self._config.duration_sec / self._config.physics_dt_sec)
        complete = self._samples == expected
        return dict(
            schema="soccer.foot_only_contact.v1",
            contract_hash=self._hash,
            samples=self._samples,
            expected_samples=expected,
            complete=complete,
            faulted=self._faulted,
            body_safe_entire_episode=self._safe,
            first_foot_sec=self._first_foot,
            first_nonfoot_robot_sec=self._first_nonfoot,
            foot_only_complete=bool(
                complete
                and not self._faulted
                and self._safe
                and self._first_foot is not None
                and self._first_nonfoot is None
            ),
            activation_ceiling="SIM_ONLY",
            promotion_eligible=False,
        )
