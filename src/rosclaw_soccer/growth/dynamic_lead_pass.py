"""Data-driven lead-pass adapter for the alternating team Growth loop.

The adapter is intentionally small and interpretable.  It learns two physical
calibrations from discovery-only MuJoCo rollouts:

* receiver phase start -> longitudinal interception pocket; and
* passer yaw delta -> lateral ball delivery.

The resulting policy changes the passer's executed whole-body action.  A
``pass_reception_target`` that is only written into a metric is not considered
an action and cannot be represented by this contract.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class LeadPassCalibrationSample:
    """One discovery-only physics measurement used by the adapter."""

    sample_id: str
    receiver_phase_start_sec: float
    passer_yaw_delta_rad: float
    delivery_position_m: tuple[float, float, float]
    trajectory_hash: str
    safe: bool
    schema_version: str = "rosclaw_soccer.lead_pass_calibration_sample.v1"

    def __post_init__(self) -> None:
        values = (
            self.receiver_phase_start_sec,
            self.passer_yaw_delta_rad,
            *self.delivery_position_m,
        )
        if not self.sample_id or not self.sample_id.replace("-", "").isalnum():
            raise ValueError("lead-pass sample id is invalid")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("lead-pass calibration sample must be finite")
        if not 1.50 <= self.receiver_phase_start_sec <= 2.50:
            raise ValueError("lead-pass receiver phase start is outside the calibrated domain")
        if not -0.12 <= self.passer_yaw_delta_rad <= 0.12:
            raise ValueError("lead-pass yaw delta is outside the safe adapter envelope")
        if not self.trajectory_hash.startswith("sha256:") or len(self.trajectory_hash) != 71:
            raise ValueError("lead-pass sample must bind a trajectory hash")


@dataclass(frozen=True)
class DynamicLeadPassPolicy:
    """A fitted, SIM-only passer policy artifact."""

    longitudinal_intercept_slope_m_per_sec: float
    longitudinal_intercept_m: float
    lateral_delivery_slope_m_per_rad: float
    lateral_delivery_intercept_m: float
    discovery_sample_hashes: tuple[str, ...]
    longitudinal_fit_r2: float
    lateral_fit_r2: float
    maximum_abs_yaw_delta_rad: float = 0.08
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.dynamic_lead_pass_policy.v1"

    def __post_init__(self) -> None:
        values = (
            self.longitudinal_intercept_slope_m_per_sec,
            self.longitudinal_intercept_m,
            self.lateral_delivery_slope_m_per_rad,
            self.lateral_delivery_intercept_m,
            self.longitudinal_fit_r2,
            self.lateral_fit_r2,
            self.maximum_abs_yaw_delta_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("lead-pass policy values must be finite")
        if abs(self.longitudinal_intercept_slope_m_per_sec) < 0.10:
            raise ValueError("lead-pass longitudinal calibration has no authority")
        if abs(self.lateral_delivery_slope_m_per_rad) < 0.50:
            raise ValueError("lead-pass lateral calibration has no authority")
        if not 0.0 <= self.longitudinal_fit_r2 <= 1.0:
            raise ValueError("lead-pass longitudinal fit R2 is invalid")
        if not 0.0 <= self.lateral_fit_r2 <= 1.0:
            raise ValueError("lead-pass lateral fit R2 is invalid")
        if not 0.02 <= self.maximum_abs_yaw_delta_rad <= 0.12:
            raise ValueError("lead-pass yaw envelope is invalid")
        if len(self.discovery_sample_hashes) < 6 or len(set(self.discovery_sample_hashes)) != len(
            self.discovery_sample_hashes
        ):
            raise ValueError("lead-pass policy needs distinct discovery samples")
        if any(
            not value.startswith("sha256:") or len(value) != 71
            for value in self.discovery_sample_hashes
        ):
            raise ValueError("lead-pass discovery sample hash is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("lead-pass policy must remain SIM_ONLY")

    @property
    def artifact_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def reception_target(
        self,
        *,
        receiver_phase_start_sec: float,
        receiver_lateral_lane_m: float,
        ball_height_m: float = 0.115,
    ) -> tuple[float, float, float]:
        """Predict the future strike pocket from a receiver phase and lane."""

        if not all(
            math.isfinite(value)
            for value in (receiver_phase_start_sec, receiver_lateral_lane_m, ball_height_m)
        ):
            raise ValueError("lead-pass target request must be finite")
        target_x = (
            self.longitudinal_intercept_slope_m_per_sec * receiver_phase_start_sec
            + self.longitudinal_intercept_m
        )
        if not 0.80 <= target_x <= 1.40:
            raise ValueError("predicted lead-pass target leaves the qualified strike pocket")
        if not -0.30 <= receiver_lateral_lane_m <= 0.30:
            raise ValueError("lead-pass lane leaves the qualified strike pocket")
        if not 0.105 <= ball_height_m <= 0.130:
            raise ValueError("lead-pass height leaves the rolling-ball pocket")
        return target_x, receiver_lateral_lane_m, ball_height_m

    def passer_yaw_delta(self, *, target_lateral_m: float) -> float:
        """Invert the learned lateral plant and return the executed yaw residual."""

        if not math.isfinite(target_lateral_m):
            raise ValueError("lead-pass lateral target must be finite")
        value = (
            target_lateral_m - self.lateral_delivery_intercept_m
        ) / self.lateral_delivery_slope_m_per_rad
        if abs(value) > self.maximum_abs_yaw_delta_rad:
            raise ValueError("lead-pass action exceeds the calibrated yaw envelope")
        return float(value)

    def passer_world_yaw(self, *, target_lateral_m: float) -> float:
        """Return an equivalent yaw in MuJoCo's accepted [-pi, pi] interval."""

        delta = self.passer_yaw_delta(target_lateral_m=target_lateral_m)
        return math.pi + delta if delta <= 0.0 else -math.pi + delta


def fit_dynamic_lead_pass_policy(
    samples: tuple[LeadPassCalibrationSample, ...],
) -> DynamicLeadPassPolicy:
    """Fit a bounded policy from safe discovery rollouts only."""

    safe = tuple(item for item in samples if item.safe)
    if len(safe) != len(samples) or len(safe) < 6:
        raise ValueError("lead-pass calibration requires at least six safe samples")
    longitudinal = tuple(item for item in safe if abs(item.passer_yaw_delta_rad) <= 1.0e-12)
    lateral = tuple(
        item for item in safe if math.isclose(item.receiver_phase_start_sec, 1.96, abs_tol=1.0e-12)
    )
    if len({item.receiver_phase_start_sec for item in longitudinal}) < 3:
        raise ValueError("lead-pass longitudinal calibration needs three receiver phases")
    if len({item.passer_yaw_delta_rad for item in lateral}) < 5:
        raise ValueError("lead-pass lateral calibration needs five yaw probes")
    long_x = np.asarray([item.receiver_phase_start_sec for item in longitudinal], dtype=np.float64)
    long_y = np.asarray([item.delivery_position_m[0] for item in longitudinal], dtype=np.float64)
    lat_x = np.asarray([item.passer_yaw_delta_rad for item in lateral], dtype=np.float64)
    lat_y = np.asarray([item.delivery_position_m[1] for item in lateral], dtype=np.float64)
    long_slope, long_intercept = np.polyfit(long_x, long_y, deg=1)
    lat_slope, lat_intercept = np.polyfit(lat_x, lat_y, deg=1)
    return DynamicLeadPassPolicy(
        longitudinal_intercept_slope_m_per_sec=float(long_slope),
        longitudinal_intercept_m=float(long_intercept),
        lateral_delivery_slope_m_per_rad=float(lat_slope),
        lateral_delivery_intercept_m=float(lat_intercept),
        discovery_sample_hashes=tuple(str(hash_json(asdict(item))) for item in safe),
        longitudinal_fit_r2=_r2(long_y, long_slope * long_x + long_intercept),
        lateral_fit_r2=_r2(lat_y, lat_slope * lat_x + lat_intercept),
    )


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum(np.square(observed - predicted)))
    total = float(np.sum(np.square(observed - float(np.mean(observed)))))
    if total <= 1.0e-15:
        return 0.0
    return float(np.clip(1.0 - residual / total, 0.0, 1.0))


__all__ = [
    "DynamicLeadPassPolicy",
    "LeadPassCalibrationSample",
    "fit_dynamic_lead_pass_policy",
]
