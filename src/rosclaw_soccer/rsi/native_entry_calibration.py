"""Fit an unqualified initiation estimate from paired actual-body displacements.

This is a two-coefficient least-squares model, not a neural policy update or
contact certificate. Only historical pre-contact points enter the fit. It does
not change source poses, a physical body, PD gains, or the task gate.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
from rosclaw_soccer.rsi.motion_entry_profile import BoundMotionEntry, _hash
from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class NativeEntryCalibration:
    motion_configuration_hash: str
    parent_affordance_hash: str
    audit_hash: str
    sample_hashes: tuple[str, str]
    mean_displacement_xy_m: tuple[float, float]
    maximum_training_residual_m: float

    def __post_init__(self) -> None:
        for h in (self.motion_configuration_hash, self.parent_affordance_hash, self.audit_hash):
            _hash(h)
        if type(self.sample_hashes) is not tuple or len(self.sample_hashes) != 2:
            raise ValueError("exact two historical paired samples required")
        for h in self.sample_hashes:
            _hash(h)
        if (
            self.sample_hashes[0] == self.sample_hashes[1]
            or type(self.mean_displacement_xy_m) is not tuple
            or len(self.mean_displacement_xy_m) != 2
            or not np.isfinite(self.mean_displacement_xy_m).all()
            or max(abs(x) for x in self.mean_displacement_xy_m) > 0.4
            or not np.isfinite(self.maximum_training_residual_m)
            or not 0 <= self.maximum_training_residual_m <= 0.02
        ):
            raise ValueError("finite bounded paired displacement fit required")

    def report(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.native_entry_calibration.v1",
            **asdict(self),
            fit="least_squares_constant_new_minus_old_actual_toe_displacement_xy",
            training_laws=[0, 3],
            withheld_law=4,
            source_cue_interval=526,
            original_phase_frame=140,
            compact_phase_frame=65,
            independent_world_contexts=1,
            height_or_velocity_compensation=False,
            dynamically_qualified=False,
            activation_ceiling="SIM_ONLY",
            neural_optimizer_updates=0,
            hardware_authorized=False,
        )

    @property
    def contract_hash(self) -> str:
        return hash_json(self.report())


def fit_displacements(
    samples: np.ndarray[Any, Any],
    *,
    configuration_hash: str,
    parent_affordance_hash: str,
    audit_hash: str,
    sample_hashes: tuple[str, str],
) -> NativeEntryCalibration:
    """One deterministic fit, no offsets/windows selected by future task scores."""
    if (
        type(samples) is not np.ndarray
        or samples.dtype != np.float64
        or samples.shape != (2, 2)
        or not np.isfinite(samples).all()
    ):
        raise ValueError("two finite float64 XY displacement samples required")
    mean = samples.mean(axis=0)
    return NativeEntryCalibration(
        configuration_hash,
        parent_affordance_hash,
        audit_hash,
        sample_hashes,
        (float(mean[0]), float(mean[1])),
        float(np.linalg.norm(samples - mean, axis=1).max()),
    )


@dataclass(frozen=True)
class NativeCalibratedEntry(SkillEntryAffordance):
    parent_affordance_hash: str
    motion_configuration_hash: str
    calibration_hash: str

    def __post_init__(self) -> None:
        super().__post_init__()
        for h in (
            self.parent_affordance_hash,
            self.motion_configuration_hash,
            self.calibration_hash,
        ):
            _hash(h)

    @property
    def contract_hash(self) -> str:
        return hash_json(
            dict(
                schema="soccer.rsi.native_calibrated_entry_estimate.v1",
                **asdict(self),
                prior_kind="paired_actual_precontact_toe_displacement_not_contact_certificate",
                historical_fields_are_ancestor_lineage_not_new_window_observations=True,
                contact_qualified=False,
                dynamic_feasibility_proven=False,
                neural_training_updates=0,
                activation_ceiling="SIM_ONLY",
                hardware_authorized=False,
            )
        )


def bind_native_entry(
    binding: BoundMotionEntry,
    parent: SkillEntryAffordance,
    fit: NativeEntryCalibration,
) -> tuple[BoundMotionEntry, NativeCalibratedEntry]:
    fit.__post_init__()
    config = binding.configuration
    if (
        config.reference_side != "right"
        or config.source_window != (466, 767)
        or type(parent) is not SkillEntryAffordance
        or binding.entry_estimate_kind != "historical_side_prior"
        or binding.entry_affordance_hash != parent.contract_hash
        or fit.parent_affordance_hash != parent.contract_hash
        or fit.motion_configuration_hash != config.contract_hash
    ):
        raise ValueError(
            "exact original right prior and corresponding compact body/source required"
        )
    xy = np.array(parent.ball_offset_xy_m) + fit.mean_displacement_xy_m
    entry = NativeCalibratedEntry(
        (float(xy[0]), float(xy[1])),
        parent.historical_frame,
        parent.historical_forward_speed_mps,
        parent.historical_artifact_hash,
        parent.contract_hash,
        config.contract_hash,
        fit.contract_hash,
    )
    return BoundMotionEntry(
        config, entry.contract_hash, "native_paired_displacement_estimate"
    ), entry
