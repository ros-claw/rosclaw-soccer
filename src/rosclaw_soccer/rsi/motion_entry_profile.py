"""SIM-only binding of a source-motion configuration to its entry estimate.

Not a Core capability/permit or a dynamically admitted skill. A historical entry
point cannot silently change meaning when a motion is cropped or reflected.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
from rosclaw_soccer.rsi.bilateral_entry import contact_foot_context
from rosclaw_soccer.rsi.entry_frame_transport import transport_entry_frame
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _hash(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("explicit SHA256 motion/entry lineage required")


@dataclass(frozen=True)
class MotionConfiguration:
    world_hash: str
    motion_artifact_hash: str
    side_source_pose_hash: str
    reference_side: str
    source_window: tuple[int, int]

    def __post_init__(self) -> None:
        for h in (self.world_hash, self.motion_artifact_hash, self.side_source_pose_hash):
            _hash(h)
        if (
            self.reference_side not in ("right", "left")
            or type(self.source_window) is not tuple
            or len(self.source_window) != 2
            or any(type(i) is not int for i in self.source_window)
            or self.source_window not in ((286, 767), (466, 767))
        ):
            raise ValueError("fixed source frame/side configuration required")

    def report(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.motion_entry_configuration.v1",
            **asdict(self),
            source_frequency_hz=120,
            entry_intervals=50,
            model_variant="low_latency",
            wrist_reference_policy="six_wrists_measured_entry_quarantine",
            requires_unchanged_PD_and_world=True,
            activation_ceiling="SIM_ONLY",
            hardware_authorized=False,
        )

    @property
    def contract_hash(self) -> str:
        return hash_json(self.report())


@dataclass(frozen=True)
class ConfiguredEntryEstimate(SkillEntryAffordance):
    parent_affordance_hash: str
    motion_configuration_hash: str
    geometry_transport_hash: str

    def __post_init__(self) -> None:
        super().__post_init__()
        for h in (
            self.parent_affordance_hash,
            self.motion_configuration_hash,
            self.geometry_transport_hash,
        ):
            _hash(h)

    @property
    def contract_hash(self) -> str:
        return hash_json(
            dict(
                schema="soccer.rsi.configured_entry_estimate.v1",
                **asdict(self),
                prior_kind="SE2_source_frame_transport_not_dynamically_calibrated",
                historical_fields_are_ancestor_lineage_not_new_window_observations=True,
                contact_qualified=False,
                dynamic_feasibility_proven=False,
                neural_training_updates=0,
                activation_ceiling="SIM_ONLY",
                hardware_authorized=False,
            )
        )


@dataclass(frozen=True)
class BoundMotionEntry:
    configuration: MotionConfiguration
    entry_affordance_hash: str
    entry_estimate_kind: str

    def __post_init__(self) -> None:
        _hash(self.entry_affordance_hash)
        if self.entry_estimate_kind not in (
            "historical_side_prior",
            "source_frame_transported_estimate",
            "native_paired_displacement_estimate",
        ):
            raise ValueError("declared historical or transported estimate required")

    def report(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.bound_motion_entry.v1",
            configuration=self.configuration.report(),
            configuration_hash=self.configuration.contract_hash,
            entry_affordance_hash=self.entry_affordance_hash,
            entry_estimate_kind=self.entry_estimate_kind,
            reference_side_not_observed_foot=True,
            requires_known_before_navigation=True,
            contact_qualified=False,
            dynamic_feasibility_proven=False,
            activation_ceiling="SIM_ONLY",
            hardware_authorized=False,
        )

    @property
    def contract_hash(self) -> str:
        return hash_json(self.report())

    def validate(
        self,
        source: np.ndarray[Any, Any],
        entry: SkillEntryAffordance,
        *,
        world_hash: str,
        source_window: tuple[int, int],
        reference_side: str,
    ) -> None:
        self.configuration.__post_init__()
        entry.__post_init__()
        if (
            type(source) is not np.ndarray
            or source.ndim != 2
            or source.shape[1] != 36
            or not 767 <= len(source) <= 10000
            or source.dtype != np.float64
            or not np.isfinite(source).all()
            or np.any(abs(np.linalg.norm(source[:, 3:7], axis=1) - 1) > 1e-5)
            or np.any(abs(source[:, 7:]) > 3.2)
        ):
            raise ValueError("finite canonical36 source required before any episode allocation")
        if (
            world_hash != self.configuration.world_hash
            or source_window != self.configuration.source_window
            or reference_side != self.configuration.reference_side
            or entry.contract_hash != self.entry_affordance_hash
            or hash_bytes(source.tobytes()) != self.configuration.side_source_pose_hash
        ):
            raise ValueError("motion/entry/world/side binding mismatch; no episode may start")
        if (
            hasattr(entry, "motion_configuration_hash")
            and entry.motion_configuration_hash != self.configuration.contract_hash
        ):
            raise ValueError("transported entry belongs to a different motion configuration")


def build_motion_entry(
    source: np.ndarray[Any, Any],
    original_prior: SkillEntryAffordance,
    *,
    world_hash: str,
    motion_artifact_hash: str,
    reference_side: str,
    source_window: tuple[int, int],
    transport: bool,
) -> tuple[BoundMotionEntry, SkillEntryAffordance, np.ndarray[Any, Any]]:
    if type(transport) is not bool or (transport and source_window != (466, 767)):
        raise ValueError("only declared compact source may use transported entry")
    if type(original_prior) is not SkillEntryAffordance:
        raise ValueError("start from the original right prior; no repeated reflection/transport")
    poses, entry, _ = contact_foot_context(source, original_prior, reference_side)
    owned = poses.copy()
    config = MotionConfiguration(
        world_hash, motion_artifact_hash, hash_bytes(owned.tobytes()), reference_side, source_window
    )
    if transport:
        t = transport_entry_frame(
            entry,
            owned[:, :7].copy(),
            original_source_frame=286,
            target_source_frame=466,
            source_motion_artifact_hash=motion_artifact_hash,
        )
        entry = ConfiguredEntryEstimate(
            t.transported_ball_offset_xy_m,
            entry.historical_frame,
            entry.historical_forward_speed_mps,
            entry.historical_artifact_hash,
            entry.contract_hash,
            config.contract_hash,
            t.contract_hash,
        )
    bound = BoundMotionEntry(
        config,
        entry.contract_hash,
        "source_frame_transported_estimate" if transport else "historical_side_prior",
    )
    bound.validate(
        owned,
        entry,
        world_hash=world_hash,
        source_window=source_window,
        reference_side=reference_side,
    )
    owned.setflags(write=False)
    return bound, entry, owned
