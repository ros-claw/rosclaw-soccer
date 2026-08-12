"""Provenance-separated, relative-path-only motion dataset classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosclaw.dataset.contracts import DatasetSourceDescriptor

_LABEL_IDS = (
    "contact.annotation",
    "license.material",
    "motion.reference",
    "split.test",
    "split.train",
    "split.validation",
)


@dataclass(frozen=True)
class SoccerMotionDatasetSource:
    """Classify names without receiving roots, file handles, or runtime access."""

    source_id: str
    dataset_id: str
    source_uri: str
    revision: str
    motion_suffixes: tuple[str, ...]
    contact_markers: tuple[str, ...] = ()

    @property
    def descriptor(self) -> DatasetSourceDescriptor:
        from rosclaw.dataset.contracts import DatasetSourceDescriptor

        return DatasetSourceDescriptor(
            source_id=self.source_id,
            dataset_ids=(self.dataset_id,),
            label_ids=_LABEL_IDS,
            source_uri=self.source_uri,
            revision=self.revision,
        )

    def classify_file(self, dataset_id: str, relative_path: str) -> tuple[str, ...]:
        if dataset_id != self.dataset_id:
            return ()
        if not isinstance(relative_path, str):
            raise ValueError("dataset classifier requires a safe relative path")
        normalized = relative_path.replace("\\", "/").lower()
        if (
            not normalized
            or normalized.startswith("/")
            or ".." in normalized.split("/")
            or any(ord(character) < 32 for character in normalized)
        ):
            raise ValueError("dataset classifier requires a safe relative path")
        labels: set[str] = set()
        name = normalized.rsplit("/", 1)[-1]
        if name.startswith(("license", "readme", "citation")):
            labels.add("license.material")
        if any(marker in normalized for marker in self.contact_markers):
            labels.add("contact.annotation")
        if name.endswith(self.motion_suffixes):
            labels.add("motion.reference")
        parts = set(normalized.split("/"))
        if parts.intersection({"train", "training"}) or name.startswith("train"):
            labels.add("split.train")
        if parts.intersection({"val", "valid", "validation"}) or name.startswith("val"):
            labels.add("split.validation")
        if "test" in parts or name.startswith("test"):
            labels.add("split.test")
        return tuple(label for label in _LABEL_IDS if label in labels)


MOTIONDECODE_SOURCE = SoccerMotionDatasetSource(
    source_id="soccer.motiondecode",
    dataset_id="MotionDecode",
    source_uri="https://huggingface.co/datasets/CMRobot/MotionDecode",
    revision="f71451a3e3285e83f11fe8738fc1d4750cab84f2",
    motion_suffixes=(".csv",),
)
OMNICONTACT_SOURCE = SoccerMotionDatasetSource(
    source_id="soccer.omnicontact",
    dataset_id="OmniContact",
    source_uri="https://huggingface.co/datasets/lightcone02/OmniContact-Dataset",
    revision="d3bdc7aefd7d17f93feb1dc1005fc6964f2b5cf1",
    motion_suffixes=(".bvh", ".npy", ".npz"),
    contact_markers=("contact",),
)
G1_RETARGETED_MOTIONS_SOURCE = SoccerMotionDatasetSource(
    source_id="soccer.g1-retargeted-motions",
    dataset_id="g1-retargeted-motions",
    source_uri="https://huggingface.co/datasets/openhe/g1-retargeted-motions",
    revision="ed9c2541c2e6d518e9f7be61515e86f32daf59b8",
    motion_suffixes=(".pkl",),
)
SOCCER_MOTION_SOURCES = (
    G1_RETARGETED_MOTIONS_SOURCE,
    MOTIONDECODE_SOURCE,
    OMNICONTACT_SOURCE,
)

__all__ = [
    "G1_RETARGETED_MOTIONS_SOURCE",
    "MOTIONDECODE_SOURCE",
    "OMNICONTACT_SOURCE",
    "SOCCER_MOTION_SOURCES",
    "SoccerMotionDatasetSource",
]
