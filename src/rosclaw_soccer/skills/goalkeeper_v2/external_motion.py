"""License-isolated motion decoder for research goalkeeper demonstrations.

The separately licensed motion tensors remain outside the source checkout.
This module converts them into a numeric, content-addressed upper-body basis
for SIM_ONLY research.  The bundle is not Champion-eligible and cannot support
a commercial/publicity claim without separate permission from its authors.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json

_REFERENCE_COMMIT = "976a81ff19b7306bafbe923d2890066b68a85271"
_REFERENCE_REPOSITORY = "https://github.com/InternRobotics/Humanoid-Goalkeeper"
_LICENSE_RELATIVE_PATH = Path("LICENSE")
_MOTION_ROOT = Path("legged_gym/resources/datasets/goalkeeper")
_EXPECTED_LICENSE_HASH = "sha256:6c8cd1cdbe7accec4f63b6c3afb45ce0ffae9ed6abc0ca55acf5900b37970a82"
_FAMILIES = (
    "lefthand",
    "righthand",
    "leftjump",
    "rightjump",
    "leftstep",
    "rightstep",
)
_SOURCE_TO_G1 = (
    (12, 12),
    (13, 15),
    (14, 16),
    (15, 17),
    (16, 18),
    (17, 22),
    (18, 23),
    (19, 24),
    (20, 25),
)


@dataclass(frozen=True)
class ExternalGoalkeeperMotionManifest:
    archive_file: str
    archive_hash: str
    source_motion_hashes: tuple[tuple[str, str], ...]
    source_license_hash: str
    source_commit: str
    family_names: tuple[str, ...]
    frame_count: int = 25
    motion_duration_sec: float = 0.80
    attribution_required: bool = True
    commercial_use_allowed: bool = False
    share_alike_required: bool = True
    external_teacher_only: bool = True
    champion_eligible: bool = False
    activation_ceiling: str = "SIM_ONLY"
    repository: str = _REFERENCE_REPOSITORY
    schema_version: str = "rosclaw_soccer.external_goalkeeper_motion_manifest.v1"

    def __post_init__(self) -> None:
        if Path(self.archive_file).name != self.archive_file or not self.archive_file.endswith(
            ".npz"
        ):
            raise ValueError("external goalkeeper motion requires a sibling numeric archive")
        if self.family_names != _FAMILIES or len(self.source_motion_hashes) != len(_FAMILIES):
            raise ValueError("external goalkeeper motion family contract changed")
        if tuple(name for name, _ in self.source_motion_hashes) != _FAMILIES:
            raise ValueError("external goalkeeper motion provenance order changed")
        for value in (
            self.archive_hash,
            self.source_license_hash,
            *(value for _, value in self.source_motion_hashes),
        ):
            if not value.startswith("sha256:"):
                raise ValueError("external goalkeeper motion provenance requires hashes")
        if self.source_commit != _REFERENCE_COMMIT or self.repository != _REFERENCE_REPOSITORY:
            raise ValueError("external goalkeeper motion source identity changed")
        if self.frame_count != 25 or not math.isclose(
            self.motion_duration_sec, 0.80, abs_tol=1e-12
        ):
            raise ValueError("external goalkeeper motion timing contract changed")
        if (
            not self.attribution_required
            or self.commercial_use_allowed
            or not self.share_alike_required
            or not self.external_teacher_only
            or self.champion_eligible
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("external goalkeeper motion violated its research boundary")

    @property
    def manifest_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["source_motion_hashes"] = [list(item) for item in self.source_motion_hashes]
        value["family_names"] = list(self.family_names)
        if include_hash:
            value["manifest_hash"] = self.manifest_hash
        return value


class ExternalGoalkeeperMotionDecoder:
    """Interpolate a bounded 29-joint research motion basis by region."""

    def __init__(
        self,
        manifest: ExternalGoalkeeperMotionManifest,
        motions: dict[str, NDArray[np.float64]],
    ) -> None:
        self.manifest = manifest
        self._motions = motions

    def residual(self, *, region: int, elapsed_sec: float) -> NDArray[np.float64]:
        if not 0 <= region < len(_FAMILIES) or not math.isfinite(elapsed_sec):
            raise ValueError("external goalkeeper motion query is invalid")
        if elapsed_sec <= 0.0:
            return np.zeros(29, dtype=np.float64)
        duration = self.manifest.motion_duration_sec
        phase = float(np.clip(elapsed_sec / duration, 0.0, 1.0))
        trajectory = self._motions[_FAMILIES[region]]
        position = phase * (len(trajectory) - 1)
        lower = int(math.floor(position))
        upper = min(len(trajectory) - 1, lower + 1)
        fraction = position - lower
        residual = (1.0 - fraction) * trajectory[lower] + fraction * trajectory[upper]
        if elapsed_sec > duration:
            release = float(np.clip(1.0 - (elapsed_sec - duration) / 0.40, 0.0, 1.0))
            residual *= release * release * (3.0 - 2.0 * release)
        return np.asarray(residual, dtype=np.float64)


def build_external_goalkeeper_motion_bundle(
    *,
    reference_checkout: Path,
    output_manifest_path: Path,
    source_checkout: Path,
) -> ExternalGoalkeeperMotionManifest:
    """Extract bounded upper-body sequences without vendoring source tensors."""

    root = reference_checkout.expanduser().resolve()
    output = output_manifest_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("external goalkeeper motion must remain outside source checkout")
    if output.suffix != ".json" or output.exists():
        raise ValueError("external goalkeeper motion manifest requires a new JSON path")
    license_path = root / _LICENSE_RELATIVE_PATH
    if _file_hash(license_path) != _EXPECTED_LICENSE_HASH:
        raise ValueError("external goalkeeper motion license changed")
    if "Attribution-NonCommercial-ShareAlike 4.0" not in license_path.read_text(encoding="utf-8"):
        raise ValueError("external goalkeeper motion license terms are unavailable")
    commit = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != _REFERENCE_COMMIT:
        raise ValueError("external goalkeeper motion checkout is not pinned")
    import torch

    archive: dict[str, NDArray[np.float32]] = {}
    hashes: list[tuple[str, str]] = []
    for family in _FAMILIES:
        path = root / _MOTION_ROOT / f"{family}.pt"
        hashes.append((family, _file_hash(path)))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        source = payload.get("joint_position") if isinstance(payload, dict) else None
        if source is None or source.ndim != 2 or source.shape[1] != 21:
            raise ValueError(f"external goalkeeper motion tensor changed: {family}")
        q = np.asarray(source.detach().cpu().numpy(), dtype=np.float64)
        upper = q[:, 12:21]
        energy = np.sqrt(np.mean(np.square(upper - upper[:1]), axis=1))
        peak = int(np.argmax(energy))
        start = max(0, peak - 24)
        indices = np.linspace(start, peak, 25)
        sampled = np.stack(
            [
                (1.0 - (index - math.floor(index))) * q[int(math.floor(index))]
                + (index - math.floor(index)) * q[min(len(q) - 1, int(math.ceil(index)))]
                for index in indices
            ]
        )
        sampled -= sampled[:1]
        mapped = np.zeros((25, 29), dtype=np.float64)
        for source_index, target_index in _SOURCE_TO_G1:
            mapped[:, target_index] = sampled[:, source_index]
        mapped = np.clip(mapped, -0.55, 0.55)
        if not np.all(np.isfinite(mapped)):
            raise ValueError(f"external goalkeeper motion is non-finite: {family}")
        archive[family] = np.asarray(mapped, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_path = output.with_suffix(".npz")
    if archive_path.exists():
        raise ValueError("external goalkeeper motion archive output already exists")
    np.savez_compressed(archive_path, **archive)  # type: ignore[arg-type]
    manifest = ExternalGoalkeeperMotionManifest(
        archive_file=archive_path.name,
        archive_hash=_file_hash(archive_path),
        source_motion_hashes=tuple(hashes),
        source_license_hash=_EXPECTED_LICENSE_HASH,
        source_commit=_REFERENCE_COMMIT,
        family_names=_FAMILIES,
    )
    output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_external_goalkeeper_motion_decoder(
    manifest_path: Path,
) -> ExternalGoalkeeperMotionDecoder:
    path = manifest_path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = str(payload.pop("manifest_hash", ""))
    try:
        payload["source_motion_hashes"] = tuple(
            (str(item[0]), str(item[1])) for item in payload["source_motion_hashes"]
        )
        payload["family_names"] = tuple(payload["family_names"])
        manifest = ExternalGoalkeeperMotionManifest(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("external goalkeeper motion manifest is invalid") from exc
    if claimed_hash != manifest.manifest_hash:
        raise ValueError("external goalkeeper motion manifest hash mismatch")
    archive_path = path.parent / manifest.archive_file
    if _file_hash(archive_path) != manifest.archive_hash:
        raise ValueError("external goalkeeper motion archive hash mismatch")
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != set(_FAMILIES):
            raise ValueError("external goalkeeper motion family set changed")
        motions = {name: np.asarray(archive[name], dtype=np.float64) for name in _FAMILIES}
    for name, value in motions.items():
        if value.shape != (manifest.frame_count, 29) or not np.all(np.isfinite(value)):
            raise ValueError(f"external goalkeeper motion archive is invalid: {name}")
        if np.any(value[:, :12]) or np.max(np.abs(value)) > 0.55 + 1e-9:
            raise ValueError(f"external goalkeeper motion exceeded its upper-body boundary: {name}")
    return ExternalGoalkeeperMotionDecoder(manifest, motions)


def _file_hash(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > 1024 * 1024 * 1024:
        raise ValueError(f"external goalkeeper motion file is unavailable or oversized: {path}")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "ExternalGoalkeeperMotionDecoder",
    "ExternalGoalkeeperMotionManifest",
    "build_external_goalkeeper_motion_bundle",
    "load_external_goalkeeper_motion_decoder",
]
