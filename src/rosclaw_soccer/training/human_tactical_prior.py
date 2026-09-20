"""NumPy-only inference for the four-intent human tracking research prior.

Outputs are attack-oriented displacement proposals, not robot commands, skill
success probabilities, or calibrated intent confidence. No execution is granted.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes

INTENTS = ("SUPPORT", "ADVANCE", "RUN_BEHIND", "PRESS")
SCHEMA = "soccer.human_tactical_prior.20x64x64.4intent.v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_SHAPES = {
    "encoder.0.weight": (64, 20),
    "encoder.0.bias": (64,),
    "encoder.2.weight": (64, 64),
    "encoder.2.bias": (64,),
    "waypoint.weight": (2, 64),
    "waypoint.bias": (2,),
    "intent.weight": (4, 64),
    "intent.bias": (4,),
    "mean": (20,),
    "scale": (20,),
    "target_scale": (2,),
}
_METADATA = {"schema", "checkpoint_hash", "dataset_manifest_hash", "intent_labels"}


class HumanTacticalPrior:
    """Load a hash-pinned, non-pickle export; retain its training provenance.

    A digest verifies content identity, not who trained or approved the model.
    This class never interprets human displacement as G1 movement permission.
    """

    def __init__(self, path: Path, *, expected_hash: str) -> None:
        if not isinstance(expected_hash, str) or not _HASH.fullmatch(expected_hash):
            raise ValueError("explicit SHA-256 artifact identity required")
        with path.open("rb") as handle:
            content = handle.read(1_000_001)
        if len(content) > 1_000_000 or hash_bytes(content) != expected_hash:
            raise ValueError("prior artifact size or digest mismatch")
        with zipfile.ZipFile(io.BytesIO(content)) as compressed:
            if sum(item.file_size for item in compressed.infolist()) > 1_000_000:
                raise ValueError("expanded prior artifact is too large")
        weights: dict[str, NDArray[np.float32]] = {}
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)) or set(archive.files) != (
                set(_SHAPES) | _METADATA
            ):
                raise ValueError("unexpected prior export fields")
            for name in ("schema", "checkpoint_hash", "dataset_manifest_hash"):
                value = archive[name]
                if value.shape != () or value.dtype.kind != "U":
                    raise ValueError("scalar text provenance required")
            if str(archive["schema"]) != SCHEMA:
                raise ValueError("unsupported feature/architecture schema")
            self.checkpoint_hash = str(archive["checkpoint_hash"])
            self.dataset_manifest_hash = str(archive["dataset_manifest_hash"])
            if any(
                not _HASH.fullmatch(value)
                for value in (self.checkpoint_hash, self.dataset_manifest_hash)
            ):
                raise ValueError("checkpoint and dataset identities required")
            labels = archive["intent_labels"]
            if labels.shape != (4,) or labels.dtype.kind != "U" or tuple(labels) != INTENTS:
                raise ValueError("intent order mismatch")
            for name, shape in _SHAPES.items():
                value = archive[name]
                if (
                    value.shape != shape
                    or value.dtype != np.float32
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"invalid prior tensor: {name}")
                weights[name] = value.copy()
                weights[name].flags.writeable = False
        if any(np.any(weights[name] <= 0) for name in ("scale", "target_scale")):
            raise ValueError("normalization scales must be positive")
        self._weights: Mapping[str, NDArray[np.float32]] = MappingProxyType(weights)
        self.artifact_hash = expected_hash

    def predict(
        self,
        features: NDArray[np.float32],
        *,
        pitch_dimensions_m: tuple[float, float],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return half-second delta in metres and four raw intent logits.

        Features: own xy, past-half-second dxy, relative-ball xy, past-ball dxy,
        nearest teammate/opponent relative xy; all normalized by pitch axes and
        rotated toward +x attack. Then role one-hot (GK, DF, MF, FW, unknown),
        possession one-hot (own, opponent, unknown). No future fields allowed.
        This v1 was trained on adult human pitches; scaling to robot pitches is
        deliberately rejected until a separate transfer model is evaluated.
        """
        if (
            not isinstance(features, np.ndarray)
            or features.dtype != np.float32
            or features.ndim != 2
            or features.shape[1] != 20
            or not 1 <= len(features) <= 100_000
            or not np.isfinite(features).all()
        ):
            raise ValueError("finite float32 [batch,20] feature matrix required")
        for group in (features[:, 12:17], features[:, 17:20]):
            if not np.all((group == 0) | (group == 1)) or not np.all(group.sum(axis=1) == 1):
                raise ValueError("explicit role/possession one-hot required; missing is not zero")
        dims = np.asarray(pitch_dimensions_m, dtype=np.float32)
        if (
            dims.shape != (2,)
            or not np.isfinite(dims).all()
            or not 50 <= dims[0] <= 150
            or not 30 <= dims[1] <= 100
        ):
            raise ValueError("prior requires its human-pitch dimensional domain")
        w = self._weights
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                hidden = (features - w["mean"]) / w["scale"]
                for layer in ("encoder.0", "encoder.2"):
                    hidden = np.tanh(hidden @ w[f"{layer}.weight"].T + w[f"{layer}.bias"])
                delta = (
                    features[:, 2:4] * dims
                    + (hidden @ w["waypoint.weight"].T + w["waypoint.bias"]) * w["target_scale"]
                )
                logits = hidden @ w["intent.weight"].T + w["intent.bias"]
            except FloatingPointError as exc:
                raise ValueError("non-finite prior computation") from exc
        if not np.isfinite(delta).all() or not np.isfinite(logits).all():
            raise ValueError("non-finite prior output")
        return delta.astype(np.float32, copy=False), logits.astype(np.float32, copy=False)
