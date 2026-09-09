"""Frozen, data-trained approach selection; returns a proposal, never motion.

The four features describe a qualified standing-start, ground-ball course.
This is not a general goalkeeper, aerial-ball or arbitrary-posture policy.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def load_bounded_reference_parameters(
    weights: Path, expected_hash: object, shapes: dict[str, tuple[int, ...]]
) -> tuple[dict[str, np.ndarray], str]:
    """Read numeric reference-selector weights after bounded ZIP/NPY checks."""
    if (
        not isinstance(expected_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash) is None
    ):
        raise ValueError("explicit reference checkpoint hash required")
    if weights.stat().st_size > 1_000_000:
        raise ValueError("approach checkpoint exceeds size bound")
    raw = weights.read_bytes()
    policy_hash = str(hash_bytes(raw))
    if policy_hash != expected_hash:
        raise ValueError("approach checkpoint hash differs")
    import io

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if (
            set(names) != {name + ".npy" for name in shapes}
            or len(names) != len(shapes)
            or sum(item.file_size for item in archive.infolist()) > 1_000_000
        ):
            raise ValueError("unexpected or oversized approach tensors")
        # A small ZIP member can still declare a huge array in its NPY
        # header. Validate the header before np.load can allocate it.
        for key, shape in shapes.items():
            entry = archive.getinfo(key + ".npy")
            with archive.open(entry) as stream:
                # NumPy's public NPY header helpers lack type stubs.
                version = np.lib.format.read_magic(stream)  # type: ignore[no-untyped-call]
                if version == (1, 0):
                    declared, _, dtype = np.lib.format.read_array_header_1_0(stream)  # type: ignore[no-untyped-call]
                elif version == (2, 0):
                    declared, _, dtype = np.lib.format.read_array_header_2_0(stream)  # type: ignore[no-untyped-call]
                else:
                    raise ValueError("unsupported approach array header")
                if (
                    declared != shape
                    or dtype != np.dtype(np.float32)
                    or entry.file_size != stream.tell() + int(np.prod(shape)) * 4
                ):
                    raise ValueError("approach array header differs from bounded shape")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        parameters = {key: archive[key].copy() for key in shapes}
    for key, value in parameters.items():
        if value.shape != shapes[key] or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("finite float32 approach tensor shape required")
        value.setflags(write=False)
    return parameters, policy_hash


@dataclass(frozen=True)
class ApproachReferenceProposal:
    reference_index: int
    predicted_utility: float
    policy_hash: str
    observation_hash: str
    activation_ceiling: str = "SIM_ONLY"


class G1ApproachRouter:
    """4 → 64 → 64 → K utility regressor with exact downstream bindings.

    Utilities are not probabilities or evidence of success. The caller must
    qualify the body posture and reference bank, then retain ordinary motor,
    contact and promotion gates. There is no automatic policy activation.
    """

    def __init__(
        self,
        manifest: Path,
        *,
        expected_body_actor_hash: str,
        expected_reference_library_hash: str,
    ) -> None:
        for value in (expected_body_actor_hash, expected_reference_library_hash):
            if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ValueError("explicit body and reference content bindings required")
        if manifest.stat().st_size > 32_768:
            raise ValueError("approach manifest exceeds size bound")
        metadata = json.loads(manifest.read_text())
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema") != "rosclaw_soccer.g1_approach_router.v1"
            or metadata.get("activation_ceiling") != "SIM_ONLY"
            or metadata.get("body_actor_hash") != expected_body_actor_hash
            or metadata.get("reference_library_hash") != expected_reference_library_hash
        ):
            raise ValueError("approach router body/reference or simulation boundary differs")
        choices = metadata.get("reference_indices")
        if (
            not isinstance(choices, list)
            or not 2 <= len(choices) <= 32
            or any(type(i) is not int or not 0 <= i < 4096 for i in choices)
            or len(set(choices)) != len(choices)
        ):
            raise ValueError("unique bounded reference indices required")
        self.reference_indices = tuple(choices)
        self._scales = self._vector(metadata.get("feature_scales"))
        self._lower = self._vector(metadata.get("feature_lower"))
        self._upper = self._vector(metadata.get("feature_upper"))
        if np.any(self._scales <= 0) or np.any(self._lower >= self._upper):
            raise ValueError("positive feature scales and ordered admitted domain required")
        shapes = {
            "0.weight": (64, 4),
            "0.bias": (64,),
            "2.weight": (64, 64),
            "2.bias": (64,),
            "4.weight": (len(choices), 64),
            "4.bias": (len(choices),),
        }
        parameters, self.policy_hash = load_bounded_reference_parameters(
            manifest.with_suffix(".npz"), metadata.get("weights_hash"), shapes
        )
        self._parameters = parameters
        self.contract_hash = str(hash_json(metadata))

    @staticmethod
    def _vector(values: object) -> np.ndarray:
        if (
            not isinstance(values, (tuple, list))
            or len(values) != 4
            or any(type(x) not in (float, int) for x in values)
        ):
            raise ValueError("four finite numeric approach features required")
        result = np.asarray(values, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("four finite numeric approach features required")
        return result

    def propose(
        self, *, ball_offset_xy_m: tuple[float, float], ball_velocity_xy_mps: tuple[float, float]
    ) -> ApproachReferenceProposal:
        if len(ball_offset_xy_m) != 2 or len(ball_velocity_xy_mps) != 2:
            raise ValueError("planar ball offset and velocity required")
        raw = self._vector([*ball_offset_xy_m, *ball_velocity_xy_mps])
        if np.any(raw < self._lower) or np.any(raw > self._upper):
            raise ValueError("ball state outside the learned approach domain")
        value = (raw / self._scales).astype(np.float32)
        for index in (0, 2, 4):
            value = (
                value @ self._parameters[f"{index}.weight"].T + self._parameters[f"{index}.bias"]
            )
            if index != 4:
                value = np.tanh(value)
        if not np.isfinite(value).all():
            raise FloatingPointError("nonfinite approach utility")
        selected = int(np.argmax(value))
        return ApproachReferenceProposal(
            reference_index=self.reference_indices[selected],
            predicted_utility=float(value[selected]),
            policy_hash=self.policy_hash,
            observation_hash=str(hash_json(raw.tolist())),
        )
