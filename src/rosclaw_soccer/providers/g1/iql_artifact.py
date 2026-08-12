"""Safe NumPy actor artifacts for the bundled G1 simulation provider.

This is the inference-only subset needed by Soccer Academy.  Recovery training
and promotion remain outside this provider, and every loaded candidate is
SIM_ONLY, no-pickle, hash-bound, and unevaluated.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from rosclaw.feedback.contracts import canonical_hash

from rosclaw_soccer.growth.approach_strike_contracts import STATE_FEATURES


@dataclass(frozen=True)
class IQLResidualGuardConfig:
    """Bound an offline actor to a small correction around a proven controller.

    The standardized envelope is deliberately only a support heuristic, not a
    calibrated OOD probability. Rejected states fall back to the structured
    controller and every accepted correction remains amplitude bounded.
    """

    residual_fraction: float = 0.05
    maximum_residual_nm: float = 2.0
    maximum_standardized_rms: float = 4.0
    maximum_standardized_abs: float = 20.0
    joint_group: str = "legs"
    schema_version: str = "rosclaw.growth.iql_residual_guard_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.residual_fraction,
            self.maximum_residual_nm,
            self.maximum_standardized_rms,
            self.maximum_standardized_abs,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("IQL residual guard values must be finite")
        if not 0.0 < self.residual_fraction <= 0.50:
            raise ValueError("IQL residual fraction must be in (0, 0.50]")
        if not 0.10 <= self.maximum_residual_nm <= 20.0:
            raise ValueError("IQL maximum residual must be in [0.10, 20] Nm")
        if not 0.50 <= self.maximum_standardized_rms <= 20.0:
            raise ValueError("IQL standardized RMS bound must be in [0.50, 20]")
        if not 1.0 <= self.maximum_standardized_abs <= 100.0:
            raise ValueError("IQL standardized absolute bound must be in [1, 100]")
        if self.joint_group not in {"legs", "lower_body", "whole_body"}:
            raise ValueError("IQL residual joint group is invalid")


@dataclass(frozen=True)
class IQLResidualDecision:
    """One auditable residual action decision."""

    residual_torque: np.ndarray
    accepted: bool
    confidence: float
    standardized_rms: float
    standardized_abs: float
    peak_residual_nm: float
    reason: str


@dataclass(frozen=True)
class NumpyIQLActor:
    """No-pickle inference form of an unevaluated IQL recovery actor."""

    layer_weights: tuple[np.ndarray, ...]
    layer_biases: tuple[np.ndarray, ...]
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    candidate_hash: str
    state_features: tuple[str, ...]
    task_id: str
    actor_output: str

    @classmethod
    def load(cls, candidate_path: Path) -> NumpyIQLActor:
        metadata_path = candidate_path.expanduser().resolve()
        if not metadata_path.is_file() or metadata_path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("IQL candidate metadata is missing or oversized")
        candidate = json.loads(metadata_path.read_text(encoding="utf-8"))
        claimed = candidate.get("candidate_hash")
        unsigned = dict(candidate)
        unsigned.pop("candidate_hash", None)
        if claimed != canonical_hash(unsigned):
            raise ValueError("IQL candidate hash mismatch")
        if candidate.get("schema_version") != "rosclaw.growth.iql_candidate.v1":
            raise ValueError("unsupported IQL candidate schema")
        if candidate.get("status") != "CANDIDATE_UNEVALUATED":
            raise ValueError("only an unevaluated IQL candidate can enter SIM-only evaluation")
        for name, expected in (
            ("activation_ceiling", "SIM_ONLY"),
            ("promotion_authorized", False),
            ("hardware_command_sent", False),
        ):
            if candidate.get(name) != expected:
                raise ValueError(f"IQL candidate requires {name}={expected!r}")
        artifact = candidate.get("artifact", {})
        if artifact.get("format") != "numpy_npz_no_pickle":
            raise ValueError("IQL actor requires the safe NumPy artifact format")
        actor_output = artifact.get("actor_output")
        if (
            actor_output
            not in {
                "executed_torque_nm",
                "sim_teacher_residual_torque_nm",
            }
            or artifact.get("learned_output_fraction") != 1.0
        ):
            raise ValueError("IQL actor output semantics are incompatible")
        weights_path = Path(str(artifact.get("weights_path", ""))).expanduser().resolve()
        if weights_path.parent != metadata_path.parent:
            raise ValueError("IQL actor weights must be adjacent to candidate metadata")
        if not weights_path.is_file() or weights_path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("IQL actor weights are missing or oversized")
        if _file_hash(weights_path) != artifact.get("weights_hash"):
            raise ValueError("IQL actor weight hash mismatch")
        with np.load(weights_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}
        if _array_content_hash(arrays) != artifact.get("weights_content_hash"):
            raise ValueError("IQL actor weight content hash mismatch")
        required = {
            "net__0__weight",
            "net__0__bias",
            "net__2__weight",
            "net__2__bias",
            "net__4__weight",
            "net__4__bias",
            "state_mean",
            "state_std",
            "action_mean",
            "action_std",
        }
        missing = sorted(required.difference(arrays))
        if missing:
            raise ValueError(f"IQL actor artifact is missing arrays: {missing}")
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("IQL actor artifact contains non-finite arrays")
        weights = tuple(arrays[f"net__{index}__weight"] for index in (0, 2, 4))
        biases = tuple(arrays[f"net__{index}__bias"] for index in (0, 2, 4))
        raw_features = artifact.get("state_features", list(STATE_FEATURES))
        if not isinstance(raw_features, list) or not all(
            isinstance(item, str) and item for item in raw_features
        ):
            raise ValueError("IQL actor state feature contract is invalid")
        state_features = tuple(raw_features)
        state_size = len(state_features)
        if weights[0].shape[1] != state_size or weights[-1].shape[0] != 29:
            raise ValueError("IQL actor input/output contract mismatch")
        if any(
            weight.shape[0] != bias.shape[0] for weight, bias in zip(weights, biases, strict=True)
        ):
            raise ValueError("IQL actor layer bias contract mismatch")
        if weights[1].shape[1] != weights[0].shape[0] or weights[2].shape[1] != weights[1].shape[0]:
            raise ValueError("IQL actor hidden layer contract mismatch")
        if arrays["state_mean"].shape != (state_size,) or arrays["state_std"].shape != (
            state_size,
        ):
            raise ValueError("IQL actor state normalization contract mismatch")
        if arrays["action_mean"].shape != (29,) or arrays["action_std"].shape != (29,):
            raise ValueError("IQL actor action normalization contract mismatch")
        if np.any(arrays["state_std"] <= 0.0) or np.any(arrays["action_std"] <= 0.0):
            raise ValueError("IQL actor normalization scales must be positive")
        return cls(
            layer_weights=weights,
            layer_biases=biases,
            state_mean=arrays["state_mean"],
            state_std=arrays["state_std"],
            action_mean=arrays["action_mean"],
            action_std=arrays["action_std"],
            candidate_hash=str(claimed),
            state_features=state_features,
            task_id=str(candidate.get("task_id", "g1_post_impact_recovery")),
            actor_output=str(actor_output),
        )

    def action(self, state: np.ndarray) -> NDArray[np.float64]:
        value = np.asarray(state, dtype=np.float32)
        if value.shape != self.state_mean.shape or not np.all(np.isfinite(value)):
            raise ValueError("IQL actor state must be one finite state vector")
        value = (value - self.state_mean) / self.state_std
        for index, (weight, bias) in enumerate(
            zip(self.layer_weights, self.layer_biases, strict=True)
        ):
            value = weight @ value + bias
            if index < len(self.layer_weights) - 1:
                value = value / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))
        action = value * self.action_std + self.action_mean
        if action.shape != (29,) or not np.all(np.isfinite(action)):
            raise RuntimeError("IQL actor produced an invalid action")
        return cast(NDArray[np.float64], action.astype(np.float64))

    def standardized_state(self, state: np.ndarray) -> NDArray[np.float64]:
        """Return the actor's frozen standardized state with strict validation."""

        value = np.asarray(state, dtype=np.float32)
        if value.shape != self.state_mean.shape or not np.all(np.isfinite(value)):
            raise ValueError("IQL actor state must be one finite state vector")
        standardized = (value - self.state_mean) / self.state_std
        if not np.all(np.isfinite(standardized)):
            raise RuntimeError("IQL actor standardized state is non-finite")
        return cast(NDArray[np.float64], standardized.astype(np.float64))


@dataclass(frozen=True)
class SupportBoundIQLResidualActor:
    """Use an IQL actor only as an envelope-gated residual torque proposer."""

    actor: NumpyIQLActor
    config: IQLResidualGuardConfig

    @classmethod
    def load(
        cls,
        candidate_path: Path,
        config: IQLResidualGuardConfig | None = None,
    ) -> SupportBoundIQLResidualActor:
        return cls(
            actor=NumpyIQLActor.load(candidate_path),
            config=config or IQLResidualGuardConfig(),
        )

    @property
    def candidate_hash(self) -> str:
        return self.actor.candidate_hash

    def action(self, state: np.ndarray, baseline_torque: np.ndarray) -> IQLResidualDecision:
        baseline = np.asarray(baseline_torque, dtype=np.float64)
        if baseline.shape != (29,) or not np.all(np.isfinite(baseline)):
            raise ValueError("IQL residual baseline must be one finite 29-joint torque")
        standardized = self.actor.standardized_state(state)
        rms = float(np.sqrt(np.mean(np.square(np.clip(standardized, -1e3, 1e3)))))
        maximum = float(np.max(np.abs(standardized)))
        supported = bool(
            rms <= self.config.maximum_standardized_rms
            and maximum <= self.config.maximum_standardized_abs
        )
        if not supported:
            return IQLResidualDecision(
                residual_torque=np.zeros(29, dtype=np.float64),
                accepted=False,
                confidence=0.0,
                standardized_rms=rms,
                standardized_abs=maximum,
                peak_residual_nm=0.0,
                reason="outside_standardized_support_envelope",
            )
        learned = self.actor.action(state)
        proposed_residual = (
            learned
            if self.actor.actor_output == "sim_teacher_residual_torque_nm"
            else learned - baseline
        )
        residual = np.clip(
            proposed_residual,
            -self.config.maximum_residual_nm,
            self.config.maximum_residual_nm,
        )
        mask = np.zeros(29, dtype=np.float64)
        stop = {"legs": 12, "lower_body": 15, "whole_body": 29}[self.config.joint_group]
        mask[:stop] = 1.0
        # Confidence decays smoothly inside the admitted envelope. It never
        # enlarges residual_fraction and becomes zero at the RMS boundary.
        confidence = max(0.0, 1.0 - rms / self.config.maximum_standardized_rms)
        residual = residual * mask * self.config.residual_fraction * confidence
        peak = float(np.max(np.abs(residual)))
        return IQLResidualDecision(
            residual_torque=residual.astype(np.float64),
            accepted=True,
            confidence=confidence,
            standardized_rms=rms,
            standardized_abs=maximum,
            peak_residual_nm=peak,
            reason=(
                "accepted_bounded_teacher_distillation_residual"
                if self.actor.actor_output == "sim_teacher_residual_torque_nm"
                else "accepted_bounded_residual"
            ),
        )


def _array_content_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "IQLResidualDecision",
    "IQLResidualGuardConfig",
    "NumpyIQLActor",
    "SupportBoundIQLResidualActor",
]
