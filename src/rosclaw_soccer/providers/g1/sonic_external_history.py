"""SIM-only history adapter from applied HG targets, never raw foreign actions.

An inverse action codec is a numerical representation of the actually applied
joint target, NOT a claim that SONIC emitted the foreign motor's action. Native
trace provenance and physical gain compatibility require separate experiments.
"""

import collections
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
from rosclaw_soccer.providers.g1.sonic_runup import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    G1SonicRunupController,
    _gravity_in_body,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _array(a: Any, shape: tuple[int, ...]) -> np.ndarray:
    if (
        type(a) is not np.ndarray
        or a.shape != shape
        or a.dtype != np.float64
        or not np.isfinite(a).all()
    ):
        raise ValueError("finite owned canonical float64 history input required")
    return a.copy()


def equivalent_sonic_action(
    target: np.ndarray, defaults: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, float]:
    target, defaults, scale = (_array(a, (29,)) for a in (target, defaults, scale))
    if np.any(scale <= 0) or np.max(abs(target)) > 10 or np.max(abs(defaults)) > 3.2:
        raise ValueError("positive actual SONIC scales and bounded HG29 target required")
    normalized = (target - defaults) / scale
    if np.max(abs(normalized)) > 100:
        raise ValueError("bounded equivalent previous action required; no clipping")
    action = normalized[MUJOCO_TO_ISAACLAB].astype(np.float32)
    reconstructed = defaults + action[ISAACLAB_TO_MUJOCO] * scale
    residual = float(np.max(abs(reconstructed - target)))
    if residual > 2e-7:
        raise ValueError("inverse previous-action float32 representation exceeds numeric tolerance")
    return action, residual


@dataclass(frozen=True)
class ExternalSonicHistory:
    samples: tuple[tuple[tuple[float, ...], ...], ...]
    last_qpos_bytes: bytes
    last_qvel_bytes: bytes
    last_target_bytes: bytes
    default_bytes: bytes
    scale_bytes: bytes
    trace_hash: str
    maximum_target_roundtrip_error_rad: float

    def __post_init__(self) -> None:
        if (
            type(self.samples) is not tuple
            or len(self.samples) != 10
            or any(
                type(row) is not tuple
                or tuple(len(field) for field in row) != (3, 29, 29, 29, 3)
                or any(type(field) is not tuple for field in row)
                or any(type(x) is not float or not np.isfinite(x) for field in row for x in field)
                for row in self.samples
            )
            or any(
                type(b) is not bytes or len(b) != width * 8
                for b, width in (
                    (self.last_qpos_bytes, 43),
                    (self.last_qvel_bytes, 41),
                    (self.last_target_bytes, 29),
                    (self.default_bytes, 29),
                    (self.scale_bytes, 29),
                )
            )
            or type(self.trace_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.trace_hash) is None
            or type(self.maximum_target_roundtrip_error_rad) is not float
            or not 0 <= self.maximum_target_roundtrip_error_rad <= 2e-7
        ):
            raise ValueError("immutable finite complete external history required")

    def report(self) -> dict[str, Any]:
        return dict(
            schema="soccer.sonic_external_applied_target_history.v1",
            trace_hash=self.trace_hash,
            samples=10,
            synthetic_history_padding=False,
            previous_action_kind="inverse_codec_of_actual_applied_target_not_foreign_raw_action",
            maximum_target_roundtrip_error_rad=self.maximum_target_roundtrip_error_rad,
            measured_last_qpos_hash=hash_bytes(self.last_qpos_bytes),
            measured_last_qvel_hash=hash_bytes(self.last_qvel_bytes),
            physical_gain_compatibility_qualified=False,
            native_trace_provenance_qualified=False,
            body_state_modified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )

    @property
    def contract_hash(self) -> str:
        return hash_json(self.report())


def prepare_external_history(
    qpos_after: np.ndarray,
    qvel_after: np.ndarray,
    applied_targets: np.ndarray,
    *,
    defaults: np.ndarray,
    scale: np.ndarray,
) -> ExternalSonicHistory:
    q = _array(qpos_after, (10, 43))
    v = _array(qvel_after, (10, 41))
    targets = _array(applied_targets, (10, 29))
    defaults, scale = _array(defaults, (29,)), _array(scale, (29,))
    if np.any(abs(np.linalg.norm(q[:, 3:7], axis=1) - 1) > 1e-5):
        raise ValueError("actual unit wxyz history required")
    samples = []
    residuals = []
    for pose, vel, target in zip(q, v, targets, strict=True):
        action, error = equivalent_sonic_action(target, defaults, scale)
        entry = (
            vel[3:6].copy(),
            (pose[7:36] - defaults)[MUJOCO_TO_ISAACLAB],
            vel[6:35][MUJOCO_TO_ISAACLAB],
            action.astype(np.float64),
            _gravity_in_body(pose[3:7]),
        )
        samples.append(tuple(tuple(float(x) for x in field) for field in entry))
        residuals.append(error)
    return ExternalSonicHistory(
        tuple(samples),
        q[-1].tobytes(),
        v[-1].tobytes(),
        targets[-1].tobytes(),
        defaults.tobytes(),
        scale.tobytes(),
        hash_json(
            dict(
                qpos=hash_bytes(q.tobytes()),
                qvel=hash_bytes(v.tobytes()),
                targets=hash_bytes(targets.tobytes()),
                defaults=hash_bytes(defaults.tobytes()),
                scale=hash_bytes(scale.tobytes()),
            )
        ),
        max(residuals),
    )


def attach_external_history(
    controller: G1SonicRunupController,
    history: ExternalSonicHistory,
    reference: SonicPoseReference,
    *,
    current_qpos: np.ndarray,
    current_qvel: np.ndarray,
    padded_reference_frames: int,
) -> dict[str, Any]:
    """Initialize a fresh recipient's measured history, not its body or PD.

    Caller must separately bind these ten *post-step* states to ten preceding
    applied targets. No reset(), planner call, fabricated zero action or repeated
    current-state padding occurs here. Failure before assignment leaves the
    recipient untouched; one successful attachment cannot be repeated.
    """
    q, v = _array(current_qpos, (43,)), _array(current_qvel, (41,))
    if (
        type(controller) is not G1SonicRunupController
        or len(controller._history) != 0
        or controller.reference.shape != (0, 36)
        or controller._history_handoff_binding is not None
        or controller.config.model_variant != "low_latency"
        or controller.config.onnx_graph_encoder_layout
        or controller.default_angles.tobytes() != history.default_bytes
        or controller._action_scale.tobytes() != history.scale_bytes
        or q.tobytes() != history.last_qpos_bytes
        or v.tobytes() != history.last_qvel_bytes
        or len(history.samples) != 10
        or any(tuple(len(x) for x in row) != (3, 29, 29, 29, 3) for row in history.samples)
    ):
        raise ValueError(
            "fresh exact recipient, complete history and current measured boundary required"
        )
    targets = np.frombuffer(history.last_target_bytes, dtype=np.float64).copy()
    action, _ = equivalent_sonic_action(
        targets, controller.default_angles, controller._action_scale
    )
    if not np.array_equal(action.astype(np.float64), history.samples[-1][3]):
        raise ValueError("latest equivalent action differs from actual last target")
    expected_last = controller._history_entry(SimpleNamespace(qpos=q, qvel=v), action)
    if any(
        not np.array_equal(a, b) for a, b in zip(expected_last, history.samples[-1], strict=True)
    ):
        raise ValueError("latest history proprioception differs from actual boundary")
    padded = reference.initialize(q[:36], frames=padded_reference_frames)
    receipt = dict(
        history.report(),
        history_contract_hash=history.contract_hash,
        reference_contract_hash=reference.contract_hash,
        padded_reference_hash=hash_bytes(np.ascontiguousarray(padded).tobytes()),
        measured_boundary_checked_exactly=True,
        recipient_history_reset=False,
        no_simulator_state_or_gain_access=True,
    )
    binding = hash_json(receipt)
    # Owned arrays: neither the historical source nor the actual body is exposed.
    copied = collections.deque(
        (tuple(np.asarray(x, dtype=np.float64).copy() for x in row) for row in history.samples),
        maxlen=10,
    )
    controller.reference = padded
    controller.reference_digest = receipt["padded_reference_hash"]
    controller.action = action.copy()
    controller.target = targets
    controller._history = copied
    controller._history_handoff_binding = binding
    return dict(receipt, binding_hash=binding)
