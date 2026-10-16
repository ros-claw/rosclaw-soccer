"""Check reconstructed planner references against recorded encoder features.

This is the deployed low-latency legacy-layout contract only. Reconstructs
features in NumPy without calling the producer tracker or a neural encoder.
Callers authenticate assets, physical traces and source receipts separately.
It proves reference consumption, not physical improvement or teacher quality.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.sonic_locomotion_reference_audit import replay_locomotion_segments
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

# Exact G1 low-latency ISAAC ordering, independent of the producer imports.
ORDER = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)
ATOL = 1e-5


def low_latency_legacy_features(reference: np.ndarray, root_quaternion: np.ndarray) -> np.ndarray:
    """Ten stride-one poses, reference gradient, body-relative orientation."""
    if (
        reference.shape != (10, 36)
        or root_quaternion.shape != (4,)
        or not np.isfinite(reference).all()
        or not np.isfinite(root_quaternion).all()
        or np.max(np.abs(reference)) > 1e6
        or not np.allclose(np.linalg.norm(reference[:, 3:7], axis=1), 1, atol=1e-4, rtol=0)
        or abs(float(np.linalg.norm(root_quaternion)) - 1) > 1e-4
    ):
        raise ValueError("ten finite normalized low-latency reference poses required")
    future = reference.astype(np.float32)
    root = root_quaternion.astype(np.float32).copy()
    root[1:] *= -1
    positions = future[:, 7:36][:, ORDER]
    velocities = np.gradient(positions, 0.02, axis=0)
    a, b, c, d = root
    e, f, g, h = future[:, 3:7].T
    w = a * e - b * f - c * g - d * h
    x = a * f + b * e + c * h - d * g
    y = a * g - b * h + c * e + d * f
    z = a * h + b * g - c * f + d * e
    rotation = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
        ),
        axis=1,
    )
    # Preserve the deployed legacy orientation offset; do not "repair" layout
    # in an auditor and thereby verify a different network input.
    result = np.concatenate(
        (positions.ravel(), velocities.ravel(), rotation.ravel()[6:], np.zeros(6, dtype=np.float32))
    )
    return result


def audit_encoder_reference_consumption(
    session: Any, trace: dict[str, Any], events: list[dict[str, Any]], mode: int
) -> dict[str, Any]:
    """Fifteen independent planner calls and 300 measured feature comparisons."""
    recorded = np.asarray(trace.get("foundation_neural_encoder_features"))
    if (
        recorded.shape != (300, 1, 640)
        or recorded.dtype != np.float64
        or not np.isfinite(recorded).all()
        or np.max(np.abs(recorded)) > 1e6
    ):
        raise ValueError("complete finite float64 recorded encoder features required")
    rebuilt = np.empty((300, 640), dtype=np.float32)
    seen: list[int] = []

    def observe(frame: int, reference: np.ndarray) -> None:
        for tick in range(frame, frame + 20):
            rebuilt[tick] = low_latency_legacy_features(
                reference[tick : tick + 10], trace["canonical_qpos"][tick, 0, 3:7]
            )
            seen.append(tick)

    rows = replay_locomotion_segments(session, trace, events, mode, reference_observer=observe)
    if seen != list(range(300)):
        raise ValueError("complete ordered encoder consumption coverage required")
    error = np.abs(rebuilt.astype(np.float64) - recorded[:, 0])
    if np.any(error > ATOL):
        tick, feature = (int(i) for i in np.unravel_index(np.argmax(error), error.shape))
        raise ValueError(f"planner-to-encoder mismatch at frame {tick}, feature {feature}")
    result = dict(
        schema="soccer.rsi.sonic_encoder_reference_consumption.v1",
        model_variant="low_latency",
        legacy_encoder_layout=True,
        reference_stride=1,
        native_reference_velocity=False,
        heading_normalized=False,
        actual_cpu_onnx_planner_calls=len(rows),
        compared_encoder_frames=len(seen),
        compared_features_per_frame=640,
        comparison_atol=ATOL,
        comparison_rtol=0,
        maximum_absolute_error=float(np.max(error)),
        reconstructed_features_hash=hash_bytes(rebuilt.tobytes()),
        recorded_features_hash=hash_bytes(recorded.tobytes()),
        planner_events=rows,
        producer_tracker_called=False,
        neural_encoder_called_here=False,
        source_authentication_required_from_caller=True,
        new_physics_executions=0,
        neural_optimizer_updates=0,
        physical_improvement_proven=False,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
