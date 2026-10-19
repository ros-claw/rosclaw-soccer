"""Rebuild default planner-to-encoder inputs without producer planner events.

This explicit contract compares new deterministic planner output against actual
recorded encoder features; it does NOT authenticate nonexistent event logs.
The legacy recorded-event audit remains strict. Caller authenticates original
planner weights, source, measured trace and physical replay separately.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.sonic_encoder_reference_audit import ATOL, low_latency_legacy_features
from rosclaw_soccer.rsi.sonic_locomotion_reference_audit import (
    _observations,
    _reconstruct_references,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def audit_unlogged_encoder_reference(session: Any, trace: dict[str, Any]) -> dict[str, Any]:
    """Fixed native default mode2/seed30300,15 calls,300 encoder inputs."""
    qpos, commands = _observations(trace, 2)
    recorded = trace.get("foundation_neural_encoder_features")
    if (
        not isinstance(recorded, np.ndarray)
        or recorded.shape != (300, 1, 640)
        or recorded.dtype != np.float32
        or not np.isfinite(recorded).all()
        or np.max(abs(recorded)) > 1e6
    ):
        raise ValueError("complete finite actual encoder feature captures required")
    rebuilt = np.empty((300, 640), dtype=np.float32)
    segments = []
    seen = []
    for frame, mode, segment_hash, reference in _reconstruct_references(session, qpos, commands, 2):
        segments.append(
            dict(frame=frame, planner_mode=mode, reconstructed_segment_hash=segment_hash)
        )
        for tick in range(frame, frame + 20):
            rebuilt[tick] = low_latency_legacy_features(
                reference[tick : tick + 10], qpos[tick, 3:7]
            )
            seen.append(tick)
    if seen != list(range(300)) or len(segments) != 15:
        raise ValueError("all ordered default planner/encoder frames required")
    error = abs(rebuilt.astype(np.float64) - recorded[:, 0])
    if np.any(error > ATOL):
        tick, feature = (int(i) for i in np.unravel_index(np.argmax(error), error.shape))
        raise ValueError(f"unlogged planner-to-encoder mismatch at frame {tick}, feature {feature}")
    result = dict(
        schema="soccer.rsi.unlogged_default_planner_encoder_audit.v1",
        default_high_speed_mode=2,
        planner_seed=30300,
        actual_planner_calls=15,
        encoder_frames_compared=300,
        features_per_frame=640,
        reconstructed_segments=segments,
        original_planner_event_logs_authenticated=False,
        maximum_absolute_feature_error=float(error.max()),
        comparison_atol=ATOL,
        comparison_rtol=0,
        reconstructed_features_hash=hash_bytes(rebuilt.tobytes()),
        recorded_features_hash=hash_bytes(recorded.tobytes()),
        producer_tracker_called=False,
        source_and_model_authentication_required_from_caller=True,
        physical_replay_required_from_caller=True,
        neural_arithmetic_recomputed=False,
        full_causal_policy_qualified=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
    )
    result["report_hash"] = hash_json(result)
    return result
