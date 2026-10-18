"""Same recorded body, different previous-action inputs; no physical rollouts.

Separates teacher forcing from execution projection. Self-fed here means only
the twelve past-reference features: all measured body states remain recorded
teacher states. Neither path is a student-body success claim or expert query.
"""

import numpy as np
from rosclaw.growth.causal_residual_memory import CausalResidualMemory

from rosclaw_soccer.rsi.contact_reference_labels import reference_labels
from rosclaw_soccer.rsi.pre_parent_reference_features import FEATURE_NAMES, trace_observations
from rosclaw_soccer.rsi.reference_student_execution import decoder_from_model, own_model
from rosclaw_soccer.sim.contracts import hash_bytes

PREVIOUS = tuple(
    i
    for i, name in enumerate(FEATURE_NAMES)
    if name.startswith("previous_applied_reference_increment:")
)


def clipping_readout(raw, mean, scale):
    """Fractions are feature-frame cells, not physical episodes or probabilities."""
    if raw.shape != (300, 135) or not np.isfinite(raw).all():
        raise ValueError("whole finite 300 by 135 raw input required")
    normalized = (raw - mean) / scale
    if not np.isfinite(normalized).all():
        raise ValueError("finite unprojected normalization required")
    groups = dict(
        previous_reference=list(PREVIOUS),
        measured_body_ball_navigation=[i for i in range(134) if i not in PREVIOUS],
        remaining_time=[134],
    )
    return {
        name: dict(
            clipped_cell_fraction=float(np.mean(np.abs(normalized[:, ids]) > 8)),
            maximum_absolute_unclipped_normalized_input=float(np.max(np.abs(normalized[:, ids]))),
        )
        for name, ids in groups.items()
    }


def diagnose_feedback(trace, model, *, joint_limits):
    """Compare four predictions against labels on this exact measured body only."""
    model = own_model(model)
    bounds = np.asarray(joint_limits)
    if (
        bounds.shape != (29, 2)
        or bounds.dtype != np.float64
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("original finite ordered float64 joint limits required")
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    source = trace_observations(trace)
    teacher_x, labels = source["observation"], source["same_state_reference_target_rad"]
    nav = reference_labels(trace)["navigation_joint_target_rad"][:, 0]
    memory = CausalResidualMemory(model["parameters"])
    decoder = decoder_from_model(
        model, world_hash=model["world_hash"], parent_model_hash=model["parent_model_hash"]
    )
    raw_a, applied_a, raw_b, applied_b, own_x = [], [], [], [], []
    other = [i for i in range(135) if i not in PREVIOUS]
    for frame in range(300):
        prediction = (
            memory.step(np.clip((teacher_x[frame] - mean) / scale, -8, 8), index=frame) * 0.35
        )
        # Same execution box as the pinned decoder, not a new controller.
        low = np.maximum(np.minimum(0, bounds[:12, 0] - nav[frame, :12]), -0.35)
        high = np.minimum(np.maximum(0, bounds[:12, 1] - nav[frame, :12]), 0.35)
        projection = np.zeros(12) if frame < 30 else np.clip(prediction, low, high)
        record = decoder.advance(
            trace, frame=frame, navigation_target=nav[frame], joint_limits=bounds
        )
        if teacher_x[frame, other].tobytes() != record["input_features"][other].tobytes():
            raise ValueError("only past-reference inputs may differ; measured body must be fixed")
        raw_a.append(prediction)
        applied_a.append(projection)
        raw_b.append(record["requested_reference_increment_rad"])
        applied_b.append(record["applied_reference_increment_rad"])
        own_x.append(record["input_features"])
    predictions = dict(
        teacher_forced_raw=np.stack(raw_a),
        teacher_forced_projected=np.stack(applied_a),
        self_fed_raw=np.stack(raw_b),
        self_fed_projected=np.stack(applied_b),
    )
    errors = {
        name: dict(
            mean_squared_reference_error_rad2=float(np.mean((a[30:] - labels[30:]) ** 2)),
            maximum_reference_error_rad=float(np.max(np.abs(a[30:] - labels[30:]))),
            predictions_hash=hash_bytes(a.tobytes()),
        )
        for name, a in predictions.items()
    }
    own_x = np.stack(own_x)
    return dict(
        schema="soccer.rsi.reference_feedback_same_body_diagnostic.v1",
        numeric_model_hash=model["model_hash"],
        errors=errors,
        teacher_forced_clipping=clipping_readout(teacher_x, mean, scale),
        self_fed_clipping=clipping_readout(own_x, mean, scale),
        maximum_past_reference_input_difference_rad=float(
            np.max(np.abs(teacher_x[:, PREVIOUS] - own_x[:, PREVIOUS]))
        ),
        maximum_teacher_forced_projection_change_rad=float(
            np.max(
                np.abs(
                    predictions["teacher_forced_raw"][30:]
                    - predictions["teacher_forced_projected"][30:]
                )
            )
        ),
        maximum_self_fed_projection_change_rad=float(
            np.max(
                np.abs(predictions["self_fed_raw"][30:] - predictions["self_fed_projected"][30:])
            )
        ),
        compared_post_warmup_frames=270,
        measured_body_navigation_and_force_inputs_bit_exact_between_paths=True,
        self_fed_path_is_on_recorded_teacher_body=True,
        student_body_rollout=False,
        physical_integration_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
