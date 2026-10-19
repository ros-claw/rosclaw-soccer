"""Admission and raw comparison for earlier80ms ZERO-action calibration only."""

import numpy as np

from scripts.rsi_feedback_branch_inputs import prepare as prepare_original
from scripts.rsi_feedback_branch_inputs import require_scope as old_scope
from scripts.rsi_impact_torque_authority_screen import equal
from scripts.rsi_stage_release_contact_replay import digest


def require_scope(cfg):
    fixed = dict(
        profile="six-zero-early-feedback80ms-v1",
        frame_selection="RETROSPECTIVE_CONTACT_MINUS_TWO_DIAGNOSTIC_ONLY",
        horizon_substeps=40,
        feedback_forecast_substep_budget=480,
        combined_substep_budget=18480,
        branch_encoder_decoder_pairs=36,
        contact_frame_lookback=2,
        offset_duration_substeps=30,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("exact earlier80ms zero-action calibration required")
    normalized = dict(cfg)
    normalized.update(
        profile="six-zero-feedback40ms-v1",
        frame_selection="RETROSPECTIVE_SOURCE_CONTACT_FRAME_DIAGNOSTIC_ONLY",
        horizon_substeps=20,
        feedback_forecast_substep_budget=240,
        combined_substep_budget=18240,
        branch_encoder_decoder_pairs=12,
    )
    old_scope(normalized)  # Retain every other original six-case/no-learning bound.


def prepare(path):
    p = prepare_original(path, scope_validator=require_scope)
    for case in p.cases:
        source_frame = case["frame"]
        frame = source_frame - 2
        if not 30 <= frame <= 294 or frame // 20 != (frame + 3) // 20:
            raise ValueError("entire registered early window must avoid planner refresh")
        case["original_contact_frame"] = source_frame
        case["frame"] = frame
    return p


def preserve_and_compare(root, index, branch, out, predicted, trace, frame, widths):
    """Comparison oracle stays outside the actor and runs after saving raw arrays."""
    path = root / f"case-{index}-branch-{branch}.npz"
    if len(out.next_target_records) != 3 or len(out.next_neural_records) != 3:
        raise ValueError("three actual updated targets/neural captures required")
    flattened = {}
    for j, (record, neural) in enumerate(
        zip(out.next_target_records, out.next_neural_records, strict=True), start=1
    ):
        flattened.update({f"next_{j}_" + k: v for k, v in record.items()})
        flattened.update({f"neural_next_{j}_" + k: v for k, v in neural.items()})
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            end_integration_state=out.end_integration_state,
            **out.arrays,
            **flattened,
        )
    if predicted is not None and (
        out.donor_integration_state_hash != predicted.donor_integration_state_hash
        or out.end_integration_state.tobytes() != predicted.end_integration_state.tobytes()
        or set(out.arrays) != set(predicted.arrays)
        or any(out.arrays[k].tobytes() != predicted.arrays[k].tobytes() for k in out.arrays)
        or any(
            set(a) != set(b) or any(a[k].tobytes() != b[k].tobytes() for k in a)
            for before, after in (
                (out.next_target_records, predicted.next_target_records),
                (out.next_neural_records, predicted.next_neural_records),
            )
            for a, b in zip(before, after, strict=True)
        )
    ):
        raise ValueError("independent80ms zero branches differ")
    if np.any(out.arrays["requested_offset_nm"]):
        raise ValueError("zero calibration must never request nonzero torque")
    for j, (record, neural) in enumerate(
        zip(out.next_target_records, out.next_neural_records, strict=True), start=1
    ):
        for name, source in (
            ("navigation_target_rad", "foundation_navigation_joint_target_rad"),
            ("taskspace_increment_rad", "taskspace_joint_increment_rad"),
            ("motor_delta_rad", "motor_delta_rad"),
            ("joint_target_rad", "joint_target_rad"),
            ("canonical_qpos", "canonical_qpos"),
            ("canonical_qvel", "canonical_qvel"),
        ):
            equal(record[name], trace[source][frame + j, 0], f"early next{j} " + name)
        if set(neural) != set(widths):
            raise ValueError("all three full actual NN captures required")
        for name, width in widths.items():
            a = neural[name]
            if (
                a.shape != (1, width)
                or a.dtype != np.float32
                or not np.isfinite(a).all()
                or a.tobytes() != trace["foundation_neural_" + name][frame + j].tobytes()
            ):
                raise ValueError(f"early next{j} neural bits differ: " + name)
    return dict(path=str(path), sha256=digest(path), donor_hash=out.donor_integration_state_hash)
