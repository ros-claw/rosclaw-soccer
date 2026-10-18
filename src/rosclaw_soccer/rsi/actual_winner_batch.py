"""Numeric own-path zero anchors plus successful corrections, never admission."""

import numpy as np

from rosclaw_soccer.rsi.reference_student_imitation import prepare_numeric_batch
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def prepare_actual_winner_batch(zero_x, zero_y, teacher_x, teacher_y, *, selection):
    """Keep all 16 paths; never put teacher actions on NAV observations.

    The caller must authenticate source receipts and original physical traces.
    A zero target on a successful NAV path is positive stability supervision.
    Failed paths have zero loss weight, not negative reward or fabricated labels.
    """
    arrays = [np.asarray(v) for v in (zero_x, zero_y, teacher_x, teacher_y)]
    zx, zy, tx, ty = arrays
    if (
        any(
            a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all()
            for a, shape in zip(
                arrays, ((8, 300, 135), (8, 300, 12), (8, 300, 135), (8, 300, 12)), strict=True
            )
        )
        or np.any(zy != 0)
        or selection.get("schema") != "soccer.rsi.proposed_actual_winner_supervision.v1"
        or selection.get("report_hash")
        != hash_json({k: v for k, v in selection.items() if k != "report_hash"})
        or len(selection["rows"]) != 16
        or any(
            r["role"] != ("zero" if i % 2 == 0 else "teacher")
            or r["counterfactual_labels"] is not False
            or type(r["experimental_actual_path_supervision"]) is not bool
            for i, r in enumerate(selection["rows"])
        )
    ):
        raise ValueError("whole sealed actual own-path NAV/teacher arrays required")
    x, y = np.empty((16, 300, 135)), np.empty((16, 300, 12))
    x[::2], x[1::2], y[::2], y[1::2] = zx, tx, zy, ty
    rows = selection["rows"]
    batch = prepare_numeric_batch(
        x,
        y,
        partitions=[r["partition"] for r in rows],
        source_seeds=[r["source_seed"] for r in rows],
        eligible=[r["experimental_actual_path_supervision"] for r in rows],
    )
    report = dict(
        schema="soccer.rsi.proposed_actual_winner_numeric_batch.v1",
        selection_hash=selection["report_hash"],
        whole_recorded_paths=16,
        derived_courses=8,
        source_scene_clusters=4,
        train_paths=int(batch["train"].sum()),
        validation_paths=int(batch["validation"].sum()),
        train_positive_weight_frames=int(np.count_nonzero(batch["weights"])),
        whole_observations_hash=hash_bytes(x.tobytes()),
        whole_reference_labels_hash=hash_bytes(y.tobytes()),
        actual_same_path_inputs_and_targets=True,
        failures_retained_not_negative_gradients=True,
        validation_excluded_from_gradient_and_normalization=True,
        successful_union_is_not_policy_success=True,
        source_file_authentication_performed_here=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        optimizer_updates=0,
        new_physical_steps=0,
    )
    return batch, dict(report, report_hash=hash_json(report))
