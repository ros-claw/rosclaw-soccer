"""Fit a local SIM response from paired, bounded closed-loop interventions.

The resulting model predicts one relative contact witness, not ball dynamics,
task success, physical safety or a deployable policy. Multi-axis combinations
have not been sampled just because each individual axis was sampled.
"""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.physics.small_strict_qp import solve_small_strict_qp


@dataclass(frozen=True)
class MeasuredContactResponse:
    source_hash: str
    origin_relative_xyz_m: tuple[float, float, float]
    jacobian: tuple[tuple[float, ...], ...]
    axes: tuple[int, ...]
    sampled_amplitude_rad: float


def fit_measured_contact_response(
    report: dict[str, Any], *, source_hash: str
) -> MeasuredContactResponse:
    """Fit centered finite differences only from complete, safe paired arms.

    Authentication of the report hash and raw physics evidence is the caller's
    responsibility; this function refuses missing, unsafe and malformed pairs.
    """
    if (
        not isinstance(report, dict)
        or not isinstance(source_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash) is None
        or report.get("status") != "CFTRACK_HIP_KNEE_PHYSICS_RESPONSE_CALIBRATION_NOT_PROMOTED"
        or report.get("task_promoted") is not False
        or report.get("hardware_authorized") is not False
        or report.get("witness_cftrack_control") != 95
    ):
        raise ValueError("exact simulation calibration report and source identity required")
    baseline = np.asarray(report.get("witness_parent_relative_xyz_m"), dtype=np.float64)
    records = report.get("records")
    if baseline.shape != (3,) or not np.isfinite(baseline).all() or not isinstance(records, dict):
        raise ValueError("finite witness and paired records required")
    axes = (6, 7, 8, 9)
    expected_names = {"parent", *(f"hg{i}-{sign}" for i in axes for sign in ("minus", "plus"))}
    if set(records) != expected_names:
        raise ValueError("complete prespecified four-axis pairing required")
    for record in records.values():
        if (
            record.get("complete") is not True
            or record.get("body_failed") is not False
            or record.get("runtime_failure") is not None
            or record.get("outcome", {}).get("safe") is not True
        ):
            raise ValueError("incomplete or unsafe physical future cannot train response")
    columns = []
    for axis in axes:
        vectors = []
        for name, sign in (("minus", -1), ("plus", 1)):
            row = records[f"hg{axis}-{name}"]
            vec = np.asarray(row.get("witness_relative_xyz_m"), dtype=np.float64)
            if (
                row.get("delta_hg_index") != axis
                or row.get("delta_rad") != sign * 0.08
                or vec.shape != (3,)
                or not np.isfinite(vec).all()
            ):
                raise ValueError("exact finite paired interventions required")
            vectors.append(vec)
        columns.append((vectors[1] - vectors[0]) / 0.16)
    matrix = np.stack(columns, axis=1)
    return MeasuredContactResponse(
        source_hash,
        tuple(baseline),
        tuple(tuple(float(x) for x in row) for row in matrix),
        axes,
        0.08,
    )


def propose_bounded_response(model: MeasuredContactResponse) -> tuple[np.ndarray, dict[str, Any]]:
    """One deterministic, regularized in-range proposal; validation is separate."""
    if (
        type(model) is not MeasuredContactResponse
        or model.axes != (6, 7, 8, 9)
        or model.sampled_amplitude_rad != 0.08
    ):
        raise ValueError("qualified fixed-axis measured model required")
    matrix = np.asarray(model.jacobian, dtype=np.float64)
    baseline = np.asarray(model.origin_relative_xyz_m, dtype=np.float64)
    if (
        matrix.shape != (3, 4)
        or baseline.shape != (3,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("finite 3x4 response model required")
    h = matrix.T @ matrix + 1e-5 * np.eye(4)
    gradient = matrix.T @ baseline
    inequalities = np.vstack((np.eye(4), -np.eye(4)))
    delta, certificate = solve_small_strict_qp(h, gradient, inequalities, np.full(8, 0.08))
    return delta, dict(
        predicted_gap_m=float(np.linalg.norm(baseline + matrix @ delta)),
        baseline_gap_m=float(np.linalg.norm(baseline)),
        singular_values=np.linalg.svd(matrix, compute_uv=False).tolist(),
        optimization_certificate=certificate,
        simultaneous_axes_not_calibrated=True,
        predicted_geometry_not_physical_contact=True,
        source_hash=model.source_hash,
        hardware_authorized=False,
        promotion_authorized=False,
    )
