"""Recompute recorded frozen foundation calls with original CPU ONNX graphs.

This proves recorded network-call consistency, not planner/reference truth,
learning improvement, whole-policy qualification, or real-robot authorization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_runup import qualify_g1_sonic
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.foundation_observation_capture import (
    PREFIX,
    validate_capture,
    validate_measured_history,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def recompute_calls(encoder: Any, decoder: Any, arrays: dict[str, Any]) -> dict[str, Any]:
    """One lane/frame per original graph call; no numerical batch-size changes.

    Only low-latency G1 branch and its qualified 1247->64 / 994->29 shapes
    are supported. Selector/header remain zero as in frozen G1 inference.
    """
    if encoder.get_inputs()[0].shape[-1] != 1247 or decoder.get_inputs()[0].shape[-1] != 994:
        raise ValueError("qualified low-latency foundation ONNX shapes required")
    features = arrays[PREFIX + "encoder_features"]
    token = arrays[PREFIX + "latent_token"]
    observation = arrays[PREFIX + "decoder_input"]
    action = arrays[PREFIX + "raw_action_isaac"]
    encoder_error, decoder_error = 0.0, 0.0
    count = 0
    for frame in range(features.shape[0]):
        for lane in range(features.shape[1]):
            packed = np.zeros((1, 1247), dtype=np.float32)
            packed[:, 4:584] = features[frame, lane, :580]
            packed[:, 590:650] = features[frame, lane, 580:]
            predicted_token = np.asarray(
                encoder.run(None, {encoder.get_inputs()[0].name: packed})[0]
            )
            predicted_action = np.asarray(
                decoder.run(None, {decoder.get_inputs()[0].name: observation[frame, lane][None]})[0]
            )
            if (
                predicted_token.shape != (1, 64)
                or predicted_action.size != 29
                or not np.isfinite(predicted_token).all()
                or not np.isfinite(predicted_action).all()
            ):
                raise ValueError("finite original ONNX G1 output shapes required")
            e = float(np.max(np.abs(predicted_token[0] - token[frame, lane])))
            d = float(np.max(np.abs(predicted_action.reshape(29) - action[frame, lane])))
            if e != 0 or d > 1e-4:
                raise ValueError(
                    f"recorded foundation differs from original ONNX: token={e}, action={d}"
                )
            encoder_error, decoder_error = max(encoder_error, e), max(decoder_error, d)
            count += 1
    return dict(
        recomputed_foundation_calls=count,
        encoder_max_abs_error=encoder_error,
        decoder_max_abs_error=decoder_error,
        encoder_tolerance=0.0,
        decoder_tolerance=1e-4,
    )


def audit_foundation_calls(root: Path, model_root: Path) -> dict[str, Any]:
    import onnxruntime as ort

    report = _sealed(root / "report.json")
    commitment = json.loads((root / "commitment.json").read_text())
    # Native commitments have no embedded self-hash. Read against the report
    # commitment hash rather than treating a missing self-hash as valid.
    if report.get("commitment_hash") != hash_json(commitment):
        raise ValueError("foundation physical report commitment differs")
    trace_path = root / "physical_trace.npz"
    if report.get("physical_trace_hash") != hash_bytes(trace_path.read_bytes()):
        raise ValueError("foundation physical trace hash differs")
    qualification = qualify_g1_sonic(model_root, "low_latency", inference_threads=1)
    qualification.require_eligible()
    if qualification.qualification_hash != report.get("foundation_hash"):
        raise ValueError("original foundation qualification differs from executed foundation")
    paths = [model_root / "low_latency" / f"model_{name}.onnx" for name in ("encoder", "decoder")]
    weights = [hash_bytes(p.read_bytes()) for p in paths]
    if weights != [qualification.encoder_hash, qualification.decoder_hash]:
        raise ValueError("original foundation weights differ")
    with np.load(trace_path, allow_pickle=False) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    validate_capture(commitment.get("foundation_observation_capture"), arrays, frames=300, lanes=1)
    validate_measured_history(arrays)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = [
        ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"]) for p in paths
    ]
    result = recompute_calls(sessions[0], sessions[1], arrays)
    if [hash_bytes(p.read_bytes()) for p in paths] != weights:
        raise ValueError("original foundation weights changed during audit")
    result.update(
        schema="soccer.rsi.foundation_onnx_call_audit.v1",
        physical_report_hash=report["report_hash"],
        physical_trace_hash=report["physical_trace_hash"],
        audit_source_hash=hash_bytes(Path(__file__).read_bytes()),
        foundation_qualification_hash=qualification.qualification_hash,
        encoder_hash=weights[0],
        decoder_hash=weights[1],
        measured_history_reconstructed=True,
        original_encoder_decoder_calls_recomputed=True,
        planner_reference_reconstructed=False,
        independently_trained_policy=False,
        optimizer_updates=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result
