"""Experimentally make frozen SONIC ONNX components batch-dynamic.

This emits only a SIM_ONLY derivative of a qualified ONNX file. It never
replaces the original model and refuses export without per-sample parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def batch_dynamic_encoder(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace constant leading batch=1 in Reshape shapes with inferred batch."""

    graph = model.graph
    producers = {name: node for node in graph.node for name in node.output}
    converted = 0
    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_node = producers.get(node.input[1])
        if shape_node is None or shape_node.op_type != "Constant":
            continue
        attrs = [attr for attr in shape_node.attribute if attr.name == "value"]
        if len(attrs) != 1:
            continue
        shape = numpy_helper.to_array(attrs[0].t).copy()
        if shape.ndim != 1 or not len(shape) or shape[0] != 1:
            continue
        shape[0] = -1
        attrs[0].t.CopyFrom(numpy_helper.from_array(shape))
        converted += 1
    if converted < 10:
        raise ValueError("unexpected SONIC encoder shape graph; no batch export")
    for value in (*graph.input, *graph.output):
        value.type.tensor_type.shape.dim[0].ClearField("dim_value")
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    onnx.checker.check_model(model)
    return model, converted


def batch_dynamic_decoder(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Decoder has no fixed-batch Reshape; expose its existing batch dimension."""

    graph = model.graph
    if (
        len(graph.input) != 1
        or len(graph.output) != 1
        or graph.input[0].type.tensor_type.shape.dim[1].dim_value != 994
        or graph.output[0].type.tensor_type.shape.dim[1].dim_value != 29
    ):
        raise ValueError("unexpected SONIC decoder graph")
    for value in (*graph.input, *graph.output):
        value.type.tensor_type.shape.dim[0].ClearField("dim_value")
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    onnx.checker.check_model(model)
    return model, 0


def validate_batch_parity(
    original_path: Path, candidate: onnx.ModelProto, *, count: int, width: int
) -> float:
    if not 2 <= count <= 64:
        raise ValueError("batch validation count must be 2..64")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reference = ort.InferenceSession(
        str(original_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    proposed = ort.InferenceSession(
        candidate.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(20260928)
    inputs = rng.normal(size=(count, width)).astype(np.float32)
    actual = proposed.run(None, {proposed.get_inputs()[0].name: inputs})[0]
    expected = np.concatenate(
        [reference.run(None, {reference.get_inputs()[0].name: row[None, :]})[0] for row in inputs],
        axis=0,
    )
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("batch SONIC component emitted invalid output shape or values")
    difference = float(np.max(np.abs(actual - expected)))
    if difference > 1e-4:
        raise ValueError(f"batch SONIC component parity failed: {difference}")
    return difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch-count", type=int, default=8)
    parser.add_argument("--component", choices=("encoder", "decoder"), default="encoder")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("new output and report paths required")
    original_hash = hash_bytes(args.original.read_bytes())
    transform = batch_dynamic_encoder if args.component == "encoder" else batch_dynamic_decoder
    candidate, changed = transform(onnx.load(str(args.original)))
    max_difference = validate_batch_parity(
        args.original,
        candidate,
        count=args.batch_count,
        width=1247 if args.component == "encoder" else 994,
    )
    onnx.save_model(candidate, str(args.output))
    report = {
        "schema": "rsi_sonic_batch_component_export_v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "original_model_hash": original_hash,
        "candidate_model_hash": hash_bytes(args.output.read_bytes()),
        "component": args.component,
        "batch_count": args.batch_count,
        "dynamic_reshape_count": changed,
        "maximum_reference_difference": max_difference,
    }
    report["report_hash"] = hash_json(report)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
