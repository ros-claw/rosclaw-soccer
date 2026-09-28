"""Benchmark qualified frozen SONIC batched Torch vs original per-player ONNX.

Pure inference only: no simulator, policy update, motor target or activation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
from rosclaw_soccer.sim.contracts import hash_json


def benchmark(model_root: Path, *, count: int, frames: int, device: str) -> dict:
    if not 2 <= count <= 64 or not 20 <= frames <= 300:
        raise ValueError("bounded independent inference batch required")
    model = FrozenSonicG1Torch(model_root, variant="low_latency", device=device)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    paths = [model_root / "low_latency" / f"model_{part}.onnx" for part in ("encoder", "decoder")]
    encoder, decoder = [
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        for path in paths
    ]
    rng = np.random.default_rng(20260928)
    features = rng.normal(0, 0.2, size=(count, 640)).astype(np.float32)
    histories = rng.normal(0, 0.2, size=(count, 930)).astype(np.float32)
    encoder_input = np.zeros((count, 1247), dtype=np.float32)
    encoder_input[:, 4:584] = features[:, :580]
    encoder_input[:, 590:650] = features[:, 580:]
    reference_tokens = np.concatenate(
        [encoder.run(None, {"obs_dict": row[None]})[0] for row in encoder_input], axis=0
    )
    decoder_input = np.concatenate((reference_tokens, histories), axis=1)
    reference_actions = np.concatenate(
        [decoder.run(None, {"obs_dict": row[None]})[0] for row in decoder_input], axis=0
    )
    feature_tensor = torch.from_numpy(features).to(model.device)
    history_tensor = torch.from_numpy(histories).to(model.device)
    with torch.no_grad():
        token = model.encode_g1(feature_tensor)
        action = model.decode(torch.cat((token, history_tensor), dim=1))
        token_error = float(np.max(np.abs(token.cpu().numpy() - reference_tokens)))
        action_error = float(np.max(np.abs(action.cpu().numpy() - reference_actions)))
        if token_error > 1e-4 or action_error > 1e-4:
            raise ValueError("batched SONIC Torch path differs from frozen ONNX")
        for _ in range(10):
            model.decode(torch.cat((model.encode_g1(feature_tensor), history_tensor), dim=1))
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        for _ in range(frames):
            model.decode(torch.cat((model.encode_g1(feature_tensor), history_tensor), dim=1))
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        torch_seconds = time.perf_counter() - start
    for _ in range(2):
        for row in encoder_input:
            encoder.run(None, {"obs_dict": row[None]})
        for row in decoder_input:
            decoder.run(None, {"obs_dict": row[None]})
    start = time.perf_counter()
    for _ in range(frames):
        for row in encoder_input:
            encoder.run(None, {"obs_dict": row[None]})
        for row in decoder_input:
            decoder.run(None, {"obs_dict": row[None]})
    onnx_seconds = time.perf_counter() - start
    report = {
        "schema": "rsi_sonic_torch_batch_inference_benchmark_v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "physical_episode_count": 0,
        "variant": "low_latency",
        "foundation_qualification_hash": model.qualification.qualification_hash,
        "batch_count": count,
        "frames": frames,
        "device": str(model.device),
        "encoder_max_abs_difference": token_error,
        "decoder_max_abs_difference": action_error,
        "torch_batch_seconds": torch_seconds,
        "onnx_separate_seconds": onnx_seconds,
        "speedup": onnx_seconds / torch_seconds,
    }
    report["report_hash"] = hash_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--batch-count", type=int, default=8)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = benchmark(
        args.model_root, count=args.batch_count, frames=args.frames, device=args.device
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
