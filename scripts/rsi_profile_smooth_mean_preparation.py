"""Read-only representative preparation profile; no physics or optimizer steps.

Samples are uniformly spaced from the complete already consumed manifest/NPZ.
This diagnoses implementation cost, never policy quality or fresh performance.
"""

import argparse
import cProfile
import pstats
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.smooth_memory_motor import CompiledSmoothMemoryMotor, make_preview
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("behavior-model", "learning-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--samples", type=int, choices=range(4, 257), default=64)
    args = parser.parse_args()
    manifest = _sealed(args.learning_root / "rollout_manifest.json")
    model = load_json_artifact(args.behavior_model)
    dataset = args.learning_root / "rollouts.npz"
    if manifest["parent_model_hash"] != model["model_hash"] or manifest["data_hash"] != hash_bytes(
        dataset.read_bytes()
    ):
        raise ValueError("exact sealed consumed behavior data required")
    with np.load(dataset, allow_pickle=False) as data:
        x, phase = data["observation"], data["phase_index"]
    if len(x) != manifest["frame_sample_count"] or len(phase) != len(x):
        raise ValueError("complete physical observation batch required")
    ids = np.linspace(0, len(x) - 1, args.samples, dtype=int)
    args.output_root.mkdir(parents=True, exist_ok=False)
    profile = cProfile.Profile()
    started = time.perf_counter()
    decoder = profile.runcall(lambda: CompiledSmoothMemoryMotor(make_preview(model)))
    construction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    means = profile.runcall(lambda: np.stack([decoder.raw_mean(x[i], int(phase[i])) for i in ids]))
    inference_seconds = time.perf_counter() - started
    profile.dump_stats(str(args.output_root / "preparation.pstats"))
    pstats.Stats(profile).sort_stats("cumulative").print_stats(20)
    report = dict(
        schema="soccer.rsi.smooth_mean_preparation_profile.v1",
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        model_hash=model["model_hash"],
        manifest_hash=manifest["report_hash"],
        dataset_hash=manifest["data_hash"],
        sample_indices=ids.tolist(),
        sampling="UNIFORM_EXISTING_CONSUMED_ROWS_NOT_NEW_EVALUATION",
        profile_hash=hash_bytes((args.output_root / "preparation.pstats").read_bytes()),
        measured_mean_hash=hash_bytes(means.tobytes()),
        construction_seconds=construction_seconds,
        inference_seconds=inference_seconds,
        new_physical_executions=0,
        completed_optimizer_steps=0,
        qualification="IMPLEMENTATION_COST_ONLY_NOT_POLICY_GAIN",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output_root / "profile_review.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
