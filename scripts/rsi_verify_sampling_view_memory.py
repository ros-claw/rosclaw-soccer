"""Prove bounded resident sampling metadata against a complete historical artifact."""

import argparse
from pathlib import Path

from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.smooth_memory_motor import make_sampling_view
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_sampling_view_memory import bounded_views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "reference-view", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    model = load_json_artifact(args.model)
    reference = load_json_artifact(args.reference_view)
    actual = bounded_views(make_sampling_view, model, [reference["seed"]])[0]
    if actual != reference or actual["mean_model"] is not model:
        raise ValueError("complete historical sampling payload differs")
    report = dict(
        schema="soccer.rsi.sampling_resident_memory_equivalence.v1",
        complete_payload_equal=True,
        sampling_policy_hash=actual["model_hash"],
        complete_payload_hash=hash_json(actual),
        mean_model_hash=model["model_hash"],
        reference_file_hash=hash_bytes(args.reference_view.read_bytes()),
        helper_hash=hash_bytes(
            Path(__file__).with_name("rsi_sampling_view_memory.py").read_bytes()
        ),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        physical_executions_added=0,
        qualification="RESIDENT_STORAGE_EQUIVALENCE_NOT_LEARNING_OR_PHYSICS_GAIN",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
