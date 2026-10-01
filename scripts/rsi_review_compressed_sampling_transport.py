"""Independently review the fixed actual compressed/plain physical comparison."""

import argparse
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_transport_equivalence import compare_transport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "reference-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    commitment = load_json_artifact(args.root / "commitment.json")
    reference = _sealed(args.reference_root / "training_summary.json")
    if (
        commitment["schema"] != "soccer.rsi.compressed_sampling_transport_commitment.v1"
        or commitment["reference_summary_hash"] != reference["report_hash"]
        or (commitment["seed"], commitment["lane"]) != (20261378, 0)
        or commitment["physical_executions_planned"] != 2
        or any(commitment[k] is not False for k in ("promotion_authorized", "hardware_authorized"))
    ):
        raise ValueError("fixed bound two-execution transport comparison required")
    stem = "seed20261378-lane0"
    parent_folder = args.root / f"{stem}-reproduction-parent"
    actor_folder = args.root / f"{stem}-sample-0-actor"
    parent = _sealed(parent_folder / "report.json")
    actor = _sealed(actor_folder / "report.json")
    old_parent = _sealed(args.reference_root / f"{stem}-reproduction-parent/report.json")
    old_actor = _sealed(args.reference_root / f"{stem}-sample-0-actor/report.json")
    if (
        old_parent["report_hash"] != commitment["reference_parent_report_hash"]
        or old_actor["report_hash"] != commitment["reference_actor_report_hash"]
    ):
        raise ValueError("historical reference reports changed")
    path = args.root / "sample-0.json.gz"
    payload = load_json_artifact(path)
    if (
        payload != load_json_artifact(args.reference_root / "models/sample-0.json")
        or payload["model_hash"] != commitment["model_hash"]
        or hash_bytes(path.read_bytes()) != commitment["compressed_file_hash"]
    ):
        raise ValueError("compressed payload or identity changed")
    for new, old in ((parent, old_parent), (actor, old_actor)):
        if (
            new["source_hash"] != commitment["runner_hash"]
            or new["asset_hash"] != commitment["asset_hash"]
            or any(
                new[k] != old[k]
                for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
            )
        ):
            raise ValueError("transport changed actual body/ball traces or source bindings")
    checked = _outcome(actor_folder, actor["contact_motor_policy_hash"], commitment)
    old_checked = _outcome(
        args.reference_root / f"{stem}-sample-0-actor",
        old_actor["contact_motor_policy_hash"],
        {"runner_hash": old_actor["source_hash"], "asset_hash": old_actor["asset_hash"]},
    )
    if any(reference["rows"][0]["samples"][0][k] != v for k, v in old_checked["outcome"].items()):
        raise ValueError("historical outcomes do not reproduce from their actual traces")
    equality = compare_transport(actor, old_actor, checked["outcome"], old_checked["outcome"])
    if parent["report_hash"] != actor["parent_report_hash"] or (
        old_parent["report_hash"] != old_actor["parent_report_hash"]
    ):
        raise ValueError("actor is not bound to its actual parent")
    with np.load(actor_folder / "contact_motor_trace.npz", allow_pickle=False) as trace:
        frames = len(trace["applied_joint_delta_rad"])
    if frames != 300 or actor["frames"] != frames:
        raise ValueError("complete physical motor trace required")
    report = dict(
        schema="soccer.rsi.compressed_sampling_transport_independent_review.v1",
        source_commitment_hash=hash_json(commitment),
        source_execution_commit=commitment["source_commit"],
        reference_summary_hash=reference["report_hash"],
        actor_report_hash=actor["report_hash"],
        complete_payload_equal=True,
        body_and_ball_trace_hashes_equal=True,
        actual_motor_actions_reconstructed=frames,
        physical_executions_reviewed=2,
        physical_executions_added=0,
        model_hash=payload["model_hash"],
        compressed_bytes=path.stat().st_size,
        plain_bytes=(args.reference_root / "models/sample-0.json").stat().st_size,
        measured_outcome=checked["outcome"],
        physical_outcome_comparison=equality,
        qualification="STORAGE_TRANSPORT_EQUIVALENCE_ONLY_NOT_LEARNING_GAIN",
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
