"""Two new physical executions check a lossless sampling-file transport.

Compare against a fixed already executed parent/sample, including all body and
ball trace hashes and independently reconstructed motor commands. Not learning.
"""

import argparse
from pathlib import Path

from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.smooth_memory_motor import make_preview
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_train_protected_online_motor_v308 import _head
from scripts.rsi_transport_equivalence import compare_transport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "reference-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    reference = _sealed(args.reference_root / "training_summary.json")
    if (
        reference["schema"] != "soccer.rsi.smooth_memory_failure_exploration.v1"
        or reference["independent_contexts"] != 4
        or reference["exploration_executions"] != 16
        or reference["commitment"]["partition"] != "TRAIN_CONSUMED"
    ):
        raise ValueError("complete fixed four-course AR transport reference required")
    row = reference["rows"][0]
    seed, lane = row["seed"], row["lane"]
    if (seed, lane) != (20261378, 0):
        raise ValueError("transport reference is fixed before the comparison")
    model_path = args.reference_root / "models/sample-0.json"
    model = load_json_artifact(model_path)
    make_preview(model)
    if model["model_hash"] != row["samples"][0]["view_hash"]:
        raise ValueError("fixed sample model identity changed")
    old_parent = _sealed(
        args.reference_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
    )
    old_actor = _sealed(args.reference_root / f"seed{seed}-lane{lane}-sample-0-actor/report.json")
    if (
        old_parent["report_hash"] != row["parent_report_hash"]
        or old_actor["report_hash"] != row["samples"][0]["report_hash"]
    ):
        raise ValueError("reference parent/sample reports changed")
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "logs").mkdir()
    path = args.output_root / "sample-0.json.gz"
    write_once(path, model)
    if load_json_artifact(path) != model:
        raise ValueError("compression changed the complete numeric sampling payload")
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    commitment = dict(
        schema="soccer.rsi.compressed_sampling_transport_commitment.v1",
        source_commit=_head(source),
        runner_hash=hash_bytes(runner.read_bytes()),
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        model_hash=model["model_hash"],
        reference_summary_hash=reference["report_hash"],
        reference_parent_report_hash=old_parent["report_hash"],
        reference_actor_report_hash=old_actor["report_hash"],
        compressed_file_hash=hash_bytes(path.read_bytes()),
        compressed_bytes=path.stat().st_size,
        plain_bytes=model_path.stat().st_size,
        seed=seed,
        lane=lane,
        physical_executions_planned=2,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output_root / "commitment.json", commitment)
    common = dict(
        root=args.output_root,
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        seed=seed,
        lane=lane,
        gpu=0,
        gain=1.2,
        negative_only=True,
        core_root=args.core_root,
        execution_timeout_s=600,
    )
    parent, _ = _run(**common, arm="reproduction", kind="parent")
    parent_path = args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
    actor, _ = _run(
        **common, arm="sample-0", kind="actor", motor_step=path, parent_report_override=parent_path
    )
    for new, old in ((parent, old_parent), (actor, old_actor)):
        if any(
            new[k] != old[k]
            for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
        ):
            raise ValueError("storage-only transport changed measured body or ball physics")
    folder = args.output_root / f"seed{seed}-lane{lane}-sample-0-actor"
    checked = _outcome(folder, actor["contact_motor_policy_hash"], commitment)
    old_checked = _outcome(
        args.reference_root / f"seed{seed}-lane{lane}-sample-0-actor",
        old_actor["contact_motor_policy_hash"],
        {"runner_hash": old_actor["source_hash"], "asset_hash": old_actor["asset_hash"]},
    )
    if any(row["samples"][0][k] != v for k, v in old_checked["outcome"].items()):
        raise ValueError("historical outcomes do not reproduce from their actual traces")
    equality = compare_transport(actor, old_actor, checked["outcome"], old_checked["outcome"])
    if _head(source) != commitment["source_commit"] or load_json_artifact(path) != model:
        raise ValueError("source or sampling payload changed during the comparison")
    report = dict(
        schema="soccer.rsi.compressed_sampling_transport_review.v1",
        source_commitment_hash=hash_json(commitment),
        reference_summary_hash=reference["report_hash"],
        complete_payload_equal=True,
        body_and_ball_trace_hashes_equal=True,
        actual_motor_actions_reconstructed=actor["frames"],
        physical_executions_added=2,
        reference_actor_report_hash=old_actor["report_hash"],
        actor_report_hash=actor["report_hash"],
        measured_outcome=checked["outcome"],
        physical_outcome_comparison=equality,
        model_hash=model["model_hash"],
        compressed_bytes=path.stat().st_size,
        plain_bytes=model_path.stat().st_size,
        qualification="STORAGE_TRANSPORT_EQUIVALENCE_ONLY_NOT_LEARNING_GAIN",
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output_root / "transport_review.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
