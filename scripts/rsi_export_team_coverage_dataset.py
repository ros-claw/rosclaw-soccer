"""Export independently re-scored, paired SIM coverage data; never promote."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.team_coverage_dataset import build_coverage_dataset
from rosclaw_soccer.rsi.team_receiving_selector import TeamReceivingSelector
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "audit", "training", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-hash", "audit-hash", "training-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    paths = (args.source, args.audit, args.training)
    hashes = (args.source_hash, args.audit_hash, args.training_hash)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in zip(paths, hashes, strict=True))
    ):
        raise ValueError("exact inputs and unused external output required")
    report, audit, training = (json.loads(p.read_text()) for p in paths)
    if audit["source_hash"] != args.source_hash or report["training_hash"] != args.training_hash:
        raise ValueError("audit and training lineage must bind the coverage report")
    if training["status"] != "TEAM_RECEIVING_OFFLINE_ACTORS_TRAINED_NOT_EVALUATED":
        raise ValueError("original offset model required")
    record = next(r for r in training["records"] if r["seed"] == 17)
    model_path = args.training.parent / record["model_path"]
    if (
        model_path.is_symlink()
        or model_path.parent != args.training.parent
        or digest(model_path) != record["model_hash"]
    ):
        raise ValueError("original confined offset model required")
    offsets = TeamReceivingSelector.from_dict(json.loads(model_path.read_text())).offsets
    data = build_coverage_dataset(report, audit)
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_hash=args.source_hash,
            audit_hash=args.audit_hash,
            training_hash=args.training_hash,
            model_hash=record["model_hash"],
            source_head=args.expected_head,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    path = args.output / "dataset.npz"
    np.savez_compressed(path, **data, action_offsets=offsets)
    if (
        any(digest(p) != h for p, h in zip(paths, hashes, strict=True))
        or digest(model_path) != record["model_hash"]
    ):
        raise ValueError("input changed during export")
    write_once(
        args.output / "complete.json",
        dict(
            status="TEAM_COVERAGE_PAIRED_DATASET_NOT_PROMOTED",
            source_hash=args.source_hash,
            audit_hash=args.audit_hash,
            training_hash=args.training_hash,
            model_hash=record["model_hash"],
            source_head=args.expected_head,
            dataset_hash=digest(path),
            courses=6,
            candidate_actions=16,
            parent_is_separate_action=True,
            snapshot_receiving_recomputed=True,
            global_safety_not_independently_recomputed=True,
            physics_reintegrated=False,
            training_only=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
