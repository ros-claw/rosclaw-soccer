"""Assess all-seed retention; only admit new development, never promotion."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.team_coverage_admission import assess_coverage_retention
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "source-hash",
        "audit-hash",
        "expected-head",
        "simulation-head",
        "audit-head",
        "models-hash",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or args.source.is_symlink()
        or args.audit.is_symlink()
        or digest(args.source) != args.source_hash
        or digest(args.audit) != args.audit_hash
    ):
        raise ValueError("exact source and unused external assessment required")
    report = json.loads(args.source.read_text())
    audit = json.loads(args.audit.read_text())
    if (
        report.get("source_head") != args.simulation_head
        or audit.get("source_head") != args.audit_head
        or audit.get("models_hash") != args.models_hash
        or report.get("learned_coverage_training_hash") != args.models_hash
    ):
        raise ValueError("frozen simulation, independent audit and actor lineage required")
    result = assess_coverage_retention(report, audit, args.source_hash)
    if digest(args.source) != args.source_hash or digest(args.audit) != args.audit_hash:
        raise ValueError("source changed during assessment")
    args.output.mkdir()
    write_once(
        args.output / "complete.json",
        dict(
            status="TEAM_COVERAGE_RETENTION_ASSESSED_NOT_PROMOTED",
            source_hash=args.source_hash,
            audit_hash=args.audit_hash,
            source_head=args.expected_head,
            minimum_gains_per_seed=2,
            result=result,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
