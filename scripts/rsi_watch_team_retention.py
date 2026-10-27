"""Bounded local watcher: after successful retention, run read-only trace audit.

This deterministic follow-up cannot wake an agent, choose future experiments,
promote policies, use hardware, or resume a failed/incomplete simulator job.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "models", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("unit", "source-head", "models-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--retention-proof", type=Path)
    parser.add_argument("--retention-hash")
    parser.add_argument("--neighbor", action="store_true")
    parser.add_argument("--admission-proof", type=Path)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        args.neighbor
        and (args.retention_proof or args.retention_hash)
        or not args.neighbor
        and (not args.retention_proof or not args.retention_hash)
    ):
        raise ValueError("retention proof required only for full retention audit")
    if args.admission_proof and not args.neighbor:
        raise ValueError("admission proof only applies to neighbor audit")
    if (
        not 1 <= args.timeout_sec <= 43200
        or not args.unit.startswith("rosclaw-team-")
        or not args.unit.endswith(".service")
        or "/" in args.unit
    ):
        raise ValueError("bounded SIM-only unit and timeout required")
    if args.output.exists() or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external audit output required")
    deadline = time.monotonic() + args.timeout_sec
    source = args.source / "complete.json"
    while True:
        query = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                args.unit,
                "-p",
                "ActiveState",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        state = dict(line.split("=", 1) for line in query.stdout.splitlines() if "=" in line)
        if source.is_file() and not source.is_symlink() and state.get("ActiveState") != "active":
            if (
                state.get("Result", "success") != "success"
                or state.get("ExecMainStatus", "0") != "0"
            ):
                raise ValueError("simulation unit failed; do not audit as success")
            report_hash = digest(source)
            report = json.loads(source.read_text())
            if (
                report.get("status")
                != (
                    "TEAM_RECEIVING_NEIGHBOR_DEVELOPMENT_NOT_PROMOTED"
                    if args.neighbor
                    else "TEAM_RECEIVING_COVERAGE_LEARNED_FULL_RETENTION_NOT_PROMOTED"
                )
                or report.get("source_head") != args.source_head
                or report.get("physical_runs") != (48 if args.neighbor else 220)
                or report.get("promotion_authorized") is not False
            ):
                raise ValueError("exact complete unpromoted retention receipt required")
            break
        if state.get("ActiveState") in ("failed", "inactive") and not source.exists():
            if (
                args.neighbor
                and state.get("Result") == "success"
                and args.admission_proof
                and args.admission_proof.is_file()
                and not args.admission_proof.is_symlink()
            ):
                admission = json.loads(args.admission_proof.read_text())
                if (
                    admission.get("status") == "TEAM_COVERAGE_RETENTION_ASSESSED_NOT_PROMOTED"
                    and admission.get("result", {}).get("development_generalization_admitted")
                    is False
                ):
                    print(
                        "Retention gate rejected; neighbor audit intentionally skipped.", flush=True
                    )
                    return
            raise ValueError("simulation exited without completion; no retry or promotion")
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded watcher expired without complete simulation")
        time.sleep(15)
    command = [
        sys.executable,
        "scripts/rsi_audit_team_coverage_neighbor.py"
        if args.neighbor
        else "scripts/rsi_audit_team_coverage_retention.py",
        "--source",
        str(args.source),
        "--source-hash",
        report_hash,
        "--models",
        str(args.models),
        "--models-hash",
        args.models_hash,
        "--output",
        str(args.output),
        "--expected-head",
        args.expected_head,
    ]
    if not args.neighbor:
        command.extend(
            [
                "--retention-proof",
                str(args.retention_proof),
                "--retention-hash",
                args.retention_hash,
            ]
        )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
