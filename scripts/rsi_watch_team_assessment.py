"""Bounded follow-up that assesses a finished retention audit, never promotes."""

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
    for name in ("source", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("unit", "simulation-head", "audit-head", "models-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        not 1 <= args.timeout_sec <= 43200
        or not args.unit.startswith("rosclaw-team-")
        or not args.unit.endswith(".service")
        or "/" in args.unit
    ):
        raise ValueError("bounded SIM-only unit and timeout required")
    if args.output.exists() or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external assessment output required")
    deadline = time.monotonic() + args.timeout_sec
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
        if (
            args.audit.is_file()
            and not args.audit.is_symlink()
            and state.get("ActiveState") != "active"
        ):
            if (
                state.get("Result", "success") != "success"
                or state.get("ExecMainStatus", "0") != "0"
            ):
                raise ValueError("audit unit failed; assessment blocked")
            if args.source.is_symlink() or not args.source.is_file():
                raise ValueError("complete source receipt required")
            source_hash, audit_hash = digest(args.source), digest(args.audit)
            source, audit = (json.loads(p.read_text()) for p in (args.source, args.audit))
            if (
                source.get("source_head") != args.simulation_head
                or audit.get("source_head") != args.audit_head
                or audit.get("source_hash") != source_hash
                or audit.get("models_hash") != args.models_hash
            ):
                raise ValueError("exact bound simulation, audit and model lineage required")
            break
        if state.get("ActiveState") in ("failed", "inactive") and not args.audit.exists():
            raise ValueError("audit exited without completion; no retry or promotion")
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded watcher expired without complete audit")
        time.sleep(15)
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_assess_team_coverage_retention.py",
            "--source",
            str(args.source),
            "--source-hash",
            source_hash,
            "--audit",
            str(args.audit),
            "--audit-hash",
            audit_hash,
            "--output",
            str(args.output),
            "--simulation-head",
            args.simulation_head,
            "--audit-head",
            args.audit_head,
            "--models-hash",
            args.models_hash,
            "--expected-head",
            args.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
