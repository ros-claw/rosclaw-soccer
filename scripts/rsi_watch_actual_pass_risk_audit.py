"""Bounded replay audit after a contact-risk training shard finishes."""

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
    for name in ("source", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("unit", "dataset-hash", "training-head", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        not 1 <= args.timeout_sec <= 43200
        or args.shard_index not in (0, 1, 2, 3)
        or not args.unit.startswith("rosclaw-team-")
        or not args.unit.endswith(".service")
        or "/" in args.unit
    ):
        raise ValueError("bounded SIM-only unit and shard required")
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
                raise ValueError("training unit failed; audit blocked")
            source_hash = digest(source)
            report = json.loads(source.read_text())
            if (
                report.get("status")
                != "ACTUAL_PASS_CONTACT_RISK_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL"
                or report.get("source_head") != args.training_head
                or report.get("dataset_hash") != args.dataset_hash
                or report.get("shard_index") != args.shard_index
                or report.get("shard_count") != 4
                or report.get("promotion_authorized") is not False
            ):
                raise ValueError("exact completed unpromoted context shard required")
            break
        if state.get("ActiveState") in ("failed", "inactive") and not source.exists():
            raise ValueError("training exited without completion; no retry")
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded watcher expired without complete training shard")
        time.sleep(15)
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_audit_actual_pass_contact_risk.py",
            "--source",
            str(args.source),
            "--source-hash",
            source_hash,
            "--dataset",
            str(args.dataset),
            "--dataset-hash",
            args.dataset_hash,
            "--output",
            str(args.output),
            "--expected-head",
            args.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
