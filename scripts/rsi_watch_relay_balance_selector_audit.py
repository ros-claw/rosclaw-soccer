"""Wait for grouped learner and independently replay its predictions."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("unit", "training-head", "expected-head"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= args.timeout_sec <= 43200
        or not args.unit.startswith("rosclaw-team-")
        or not args.unit.endswith(".service")
        or "/" in args.unit
    ):
        raise ValueError("bounded SIM training unit and unused output required")
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
        if query.returncode or not {"ActiveState", "Result", "ExecMainStatus"} <= state.keys():
            raise ValueError("cannot verify training job")
        if state["ActiveState"] != "active":
            if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("training failed; no inference audit")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded model wait expired")
        time.sleep(15)
    source, dataset = args.root / "train" / "complete.json", args.root / "data" / "complete.json"
    if any(not p.is_file() or p.is_symlink() for p in (source, dataset)):
        raise ValueError("complete trained source and dataset required")
    source_hash, dataset_hash = digest(source), digest(dataset)
    report = json.loads(source.read_text())
    if (
        report.get("status") != "RELAY_BALANCE_OFFLINE_SELECTOR_HOLDOUT_NOT_PHYSICS"
        or report.get("source_head") != args.training_head
        or report.get("dataset_hash") != dataset_hash
    ):
        raise ValueError("exact trained source and data identity required")
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_audit_relay_balance_selector.py",
            "--source",
            str(source.parent),
            "--source-hash",
            source_hash,
            "--dataset",
            str(dataset.parent),
            "--dataset-hash",
            dataset_hash,
            "--output",
            str(args.output),
            "--expected-head",
            args.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
