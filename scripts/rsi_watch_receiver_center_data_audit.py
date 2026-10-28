"""Wait for reviewed center data export and independently audit it once."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SOURCE", "AUDIT"))
    for name in ("review", "dataset", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("export-unit", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.pair) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not a.export_unit.startswith("rosclaw-receiver-center-strict-export-")
        or not a.export_unit.endswith(".service")
        or "/" in a.export_unit
        or not 1 <= a.timeout_sec <= 43200
    ):
        raise ValueError("three exact sources, bounded export job and unused output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.export_unit,
                "-p",
                "LoadState",
                "-p",
                "ActiveState",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
        if state.get("ActiveState") in ("active", "activating"):
            if time.monotonic() >= deadline:
                raise TimeoutError("export observation expired; never restart live job")
            time.sleep(15)
            continue
        if state.get("LoadState") == "not-found" and state.get("ActiveState") == "inactive":
            break
        if not (
            state.get("LoadState") == "loaded"
            and state.get("ActiveState") == "inactive"
            and state.get("Result") == "success"
            and state.get("ExecMainStatus") == "0"
        ):
            raise ValueError("strict data export failed")
        break
    receipt = a.dataset / "complete.json"
    if a.review.is_symlink() or receipt.is_symlink():
        raise ValueError("owned complete review and dataset receipts required")
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_audit_receiver_onpolicy",
        "--receiver-center-strict",
        "--review",
        str(a.review),
        "--review-hash",
        digest(a.review),
        "--dataset",
        str(a.dataset),
        "--dataset-hash",
        digest(receipt),
        "--producer-head",
        a.producer_head,
        "--expected-head",
        a.expected_head,
        "--output",
        str(a.output),
    ]
    for pair in a.pair:
        command.extend(("--pair", *pair))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
