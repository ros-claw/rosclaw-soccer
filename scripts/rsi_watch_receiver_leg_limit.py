"""Observe the exact projection pilot and launch its review only after replay."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_watch_passer_residual_pair import worker_ready


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "prior", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("unit", "producer-head", "prior-hash", "prior-audit-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not a.unit.startswith("rosclaw-receiver-leg-limit-")
        or not a.unit.endswith(".service")
        or "/" in a.unit
        or not 1 <= a.timeout_sec <= 7200
    ):
        raise ValueError("one exact bounded simulation unit and unused output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.unit,
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
        if worker_ready(a.source, state):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("observation expired; do not restart the worker")
        time.sleep(15)
    receipt = a.source / "source/complete.json"
    if json.loads(receipt.read_text()).get("source_head") != a.producer_head:
        raise ValueError("exact simulation producer required")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_review_receiver_leg_limit",
            "--source",
            str(a.source),
            "--source-hash",
            digest(receipt),
            "--audit-hash",
            digest(a.source / "audit/complete.json"),
            "--prior",
            str(a.prior),
            "--prior-hash",
            a.prior_hash,
            "--prior-audit-hash",
            a.prior_audit_hash,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
