"""Observe the exact passer-recovery worker, then review its own native replay."""

import argparse
import json
import re
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
        re.fullmatch(r"rosclaw-passer-velocity-(?:recovery|blend)-b0-[0-9]{8}\.service", a.unit)
        is None
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("exact bounded research worker and unused output required")
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
            raise TimeoutError("observation expired; never restart the live worker")
        time.sleep(15)
    receipt = a.source / "source/complete.json"
    if json.loads(receipt.read_text()).get("source_head") != a.producer_head:
        raise ValueError("exact frozen simulation source required")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_review_passer_velocity_recovery",
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
