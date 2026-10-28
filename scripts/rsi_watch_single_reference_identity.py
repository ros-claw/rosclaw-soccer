"""Run one pinned development reference audit when its physical worker completes."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_watch_oof_physical_review import valid_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "candidate", "actual", "risk", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in (
        "baseline-hash",
        "candidate-hash",
        "risk-hash",
        "snapshot-hash",
        "physical-head",
        "expected-head",
        "unit",
    ):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    a = parser.parse_args()
    clean(a.expected_head)
    if (
        re.fullmatch(r"rosclaw-single-b[012]-physical-b0-20261029\.service", a.unit) is None
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(
            p.is_symlink() or digest(p) != h
            for p, h in (
                (a.baseline, a.baseline_hash),
                (a.candidate, a.candidate_hash),
                (a.risk, a.risk_hash),
            )
        )
    ):
        raise ValueError("exact worker, pinned inputs and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    while not a.actual.is_file():
        result = subprocess.run(
            ["systemctl", "--user", "show", a.unit, "-p", "ActiveState"],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip() not in ("ActiveState=active", "ActiveState=activating"):
            raise ValueError("physical worker ended without complete evidence")
        if time.monotonic() >= deadline:
            raise TimeoutError("observation expired; do not restart the worker")
        time.sleep(15)
    if a.actual.is_symlink():
        raise ValueError("non-symlink physical receipt required")
    actual_hash = digest(a.actual)
    if (
        not valid_candidate(
            json.loads(a.actual.read_text()),
            source_head=a.physical_head,
            snapshot_hash=a.snapshot_hash,
            risk_hash=a.risk_hash,
            variant="single",
        )
        or digest(a.actual) != actual_hash
    ):
        raise ValueError("exact single-model physical development result required")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_audit_oof_reference_identity",
            "--baseline",
            str(a.baseline),
            "--baseline-hash",
            a.baseline_hash,
            "--candidate",
            str(a.candidate),
            "--candidate-hash",
            a.candidate_hash,
            "--actual",
            str(a.actual),
            "--actual-hash",
            actual_hash,
            "--risk",
            str(a.risk),
            "--risk-hash",
            a.risk_hash,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
