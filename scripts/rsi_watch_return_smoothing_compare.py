"""Compare two finished return trials only after both have valid receipts."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_watch_expanded_return import wait_unit


def check_receipt(path: Path, expected_status: str, expected_head: str, source_hash: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("completed receipt required")
    before = digest(path)
    data = json.loads(path.read_text())
    if (
        data.get("status") != expected_status
        or data.get("source_head") != expected_head
        or data.get("source_hash") != source_hash
        or len(data.get("rows", [])) != 24
        or any(
            data.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "independent_fresh",
            )
        )
        or digest(path) != before
    ):
        raise ValueError("exact unchanged unpromoted development receipt required")
    return before


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "baseline", "candidate", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "source-hash",
        "audit-hash",
        "baseline-head",
        "candidate-head",
        "expected-head",
        "baseline-unit",
        "candidate-unit",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        not 1 <= a.timeout_sec <= 7200
        or re.fullmatch(
            r"rosclaw-relay-expanded-return-proof-b0-[0-9]{8}\.service", a.baseline_unit
        )
        is None
        or re.fullmatch(
            r"rosclaw-relay-smooth-candidate-b[0-9]+-[0-9]{8}\.service", a.candidate_unit
        )
        is None
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("bounded wait and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    wait_unit(a.baseline_unit, deadline)
    wait_unit(a.candidate_unit, deadline)
    baseline_hash = check_receipt(
        a.baseline / "complete.json",
        "RELAY_NONDEFAULT_CAUSAL_RETURN_CHAIN_MATCH_NOT_FOOTBALL_SUCCESS",
        a.baseline_head,
        a.source_hash,
    )
    candidate_hash = check_receipt(
        a.candidate / "complete.json",
        "RELAY_NONDEFAULT_SMOOTH_RETURN_PHYSICAL_CANDIDATE_NOT_SUCCESS",
        a.candidate_head,
        a.source_hash,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_compare_return_smoothing",
            "--source",
            str(a.source),
            "--baseline",
            str(a.baseline),
            "--candidate",
            str(a.candidate),
            "--output",
            str(a.output),
            "--source-hash",
            a.source_hash,
            "--audit-hash",
            a.audit_hash,
            "--baseline-hash",
            baseline_hash,
            "--candidate-hash",
            candidate_hash,
            "--expected-head",
            a.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
