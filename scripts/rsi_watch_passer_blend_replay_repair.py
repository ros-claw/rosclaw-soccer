"""Replay a completed source after the original audit fails; never rerun physics."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


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
        re.fullmatch(r"rosclaw-passer-blend-fix-b0-[0-9]{8}\.service", a.unit) is None
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("exact bounded worker and unused external output required")
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
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
        if state.get("LoadState") != "loaded":
            raise ValueError("exact worker state unavailable")
        if state.get("ActiveState") not in ("active", "activating"):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("worker observation expired; do not restart")
        time.sleep(15)
    root = a.source / "source"
    receipt = root / "complete.json"
    if (
        state.get("ActiveState") != "failed"
        or state.get("Result") != "exit-code"
        or not receipt.is_file()
        or (root / "failed.json").exists()
        or (a.source / "audit/complete.json").exists()
        or json.loads(receipt.read_text()).get("source_head") != a.producer_head
    ):
        raise ValueError("completed exact source and failed original audit required")
    source_hash = digest(receipt)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    (a.output / "source").symlink_to(root.resolve(), target_is_directory=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_audit_neural_relay_chain",
            "--source-root",
            str(root),
            "--source-head",
            a.producer_head,
            "--source-hash",
            source_hash,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output / "audit"),
        ],
        check=True,
    )
    if digest(receipt) != source_hash:
        raise ValueError("source changed during independent replay")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_review_passer_velocity_recovery",
            "--source",
            str(a.output),
            "--source-hash",
            source_hash,
            "--audit-hash",
            digest(a.output / "audit/complete.json"),
            "--prior",
            str(a.prior),
            "--prior-hash",
            a.prior_hash,
            "--prior-audit-hash",
            a.prior_audit_hash,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output / "review"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
