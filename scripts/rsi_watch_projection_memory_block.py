"""Review two completed replayed projection studies in one development block."""

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
    for key in ("raw", "executed", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("unit", "producer-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        not re.fullmatch(
            r"rosclaw-receiver-projection-memory-paired-b[12]-[0-9]{8}\.service", a.unit
        )
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.raw.resolve() == a.executed.resolve()
    ):
        raise ValueError("exact bounded worker and unused distinct external output required")
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
        if worker_ready(a.raw, state) and worker_ready(a.executed, state):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("observation expired; do not restart the worker")
        time.sleep(15)
    for root in (a.raw, a.executed):
        if (
            json.loads((root / "source/complete.json").read_text()).get("source_head")
            != a.producer_head
        ):
            raise ValueError("exact source producer required for both studies")
        declaration = json.loads((root / "source/declaration.json").read_text())
        if declaration.get("development_context_block") != int(a.unit.split("-b", 1)[1][0]):
            raise ValueError("study block must match the observed service")
    raw_source_hash = digest(a.raw / "source/complete.json")
    raw_audit_hash = digest(a.raw / "audit/complete.json")
    a.output.mkdir(parents=True)
    for name, root in (("raw", a.raw), ("executed", a.executed)):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.rsi_review_receiver_leg_limit",
                "--source",
                str(root),
                "--source-hash",
                digest(root / "source/complete.json"),
                "--audit-hash",
                digest(root / "audit/complete.json"),
                "--prior",
                str(a.raw),
                "--prior-hash",
                raw_source_hash,
                "--prior-audit-hash",
                raw_audit_hash,
                "--prior-leg-projection",
                "--expected-head",
                a.expected_head,
                "--output",
                str(a.output / name),
            ],
            check=True,
        )
    old, new = a.output / "raw/complete.json", a.output / "executed/complete.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_compare_leg_projection_memory",
            "--old",
            str(old),
            "--old-hash",
            digest(old),
            "--new",
            str(new),
            "--new-hash",
            digest(new),
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output / "comparison"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
