"""Wait for all grouped history models before independent paired comparison."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "history", "continuous", "parent", "parent-audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "history-hash",
        "continuous-hash",
        "parent-hash",
        "parent-audit-hash",
        "unit",
        "producer-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 43200
        or not a.unit.startswith("rosclaw-team-")
        or not a.unit.endswith(".service")
        or "/" in a.unit
    ):
        raise ValueError("bounded training unit and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.unit,
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
        state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
        if q.returncode or not {"ActiveState", "Result", "ExecMainStatus"} <= state.keys():
            raise ValueError("cannot verify training unit")
        if state["ActiveState"] != "active":
            if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("training failed; no paired inference audit")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded training wait expired; no retry")
        time.sleep(15)
    receipt = a.source / "complete.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError("complete trained source required")
    source_hash = digest(receipt)
    source = json.loads(receipt.read_text())
    if (
        source.get("status") != "RELAY_CONTACT_HISTORY_FORECAST_NOT_CONTROL"
        or source.get("source_head") != a.producer_head
        or source.get("history_hash") != a.history_hash
        or source.get("continuous_hash") != a.continuous_hash
        or source.get("parent_hash") != a.parent_hash
        or source.get("parent_audit_hash") != a.parent_audit_hash
    ):
        raise ValueError("exact trained matched source required")
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_audit_contact_motor_history_models.py",
            "--source",
            str(a.source),
            "--source-hash",
            source_hash,
            "--history",
            str(a.history),
            "--history-hash",
            a.history_hash,
            "--continuous",
            str(a.continuous),
            "--continuous-hash",
            a.continuous_hash,
            "--parent",
            str(a.parent),
            "--parent-hash",
            a.parent_hash,
            "--parent-audit",
            str(a.parent_audit),
            "--parent-audit-hash",
            a.parent_audit_hash,
            "--producer-head",
            a.producer_head,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
