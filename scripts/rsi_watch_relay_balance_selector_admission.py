"""Wait for independent selector replay and persist a fail-closed decision."""

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
    for name in ("source", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("unit", "training-head", "audit-head", "expected-head"):
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
        raise ValueError("bounded replay unit and unused external output required")
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
            raise ValueError("cannot verify replay unit")
        if state["ActiveState"] != "active":
            if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("replay failed; admission prohibited")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded replay wait expired")
        time.sleep(15)
    sp, ap = args.source / "complete.json", args.audit / "complete.json"
    if any(not p.is_file() or p.is_symlink() for p in (sp, ap)):
        raise ValueError("complete selector and replay required")
    source_hash, audit_hash = digest(sp), digest(ap)
    if (
        json.loads(sp.read_text()).get("source_head") != args.training_head
        or json.loads(ap.read_text()).get("source_head") != args.audit_head
    ):
        raise ValueError("exact training and audit sources required")
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_admit_relay_balance_selector.py",
            "--source",
            str(args.source),
            "--audit",
            str(args.audit),
            "--source-hash",
            source_hash,
            "--audit-hash",
            audit_hash,
            "--output",
            str(args.output),
            "--expected-head",
            args.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
