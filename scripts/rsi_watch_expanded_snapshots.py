"""Bounded observer chaining qualified snapshots to receiver inference."""

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
    for key in ("source", "snapshots", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "snapshot-head", "expected-head", "unit"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        re.fullmatch(r"rosclaw-relay-expanded-snapshots-b0-[0-9]{8}\.service", a.unit) is None
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("exact bounded source worker and new external output required")
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
        if state.get("ActiveState") in ("active", "activating"):
            if time.monotonic() >= deadline:
                raise TimeoutError("snapshot observation expired; never restart")
            time.sleep(15)
            continue
        if not (
            (
                state.get("LoadState") == "loaded"
                and state.get("ActiveState") == "inactive"
                and state.get("Result") == "success"
                and state.get("ExecMainStatus") == "0"
            )
            or (state.get("LoadState") == "not-found" and state.get("ActiveState") == "inactive")
        ):
            raise ValueError("snapshot worker failed or cannot be inspected")
        break
    receipt = a.snapshots / "complete.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("completed bound snapshot receipt required")
    snap_hash = digest(receipt)
    data = json.loads(receipt.read_text())
    rows = data.get("rows", [])
    keys = {(r.get("context"), r.get("side"), r.get("receiver_lateral_offset_m")) for r in rows}
    expected = {
        (c, s, o) for c in (2, 3, 5, 7) for s in (-0.35, 0.35) for o in (-0.35, -0.12, 0.35)
    }
    if (
        data.get("status") != "RELAY_NONDEFAULT_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
        or data.get("source_head") != a.snapshot_head
        or data.get("source_hash") != a.source_hash
        or data.get("audit_hash") != a.audit_hash
        or len(rows) != 24
        or keys != expected
        or data.get("saved_torque_is_not_policy") is not True
        or any(
            data.get(k) is not False
            for k in (
                "goal_complete",
                "promotion_authorized",
                "hardware_authorized",
                "independent_fresh",
            )
        )
        or digest(receipt) != snap_hash
    ):
        raise ValueError("complete exact nondefault development snapshots required")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_qualify_relay_snapshot_velocity",
            "--source",
            str(a.source),
            "--source-hash",
            a.source_hash,
            "--audit-hash",
            a.audit_hash,
            "--snapshots",
            str(a.snapshots),
            "--snapshot-hash",
            snap_hash,
            "--nondefault",
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
