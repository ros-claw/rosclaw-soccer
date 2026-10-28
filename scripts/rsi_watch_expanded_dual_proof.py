"""Wait for exact expanded inference proof, then recompute dual physics."""

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
    for key in ("source", "snapshots", "frontier", "role-proof", "velocity-proof", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "source-hash",
        "audit-hash",
        "frontier-head",
        "frontier-hash",
        "role-proof-hash",
        "velocity-head",
        "expected-head",
        "unit",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        re.fullmatch(r"rosclaw-relay-expanded-velocity-proof-b0-[0-9]{8}\.service", a.unit) is None
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.role_proof.is_symlink()
        or digest(a.role_proof) != a.role_proof_hash
    ):
        raise ValueError("exact worker and pinned baseline role evidence required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        result = subprocess.run(
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
        state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if state.get("ActiveState") in ("active", "activating"):
            if time.monotonic() >= deadline:
                raise TimeoutError("expanded inference observation expired; never restart")
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
            raise ValueError("expanded inference worker failed or cannot be inspected")
        break
    snapshot_receipt = a.snapshots / "complete.json"
    if (
        snapshot_receipt.is_symlink()
        or not snapshot_receipt.is_file()
        or a.velocity_proof.is_symlink()
        or not a.velocity_proof.is_file()
    ):
        raise ValueError("completed snapshot and inference receipts required")
    snap_hash, velocity_hash = digest(snapshot_receipt), digest(a.velocity_proof)
    proof = json.loads(a.velocity_proof.read_text())
    if (
        proof.get("status")
        != "RECEIVER_NONDEFAULT_SNAPSHOT_OWN_VELOCITY_INFERENCE_MATCH_NOT_POLICY_RESUME"
        or proof.get("source_head") != a.velocity_head
        or proof.get("source_hash") != a.source_hash
        or proof.get("audit_hash") != a.audit_hash
        or proof.get("snapshot_hash") != snap_hash
        or len(proof.get("rows", [])) != 24
        or any(
            proof.get(k) is not False
            for k in (
                "independent_fresh",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
        or digest(a.velocity_proof) != velocity_hash
        or digest(snapshot_receipt) != snap_hash
    ):
        raise ValueError("complete exact expanded development inference proof required")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_qualify_relay_snapshot_passer",
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
            "--frontier",
            str(a.frontier),
            "--frontier-head",
            a.frontier_head,
            "--frontier-hash",
            a.frontier_hash,
            "--role-proof",
            str(a.role_proof),
            "--role-proof-hash",
            a.role_proof_hash,
            "--nondefault",
            "--expanded-velocity-proof",
            str(a.velocity_proof),
            "--expanded-velocity-proof-hash",
            velocity_hash,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
