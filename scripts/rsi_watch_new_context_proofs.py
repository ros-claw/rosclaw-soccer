"""Continue a declared development bank from snapshots to own dual inference.

Bounded observer only: a failed, missing or replaced capture is never retried.
All generated results remain SIM-only and are not football success evidence.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def valid_capture(
    data: dict, *, head: str, source_hash: str, audit_hash: str, contexts: tuple
) -> bool:
    expected = {(c, s, o) for c in contexts for s in (-0.35, 0.35) for o in (-0.35, -0.12, 0.35)}
    rows = data.get("rows", [])
    return bool(
        data.get("status") == "RELAY_NONDEFAULT_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
        and data.get("source_head") == head
        and data.get("source_hash") == source_hash
        and data.get("audit_hash") == audit_hash
        and data.get("contexts") == list(contexts)
        and len(rows) == 24
        and len({r.get("index") for r in rows}) == 24
        and {(r.get("context"), r.get("side"), r.get("receiver_lateral_offset_m")) for r in rows}
        == expected
        and data.get("saved_torque_is_not_policy") is True
        and all(
            data.get(k) is False
            for k in (
                "independent_fresh",
                "promotion_authorized",
                "hardware_authorized",
                "goal_complete",
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "snapshots", "velocity-output", "dual-output", "frontier", "role-proof"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in (
        "source-hash",
        "audit-hash",
        "capture-head",
        "expected-head",
        "frontier-head",
        "frontier-hash",
        "role-proof-hash",
    ):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--contexts", type=int, nargs=4, required=True)
    parser.add_argument("--capture-pid", type=int, required=True)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    args = parser.parse_args()
    clean(args.expected_head)
    contexts = tuple(args.contexts)
    if (
        contexts not in ((8, 9, 11, 13), (15, 16, 17, 18))
        or args.capture_pid <= 1
        or not 1 <= args.timeout_sec <= 3600
        or any(
            p.exists() or p.resolve().is_relative_to(Path.cwd())
            for p in (args.velocity_output, args.dual_output)
        )
        or args.role_proof.is_symlink()
        or digest(args.role_proof) != args.role_proof_hash
    ):
        raise ValueError("declared block and unique bounded outputs required")
    receipt = args.snapshots / "complete.json"
    failure = args.snapshots / "failed.json"
    deadline = time.monotonic() + args.timeout_sec
    while not receipt.is_file():
        if failure.exists():
            raise ValueError("capture failed; preserve evidence and do not retry")
        if time.monotonic() >= deadline:
            raise TimeoutError("capture observation expired; do not restart")
        try:
            command = Path(f"/proc/{args.capture_pid}/cmdline").read_bytes().split(b"\0")
        except FileNotFoundError as error:
            raise ValueError("capture process vanished before receipt") from error
        if (
            b"scripts.rsi_capture_relay_contact_snapshots" not in command
            or os.fsencode(args.snapshots) not in command
        ):
            raise ValueError("capture PID no longer identifies the expected source job")
        time.sleep(10)
    if receipt.is_symlink() or failure.exists():
        raise ValueError("exact successful capture receipt required")
    snap_hash = digest(receipt)
    if (
        not valid_capture(
            json.loads(receipt.read_text()),
            head=args.capture_head,
            source_hash=args.source_hash,
            audit_hash=args.audit_hash,
            contexts=contexts,
        )
        or digest(receipt) != snap_hash
    ):
        raise ValueError("capture does not bind complete exact development bank")
    common = [
        "--source",
        str(args.source),
        "--source-hash",
        args.source_hash,
        "--audit-hash",
        args.audit_hash,
        "--snapshots",
        str(args.snapshots),
        "--snapshot-hash",
        snap_hash,
        "--nondefault",
        "--expected-head",
        args.expected_head,
    ]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_qualify_relay_snapshot_velocity",
            *common,
            "--output",
            str(args.velocity_output),
        ],
        check=True,
    )
    proof = args.velocity_output / "complete.json"
    if proof.is_symlink() or not proof.is_file():
        raise ValueError("completed own receiver inference receipt required")
    proof_hash = digest(proof)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_qualify_relay_snapshot_passer",
            *common,
            "--frontier",
            str(args.frontier),
            "--frontier-head",
            args.frontier_head,
            "--frontier-hash",
            args.frontier_hash,
            "--role-proof",
            str(args.role_proof),
            "--role-proof-hash",
            args.role_proof_hash,
            "--expanded-velocity-proof",
            str(proof),
            "--expanded-velocity-proof-hash",
            proof_hash,
            "--output",
            str(args.dual_output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
