"""Start return-chain replay only after exact readiness and shadow proofs."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def wait_unit(unit: str, deadline: float) -> None:
    while True:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
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
                raise TimeoutError("prerequisite observation expired; never restart")
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
            raise ValueError("prerequisite failed or cannot be inspected")
        return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "source",
        "snapshots",
        "frontier",
        "role-proof",
        "velocity-proof",
        "boundary-proof",
        "shadow-proof",
        "output",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "source-hash",
        "audit-hash",
        "frontier-head",
        "frontier-hash",
        "role-proof-hash",
        "boundary-head",
        "shadow-head",
        "expected-head",
        "boundary-unit",
        "shadow-unit",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        re.fullmatch(r"rosclaw-relay-expanded-boundary-proof-b0-[0-9]{8}\.service", a.boundary_unit)
        is None
        or re.fullmatch(r"rosclaw-relay-expanded-shadow-proof-b0-[0-9]{8}\.service", a.shadow_unit)
        is None
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.role_proof.is_symlink()
        or digest(a.role_proof) != a.role_proof_hash
    ):
        raise ValueError("exact named workers and pinned role evidence required")
    deadline = time.monotonic() + a.timeout_sec
    wait_unit(a.boundary_unit, deadline)
    wait_unit(a.shadow_unit, deadline)
    snapshot = a.snapshots / "complete.json"
    paths = (snapshot, a.velocity_proof, a.boundary_proof, a.shadow_proof)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("complete source-bound prerequisite receipts required")
    snap_hash, velocity_hash, boundary_hash, shadow_hash = map(digest, paths)
    boundary, shadow = (json.loads(path.read_text()) for path in (a.boundary_proof, a.shadow_proof))
    for proof, head in ((boundary, a.boundary_head), (shadow, a.shadow_head)):
        if (
            proof.get("source_head") != head
            or proof.get("source_hash") != a.source_hash
            or proof.get("audit_hash") != a.audit_hash
            or proof.get("snapshot_hash") != snap_hash
            or proof.get("expanded_velocity_proof_hash") != velocity_hash
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
        ):
            raise ValueError("exact unpromoted expanded prerequisite lineage required")
    if (
        boundary.get("status")
        != "RELAY_NONDEFAULT_CAUSAL_RECEIVING_BOUNDARY_MATCH_NOT_FOOTBALL_SUCCESS"
        or boundary.get("causal_receiving_boundary_reconstructed") is not True
        or shadow.get("status")
        != "RELAY_NONDEFAULT_DUAL_NEURAL_20_CONTROLS_MATCH_NOT_FOOTBALL_SUCCESS"
        or shadow.get("own_shadow_recurrence_verified") is not True
        or any(r.get("own_shadow_verified") is not True for r in shadow["rows"])
        or tuple(map(digest, paths)) != (snap_hash, velocity_hash, boundary_hash, shadow_hash)
    ):
        raise ValueError("causal boundary and own-shadow prerequisite proofs required")
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
            "--dynamic-boundary",
            "--with-shadow",
            "--return-handoff",
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
