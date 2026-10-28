"""Run the single-control paired SIM proof only after causal role qualification."""

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
    for name in ("source", "snapshots", "frontier", "role-proof", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "source-hash",
        "audit-hash",
        "snapshot-hash",
        "frontier-hash",
        "frontier-head",
        "role-head",
        "unit",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        re.fullmatch(r"rosclaw-relay-snapshot-role-proof-b0-[0-9]{8}\.service", a.unit) is None
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("exact bounded role worker and unused external output required")
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
                raise TimeoutError("role proof observation expired; never restart")
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
            raise ValueError("exact role worker failed or cannot be inspected")
        break
    if a.role_proof.is_symlink() or not a.role_proof.is_file():
        raise ValueError("completed bound role proof required")
    role_hash = digest(a.role_proof)
    role = json.loads(a.role_proof.read_text())
    if (
        role.get("status") != "PASSER_SNAPSHOT_OWN_RECURRENT_INFERENCE_MATCH_NOT_POLICY_RESUME"
        or role.get("source_head") != a.role_head
        or role.get("source_hash") != a.source_hash
        or role.get("audit_hash") != a.audit_hash
        or role.get("snapshot_hash") != a.snapshot_hash
        or role.get("frontier_hash") != a.frontier_hash
        or len(role.get("rows", [])) != 8
        or len({(r.get("context"), r.get("side")) for r in role.get("rows", [])}) != 8
        or role.get("receiver_role_selector_verified_at_boundary") is not True
        or any(r.get("receiver_role_command_exact") is not True for r in role["rows"])
        or any(
            role.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("all eight exact source-bound causal role proofs required")
    if digest(a.role_proof) != role_hash:
        raise ValueError("role proof changed during admission")
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
            a.snapshot_hash,
            "--frontier",
            str(a.frontier),
            "--frontier-hash",
            a.frontier_hash,
            "--role-proof",
            str(a.role_proof),
            "--role-proof-hash",
            role_hash,
            "--frontier-head",
            a.frontier_head,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
