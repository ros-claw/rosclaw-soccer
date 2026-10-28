"""Gate paired full-horizon SIM rollouts on verified shared actor continuation."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def valid_dual(
    proof: dict,
    *,
    source_hash: str,
    audit_hash: str,
    snapshot_hash: str,
    velocity_hash: str,
    contexts: tuple[int, ...],
) -> bool:
    rows = proof.get("rows", [])
    expected = {(c, s, o) for c in contexts for s in (-0.35, 0.35) for o in (-0.35, -0.12, 0.35)}
    return bool(
        proof.get("status") == "RELAY_NONDEFAULT_DUAL_NEURAL_20_CONTROLS_MATCH_NOT_FOOTBALL_SUCCESS"
        and proof.get("source_hash") == source_hash
        and proof.get("audit_hash") == audit_hash
        and proof.get("snapshot_hash") == snapshot_hash
        and proof.get("expanded_velocity_proof_hash") == velocity_hash
        and proof.get("coupled_physics_continuation_verified") is True
        and proof.get("native_steps") == 4800
        and len(rows) == 24
        and len({r.get("index") for r in rows}) == 24
        and {(r.get("context"), r.get("side"), r.get("receiver_lateral_offset_m")) for r in rows}
        == expected
        and all(
            r.get("consecutive_controls") == 20 and r.get("coupled_native_substeps_exact") is True
            for r in rows
        )
        and all(
            proof.get(k) is False
            for k in (
                "independent_fresh",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "future_controls_used",
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "source",
        "snapshots",
        "velocity-proof",
        "dual-proof",
        "baseline-output",
        "candidate-output",
        "review-output",
        "frontier",
        "role-proof",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in (
        "source-hash",
        "audit-hash",
        "expected-head",
        "frontier-head",
        "frontier-hash",
        "role-proof-hash",
        "unit",
    ):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--contexts", type=int, nargs=4, required=True)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    args = parser.parse_args()
    clean(args.expected_head)
    contexts = tuple(args.contexts)
    if (
        contexts not in ((8, 9, 11, 13), (15, 16, 17, 18))
        or re.fullmatch(r"rosclaw-context-b[12]-proof-chain-b0-20261029\.service", args.unit)
        is None
        or not 1 <= args.timeout_sec <= 7200
        or any(
            p.exists() or p.resolve().is_relative_to(Path.cwd())
            for p in (args.baseline_output, args.candidate_output, args.review_output)
        )
        or args.role_proof.is_symlink()
        or digest(args.role_proof) != args.role_proof_hash
    ):
        raise ValueError("declared development bank and unique external outputs required")
    deadline = time.monotonic() + args.timeout_sec
    while not args.dual_proof.is_file():
        status = subprocess.run(
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
            check=True,
        )
        state = dict(line.split("=", 1) for line in status.stdout.splitlines() if "=" in line)
        if state.get("ActiveState") not in ("active", "activating"):
            raise ValueError("proof chain ended without receipt; preserve its error")
        if time.monotonic() >= deadline:
            raise TimeoutError("proof observation expired; never duplicate source jobs")
        time.sleep(15)
    snapshot = args.snapshots / "complete.json"
    if any(
        p.is_symlink() or not p.is_file() for p in (snapshot, args.velocity_proof, args.dual_proof)
    ):
        raise ValueError("all three exact prerequisite receipts required")
    snap_hash, velocity_hash, dual_hash = (
        digest(p) for p in (snapshot, args.velocity_proof, args.dual_proof)
    )
    proof = json.loads(args.dual_proof.read_text())
    if not valid_dual(
        proof,
        source_hash=args.source_hash,
        audit_hash=args.audit_hash,
        snapshot_hash=snap_hash,
        velocity_hash=velocity_hash,
        contexts=contexts,
    ):
        raise ValueError("exact 24-world dual control proof required")
    clean(args.expected_head)
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
        "--nondefault",
        "--dynamic-boundary",
        "--with-shadow",
        "--return-handoff",
        "--smooth-transition",
        "--smooth-transition-duration",
        "10",
        "--expanded-velocity-proof",
        str(args.velocity_proof),
        "--expanded-velocity-proof-hash",
        velocity_hash,
        "--expected-head",
        args.expected_head,
    ]
    command = [sys.executable, "-m", "scripts.rsi_qualify_relay_snapshot_passer", *common]
    subprocess.run([*command, "--output", str(args.baseline_output)], check=True)
    if (
        digest(args.dual_proof) != dual_hash
        or digest(snapshot) != snap_hash
        or digest(args.velocity_proof) != velocity_hash
    ):
        raise ValueError("prerequisite changed during baseline")
    baseline = args.baseline_output / "complete.json"
    baseline_hash = digest(baseline)
    subprocess.run(
        [
            *command,
            "--long-probe",
            "--probe-leg-rad",
            "0",
            ".08546",
            "0",
            ".020332",
            "-.064212",
            "0",
            "0",
            "-.08546",
            "0",
            ".020332",
            "-.064212",
            "0",
            "--output",
            str(args.candidate_output),
        ],
        check=True,
    )
    candidate = args.candidate_output / "complete.json"
    candidate_hash = digest(candidate)
    if any(
        digest(p) != h
        for p, h in (
            (args.dual_proof, dual_hash),
            (snapshot, snap_hash),
            (args.velocity_proof, velocity_hash),
            (baseline, baseline_hash),
        )
    ):
        raise ValueError("paired evidence changed during candidate")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_compare_long_receiver_probe",
            "--baseline",
            str(baseline),
            "--baseline-hash",
            baseline_hash,
            "--candidate",
            str(candidate),
            "--candidate-hash",
            candidate_hash,
            "--snapshots",
            str(snapshot),
            "--snapshot-hash",
            snap_hash,
            "--scene",
            str(args.source / "source/scene.xml"),
            "--expected-head",
            args.expected_head,
            "--output",
            str(args.review_output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
