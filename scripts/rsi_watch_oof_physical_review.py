"""Audit one actual OOF shared-world SIM run after its exact worker completes."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def valid_candidate(
    data: dict,
    *,
    source_head: str,
    snapshot_hash: str,
    risk_hash: str,
    variant: str = "ball_augmented",
) -> bool:
    rows = data.get("rows", [])
    return bool(
        variant in ("ball_augmented", "single")
        and data.get("status")
        == (
            "RELAY_SINGLE_RISK_PHYSICAL_DEVELOPMENT_NOT_SUCCESS"
            if variant == "single"
            else "RELAY_OOF_RISK_PHYSICAL_DEVELOPMENT_NOT_SUCCESS"
        )
        and data.get("source_head") == source_head
        and data.get("snapshot_hash") == snapshot_hash
        and data.get("risk_receipt_hash") == risk_hash
        and data.get("risk_variant") == variant
        and len(rows) == 24
        and len({r.get("index") for r in rows}) == 24
        and all(
            r.get("selector_class") in (0, 1) and type(r.get("risk_probability")) is float
            for r in rows
        )
        and all(
            data.get(k) is False
            for k in (
                "future_controls_used",
                "independent_fresh",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "baseline",
        "candidate",
        "snapshots",
        "scene",
        "risk",
        "review-output",
        "team-output",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in (
        "baseline-hash",
        "snapshot-hash",
        "risk-hash",
        "physical-head",
        "expected-head",
        "unit",
    ):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument(
        "--risk-variant", choices=("ball_augmented", "single"), default="ball_augmented"
    )
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        re.fullmatch(r"rosclaw-(oof|single)-b[012]-physical-b0-20261029\.service", args.unit)
        is None
        or not args.unit.startswith(
            "rosclaw-single-" if args.risk_variant == "single" else "rosclaw-oof-"
        )
        or not 1 <= args.timeout_sec <= 7200
        or any(
            p.exists() or p.resolve().is_relative_to(Path.cwd())
            for p in (args.review_output, args.team_output)
        )
        or any(
            p.is_symlink() or digest(p) != h
            for p, h in (
                (args.baseline, args.baseline_hash),
                (args.snapshots, args.snapshot_hash),
                (args.risk, args.risk_hash),
            )
        )
    ):
        raise ValueError("exact worker, pinned inputs and unused external outputs required")
    deadline = time.monotonic() + args.timeout_sec
    while not args.candidate.is_file():
        status = subprocess.run(
            ["systemctl", "--user", "show", args.unit, "-p", "ActiveState"],
            capture_output=True,
            text=True,
            check=True,
        )
        if status.stdout.strip() not in ("ActiveState=active", "ActiveState=activating"):
            raise ValueError("physical worker ended without a complete receipt")
        if time.monotonic() >= deadline:
            raise TimeoutError("physical worker observation expired; never restart")
        time.sleep(15)
    if args.candidate.is_symlink():
        raise ValueError("non-symlink physical receipt required")
    candidate_hash = digest(args.candidate)
    if (
        not valid_candidate(
            json.loads(args.candidate.read_text()),
            source_head=args.physical_head,
            snapshot_hash=args.snapshot_hash,
            risk_hash=args.risk_hash,
            variant=args.risk_variant,
        )
        or digest(args.candidate) != candidate_hash
    ):
        raise ValueError("complete source-bound OOF physical result required")
    common = [
        "--baseline",
        str(args.baseline),
        "--baseline-hash",
        args.baseline_hash,
        "--candidate",
        str(args.candidate),
        "--candidate-hash",
        candidate_hash,
        "--snapshots",
        str(args.snapshots),
        "--snapshot-hash",
        args.snapshot_hash,
        "--scene",
        str(args.scene),
        "--expected-head",
        args.expected_head,
    ]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_compare_long_receiver_probe",
            *common,
            "--risk-receipt",
            str(args.risk),
            "--risk-receipt-hash",
            args.risk_hash,
            "--output",
            str(args.review_output),
        ],
        check=True,
    )
    review_hash = digest(args.review_output)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_audit_relay_common_successor",
            *common,
            "--review",
            str(args.review_output),
            "--review-hash",
            review_hash,
            "--output",
            str(args.team_output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
