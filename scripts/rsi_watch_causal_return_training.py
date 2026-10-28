"""Chain audited development banks to offline causal-event diagnostics.

This is a bounded simulation-only handoff, not a football policy or promotion.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import report
from scripts.rsi_watch_expanded_return import wait_unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("trial", "scene", "baseline-source", "baseline-audit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("trial-hash", "scene-hash", "baseline-audit-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument(
        "--bank", nargs=3, action="append", required=True, metavar=("SOURCE", "UNIT", "SOURCE_HASH")
    )
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        len(args.bank) != 2
        or not 1 <= args.timeout_sec <= 43200
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or len({item[0] for item in args.bank}) != 2
        or len({item[1] for item in args.bank}) != 2
    ):
        raise ValueError("two distinct bounded banks and unused external output required")
    for _, unit, source_hash in args.bank:
        if (
            re.fullmatch(r"rosclaw-relay-lateral-replay-b[12]-recovery-[0-9]{8}\.service", unit)
            is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash) is None
        ):
            raise ValueError("exact recovery unit and source hash required")
    deadline = time.monotonic() + args.timeout_sec
    for _, unit, _ in args.bank:
        wait_unit(unit, deadline)
    original = [(args.baseline_source, args.baseline_audit, args.baseline_audit_hash)]
    for source_name, _, source_hash in args.bank:
        source = Path(source_name)
        audit = source.parent / "audit" / "complete.json"
        if audit.is_symlink() or not audit.is_file():
            raise ValueError("completed independent audit required")
        audit_hash = digest(audit)
        _, _, actual_hash = report(
            source,
            audit,
            audit_hash,
            "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
            worlds=32,
        )
        if actual_hash != source_hash or digest(source / "scene.xml") != args.scene_hash:
            raise ValueError("source and scene must match pinned development bank")
        original.append((source, audit, audit_hash))
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_train_causal_return_detector",
        "--trial",
        str(args.trial),
        "--trial-hash",
        args.trial_hash,
        "--scene",
        str(args.scene),
        "--scene-hash",
        args.scene_hash,
        "--expected-head",
        args.expected_head,
        "--output",
        str(args.output),
    ]
    for source, audit, audit_hash in original:
        command.extend(("--original-source", str(source), str(audit), audit_hash))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
