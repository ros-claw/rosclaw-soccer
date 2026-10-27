"""Wait for complete paired worlds, then export, read back, and fit offline values."""

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
    parser.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("EXPANDED_UNIT", "EXPANDED_ROOT", "PILOT_SOURCE", "PILOT_AUDIT"),
    )
    parser.add_argument("--output", type=Path, required=True)
    for name in ("expected-head", "expanded-head", "pilot-head"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args = parser.parse_args()
    clean(args.expected_head)
    units = [item[0] for item in args.group]
    if (
        len(args.group) != 3
        or len(set(units)) != 3
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= args.timeout_sec <= 43200
        or any(
            not u.startswith("rosclaw-team-") or not u.endswith(".service") or "/" in u
            for u in units
        )
    ):
        raise ValueError("three bounded simulator jobs and unused output required")
    deadline = time.monotonic() + args.timeout_sec
    admitted = None
    while admitted is None:
        pending, pairs = False, []
        for unit, expanded, pilot, pilot_audit in args.group:
            query = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
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
                raise ValueError("cannot verify expanded simulator unit")
            if state["ActiveState"] == "active":
                pending = True
                continue
            if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("expanded physics/replay failed; no training")
            roots = (Path(pilot), Path(expanded) / "source")
            audits = (
                Path(pilot_audit) / "complete.json",
                Path(expanded) / "audit" / "complete.json",
            )
            for root, audit_path, head, status, worlds in zip(
                roots,
                audits,
                (args.pilot_head, args.expanded_head),
                (
                    "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY",
                    "NEURAL_RELAY_BALANCE_EXPANDED64_COLLECTED_PENDING_OWN_REPLAY",
                ),
                (32, 64),
                strict=True,
            ):
                source_file = root / "complete.json"
                if any(not p.is_file() or p.is_symlink() for p in (source_file, audit_path)):
                    raise ValueError("complete source and audit required")
                source_hash, audit_hash = digest(source_file), digest(audit_path)
                source, audit = (json.loads(p.read_text()) for p in (source_file, audit_path))
                if (
                    source.get("source_head") != head
                    or source.get("status") != status
                    or audit.get("status") != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
                    or audit.get("producer_hash") != source_hash
                    or audit.get("worlds") != worlds
                ):
                    raise ValueError("exact full source/replay commitments required")
                pairs.append((str(root), str(audit_path), audit_hash))
        if not pending:
            admitted = pairs
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded physical wait expired; no retry")
        time.sleep(15)
    args.output.mkdir(parents=True)
    data = args.output / "data"
    command = [
        sys.executable,
        "scripts/rsi_export_relay_balance_decisions.py",
        "--output",
        str(data),
        "--expected-head",
        args.expected_head,
    ]
    for pair in admitted:
        command.extend(("--pair", *pair))
    subprocess.run(command, check=True)
    data_hash = digest(data / "complete.json")
    audit = args.output / "data-audit"
    command = [
        sys.executable,
        "scripts/rsi_audit_relay_balance_decisions.py",
        "--dataset",
        str(data),
        "--dataset-hash",
        data_hash,
        "--output",
        str(audit),
        "--expected-head",
        args.expected_head,
    ]
    for pair in admitted:
        command.extend(("--pair", *pair))
    subprocess.run(command, check=True)
    audit_hash = digest(audit / "complete.json")
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_train_relay_balance_selector.py",
            "--dataset",
            str(data),
            "--dataset-hash",
            data_hash,
            "--audit",
            str(audit),
            "--audit-hash",
            audit_hash,
            "--output",
            str(args.output / "train"),
            "--expected-head",
            args.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
