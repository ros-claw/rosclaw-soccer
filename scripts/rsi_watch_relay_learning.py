"""Bounded SIM pipeline: wait for all relay replays, then export and learn.

No retries, live-control activation, promotion, or automatic Fresh examination.
The watcher cannot continue an agent or survive reboot by itself.
"""

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
        "--pair", nargs=3, action="append", required=True, metavar=("SOURCE", "AUDIT", "AUDIT_UNIT")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        len(args.pair) != 3
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= args.timeout_sec <= 43200
    ):
        raise ValueError("three distinct bounded jobs and unused external output required")
    units = [v[2] for v in args.pair]
    if len(set(units)) != 3 or any(
        not u.startswith("rosclaw-team-") or not u.endswith(".service") or "/" in u for u in units
    ):
        raise ValueError("distinct bounded SIM-only units required")
    deadline = time.monotonic() + args.timeout_sec
    admitted = None
    while admitted is None:
        current = []
        pending = False
        for source_name, audit_name, unit in args.pair:
            result = subprocess.run(
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
            state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            if result.returncode or not {"ActiveState", "Result", "ExecMainStatus"} <= state.keys():
                raise ValueError("cannot verify own replay process; no learning")
            audit_file, source_file = (
                Path(audit_name) / "complete.json",
                Path(source_name) / "complete.json",
            )
            if state["ActiveState"] == "active":
                pending = True
                continue
            if (
                state["Result"] != "success"
                or state["ExecMainStatus"] != "0"
                or not source_file.is_file()
                or not audit_file.is_file()
                or source_file.is_symlink()
                or audit_file.is_symlink()
            ):
                raise ValueError("failed, missing or unverifiable own replay; no training")
            source_hash, audit_hash = digest(source_file), digest(audit_file)
            source, audit = (json.loads(p.read_text()) for p in (source_file, audit_file))
            if (
                source.get("source_head") != args.source_head
                or source.get("status") != "NEURAL_RELAY_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                or audit.get("producer_hash") != source_hash
                or audit.get("status") != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
                or audit.get("worlds") != 32
            ):
                raise ValueError("exact 32-world source and own replay required")
            current.append((Path(source_name), audit_file, audit_hash))
        if not pending:
            admitted = current
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("no automatic retry after bounded replay wait")
        time.sleep(15)
    args.output.mkdir(parents=True)
    data_root, audit_root = args.output / "data", args.output / "data-audit"
    command = [
        sys.executable,
        "scripts/rsi_export_relay_fall_dataset.py",
        "--output",
        str(data_root),
        "--expected-head",
        args.expected_head,
    ]
    for source, audit_file, audit_hash in admitted:
        command += ["--pair", str(source), str(audit_file), audit_hash]
    subprocess.run(command, check=True)
    data_hash = digest(data_root / "complete.json")
    command = [
        sys.executable,
        "scripts/rsi_audit_relay_fall_dataset.py",
        "--dataset",
        str(data_root),
        "--dataset-hash",
        data_hash,
        "--output",
        str(audit_root / "complete.json"),
        "--expected-head",
        args.expected_head,
    ]
    for source, _, _ in admitted:
        command += ["--source", str(source)]
    subprocess.run(command, check=True)
    audit_hash = digest(audit_root / "complete.json")
    for shard in (0, 1, 2):
        subprocess.run(
            [
                sys.executable,
                "scripts/rsi_train_relay_fall_predictor.py",
                "--dataset",
                str(data_root),
                "--dataset-hash",
                data_hash,
                "--audit",
                str(audit_root),
                "--audit-hash",
                audit_hash,
                "--output",
                str(args.output / f"train-{shard}"),
                "--shard",
                str(shard),
                "--expected-head",
                args.expected_head,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
