"""Run the remaining preregistered receiver seeds/blocks sequentially in SIM."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import report
from scripts.rsi_watch_passer_residual_pair import worker_ready


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receiver-motion-pair", action="store_true")
    p.add_argument(
        "--candidate",
        nargs=4,
        action="append",
        required=True,
        metavar=("CHECKPOINT", "AUDIT", "CHECKPOINT_HASH", "AUDIT_HASH"),
    )
    for name in ("frontier-root", "dataset", "pilot-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "frontier-head",
        "frontier-hash",
        "dataset-hash",
        "pilot-unit",
        "pilot-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.candidate) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not a.pilot_unit.startswith(
            "rosclaw-receiver-motion-pair-"
            if a.receiver_motion_pair
            else "rosclaw-receiver-dense-pair-"
        )
        or not a.pilot_unit.endswith(".service")
        or "/" in a.pilot_unit
        or not 1 <= a.timeout_sec <= 7200
        or digest(a.dataset) != a.dataset_hash
    ):
        raise ValueError("three pinned candidates and exact bounded pilot required")
    for i, (cp, ap, ch, ah) in enumerate(a.candidate):
        cp, ap = Path(cp), Path(ap)
        if cp.is_symlink() or ap.is_symlink() or digest(cp) != ch or digest(ap) != ah:
            raise ValueError("every candidate must be preflighted before physics")
        audit = json.loads(ap.read_text())
        if (
            audit.get("status")
            != (
                "MOTION_RECEIVER_CANDIDATE_READBACK_MATCH"
                if a.receiver_motion_pair
                else "DENSE_RECEIVER_CANDIDATE_READBACK_MATCH"
            )
            or audit.get("checkpoint_hash") != ch
            or audit.get("dataset_hash") != a.dataset_hash
            or audit.get("seed") != 92401 + i
            or audit.get("promotion_authorized") is not False
            or audit.get("hardware_authorized") is not False
        ):
            raise ValueError("all three independently audited fixed seeds required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.pilot_unit,
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
        if worker_ready(a.pilot_root, state):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("pilot observation expired; do not restart physics")
        time.sleep(15)
    pilot, _, pilot_hash = report(
        a.pilot_root / "source",
        a.pilot_root / "audit/complete.json",
        digest(a.pilot_root / "audit/complete.json"),
        "NEURAL_RELAY_RECEIVER_MOTION_PAIR16_PENDING_OWN_REPLAY"
        if a.receiver_motion_pair
        else "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    if pilot.get("source_head") != a.pilot_head:
        raise ValueError("exact completed pilot source required")
    a.output.mkdir(parents=True)
    rows = [
        dict(
            seed=92401,
            block=0,
            root=str(a.pilot_root),
            source_hash=pilot_hash,
            audit_hash=digest(a.pilot_root / "audit/complete.json"),
        )
    ]
    try:
        for i, (checkpoint, audit, checkpoint_hash, audit_hash) in enumerate(a.candidate):
            for block in range(3):
                if i == 0 and block == 0:
                    continue
                target = a.output / f"seed{i + 1}-block{block}"
                command = [
                    sys.executable,
                    "-m",
                    "scripts.rsi_run_receiver_learned_pair",
                    "--frontier-root",
                    str(a.frontier_root),
                    "--frontier-head",
                    a.frontier_head,
                    "--frontier-hash",
                    a.frontier_hash,
                    "--receiver-checkpoint",
                    checkpoint,
                    "--receiver-checkpoint-hash",
                    checkpoint_hash,
                    "--receiver-candidate-audit",
                    audit,
                    "--receiver-candidate-audit-hash",
                    audit_hash,
                    "--receiver-dataset",
                    str(a.dataset),
                    "--receiver-dataset-hash",
                    a.dataset_hash,
                    "--new-context-block",
                    str(block),
                    "--receiver-dense-pair",
                    *(["--receiver-motion-pair"] if a.receiver_motion_pair else []),
                    "--expected-head",
                    a.expected_head,
                    "--output",
                    str(target),
                ]
                print(json.dumps({"event": "start", "seed": 92401 + i, "block": block}), flush=True)
                subprocess.run(command, check=True)
                own = target / "audit/complete.json"
                source, _, sh = report(
                    target / "source",
                    own,
                    digest(own),
                    "NEURAL_RELAY_RECEIVER_MOTION_PAIR16_PENDING_OWN_REPLAY"
                    if a.receiver_motion_pair
                    else "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY",
                    worlds=16,
                )
                if source.get("source_head") != a.expected_head:
                    raise ValueError("exact current producer required")
                rows.append(
                    dict(
                        seed=92401 + i,
                        block=block,
                        root=str(target),
                        source_hash=sh,
                        audit_hash=digest(own),
                    )
                )
                print(
                    json.dumps({"event": "complete", "seed": 92401 + i, "block": block}), flush=True
                )
        if len(rows) != 9:
            raise ValueError("all preregistered blocks required")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="MOTION_RECEIVER_CAMPAIGN_ALL_OWN_REPLAYED_NOT_PROMOTED"
                if a.receiver_motion_pair
                else "DENSE_RECEIVER_CAMPAIGN_ALL_OWN_REPLAYED_NOT_PROMOTED",
                source_head=a.expected_head,
                pilot_head=a.pilot_head,
                dataset_hash=a.dataset_hash,
                records=rows,
                seeds=[92401, 92402, 92403],
                blocks=[0, 1, 2],
                independent_fresh_contexts=False,
                trained_during_campaign=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="MOTION_RECEIVER_CAMPAIGN_FAILED_PRESERVE"
                if a.receiver_motion_pair
                else "DENSE_RECEIVER_CAMPAIGN_FAILED_PRESERVE",
                error_type=type(error).__name__,
                error=str(error),
                completed=rows,
                source_head=a.expected_head,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
