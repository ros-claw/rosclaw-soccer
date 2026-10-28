"""Audit all three receiver candidates, or retain explicit no-training decision."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "data-audit", "admission", "training", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("training-unit", "producer-head", "admission-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not a.training_unit.startswith("rosclaw-receiver-center-three-seed-training-")
        or not a.training_unit.endswith(".service")
        or "/" in a.training_unit
        or not 1 <= a.timeout_sec <= 43200
    ):
        raise ValueError("bounded training unit and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.training_unit,
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
                raise TimeoutError("training observation expired; do not restart live work")
            time.sleep(15)
            continue
        if state.get("LoadState") == "not-found" and state.get("ActiveState") == "inactive":
            break
        if not (
            state.get("LoadState") == "loaded"
            and state.get("ActiveState") == "inactive"
            and state.get("Result") == "success"
            and state.get("ExecMainStatus") == "0"
        ):
            raise ValueError("training job failed")
        break
    dp, tp, ap = (
        a.dataset / "complete.json",
        a.training / "complete.json",
        a.admission / "complete.json",
    )
    if any(x.is_symlink() for x in (dp, tp, ap, a.data_audit)):
        raise ValueError("owned evidence receipts required")
    dh, th, ah = digest(dp), digest(tp), digest(a.data_audit)
    if digest(ap) != a.admission_hash:
        raise ValueError("original anchor admission differs")
    result = json.loads(tp.read_text())
    if (
        result.get("source_head") != a.producer_head
        or result.get("dataset_hash") != dh
        or result.get("audit_hash") != ah
        or any(
            result.get(k) is not False
            for k in ("promotion_authorized", "hardware_authorized", "goal_complete")
        )
    ):
        raise ValueError("exact unpromoted training decision required")
    a.output.mkdir(parents=True)
    if result.get("status") == "CENTER_RECEIVER_INSUFFICIENT_NO_TRAINING":
        if result.get("actor_updates") != 0 or result.get("critic_updates") != 0:
            raise ValueError("insufficient data cannot update optimizer")
        write_once(
            a.output / "complete.json",
            dict(
                status="CENTER_RECEIVER_NO_CANDIDATE_TO_AUDIT",
                source_head=a.expected_head,
                training_hash=th,
                dataset_hash=dh,
                data_audit_hash=ah,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        return
    if result.get("status") != "CENTER_RECEIVER_THREE_SEED_OFFLINE_RESEARCH" or [
        r.get("seed") for r in result.get("results", [])
    ] != [92401, 92402, 92403]:
        raise ValueError("complete ordered three-seed experiment required")
    checks = []
    for entry in result["results"]:
        seed = entry["seed"]
        root = a.training / f"seed-{seed}"
        receipt = root / "complete.json"
        if (
            receipt.is_symlink()
            or digest(receipt) != entry["receipt_hash"]
            or str(receipt) != entry["receipt"]
        ):
            raise ValueError("exact seed receipt required")
        trial = json.loads(receipt.read_text())
        if trial.get("status") == "CENTER_RECEIVER_OFFLINE_REJECTED":
            if trial.get("actor_updates") != 0:
                raise ValueError("rejected actor must have zero update")
            checks.append(
                dict(seed=seed, status="REJECTED_NO_POLICY", training_hash=entry["receipt_hash"])
            )
            continue
        out = a.output / f"seed-{seed}"
        command = [
            sys.executable,
            "-m",
            "scripts.rsi_audit_receiver_candidate",
            "--receiver-center-strict",
            "--dataset",
            str(a.dataset),
            "--dataset-hash",
            dh,
            "--data-audit",
            str(a.data_audit),
            "--data-audit-hash",
            ah,
            "--admission",
            str(a.admission),
            "--admission-hash",
            a.admission_hash,
            "--training",
            str(root),
            "--training-hash",
            entry["receipt_hash"],
            "--producer-head",
            a.producer_head,
            "--seed",
            str(seed),
            "--expected-head",
            a.expected_head,
            "--output",
            str(out),
        ]
        subprocess.run(command, check=True)
        checks.append(
            dict(seed=seed, status="CANDIDATE_READBACK", audit_hash=digest(out / "complete.json"))
        )
    if (
        digest(dp) != dh
        or digest(tp) != th
        or digest(a.data_audit) != ah
        or digest(ap) != a.admission_hash
    ):
        raise ValueError("evidence changed during candidate readback")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CENTER_RECEIVER_THREE_SEED_CANDIDATES_AUDITED",
            source_head=a.expected_head,
            training_hash=th,
            dataset_hash=dh,
            data_audit_hash=ah,
            records=checks,
            independent_new_physics=False,
            physical_retention_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
