"""Wait for reviewed native pilot, then export, read back and train once in SIM."""

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
    for name in (
        "exploration",
        "review",
        "parent",
        "parent-audit",
        "parent-dataset",
        "anchor-admission",
        "anchor-dataset",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "review-unit",
        "review-head",
        "exploration-head",
        "parent-hash",
        "parent-audit-hash",
        "parent-dataset-hash",
        "anchor-admission-hash",
        "anchor-dataset-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    pins = (
        (a.parent, a.parent_hash),
        (a.parent_audit, a.parent_audit_hash),
        (a.parent_dataset, a.parent_dataset_hash),
        (a.anchor_admission, a.anchor_admission_hash),
        (a.anchor_dataset, a.anchor_dataset_hash),
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 7200
        or not a.review_unit.startswith("rosclaw-receiver-learned-exploration-review-")
        or not a.review_unit.endswith(".service")
        or "/" in a.review_unit
        or any(p.is_symlink() or digest(p) != h for p, h in pins)
    ):
        raise ValueError("exact research inputs, review service and unused output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.review_unit,
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
                raise TimeoutError("review observation expired; do not restart physics")
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
            raise ValueError("pilot review did not finish successfully")
        break
    review_path = a.review / "complete.json"
    if review_path.is_symlink():
        raise ValueError("owned review required")
    review_hash, review = digest(review_path), json.loads(review_path.read_text())
    if (
        review.get("status") != "LEARNED_RECEIVER_EXPLORATION_PILOT_REVIEWED"
        or review.get("source_head") != a.review_head
        or review.get("receiver_checkpoint_hash") != a.parent_hash
        or any(
            review.get(k) is not False
            for k in ("training_performed", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("exact independently replayed review and behavior parent required")
    a.output.mkdir(parents=True)
    dataset, audit, training = a.output / "data", a.output / "data-audit", a.output / "training"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_export_learned_receiver_pilot",
            "--exploration",
            str(a.exploration),
            "--exploration-head",
            a.exploration_head,
            "--review",
            str(review_path),
            "--review-hash",
            review_hash,
            "--expected-head",
            a.expected_head,
            "--output",
            str(dataset),
        ],
        check=True,
    )
    dataset_hash = digest(dataset / "complete.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_audit_learned_receiver_pilot",
            "--exploration",
            str(a.exploration),
            "--dataset",
            str(dataset),
            "--dataset-hash",
            dataset_hash,
            "--review",
            str(review_path),
            "--review-hash",
            review_hash,
            "--expected-head",
            a.expected_head,
            "--output",
            str(audit),
        ],
        check=True,
    )
    audit_hash = digest(audit / "complete.json")
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_train_learned_receiver_pilot",
        "--dataset",
        str(dataset),
        "--dataset-hash",
        dataset_hash,
        "--dataset-audit",
        str(audit / "complete.json"),
        "--dataset-audit-hash",
        audit_hash,
        "--expected-head",
        a.expected_head,
        "--output",
        str(training),
    ]
    for name in ("parent", "parent-audit", "parent-dataset", "anchor-admission", "anchor-dataset"):
        attr = name.replace("-", "_")
        command.extend(
            ("--" + name, str(getattr(a, attr)), "--" + name + "-hash", getattr(a, attr + "_hash"))
        )
    subprocess.run(command, check=True)
    training_path = training / "complete.json"
    training_hash, result = digest(training_path), json.loads(training_path.read_text())
    if (
        result.get("status")
        not in (
            "LEARNED_RECEIVER_PILOT_CONTINUATION_CANDIDATE",
            "LEARNED_RECEIVER_PILOT_CONTINUATION_REJECTED",
        )
        or result.get("dataset_hash") != dataset_hash
        or result.get("dataset_audit_hash") != audit_hash
        or digest(review_path) != review_hash
        or any(digest(p) != h for p, h in pins)
    ):
        raise ValueError("completed continuation and unchanged original evidence required")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="LEARNED_RECEIVER_PILOT_CONTINUATION_NOT_EVALUATED",
            source_head=a.expected_head,
            review_hash=review_hash,
            dataset_hash=dataset_hash,
            dataset_audit_hash=audit_hash,
            training_hash=training_hash,
            actor_update_accepted=result["accepted"],
            fresh_holdout=False,
            actual_new_physics=False,
            online_rl=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
