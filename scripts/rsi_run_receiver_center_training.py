"""After strict data audit, train all declared seeds or record data insufficiency."""

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
    for name in ("dataset", "audit", "anchor-admission", "anchor-dataset", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "audit-unit",
        "audit-head",
        "anchor-admission-hash",
        "anchor-dataset-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not a.audit_unit.startswith("rosclaw-receiver-center-data-audit-")
        or not a.audit_unit.endswith(".service")
        or "/" in a.audit_unit
        or not 1 <= a.timeout_sec <= 43200
    ):
        raise ValueError("bounded audited data job and unused output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.audit_unit,
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
                raise TimeoutError("audit observation expired; do not restart live audit")
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
            raise ValueError("strict physical data audit failed")
        break
    dp = a.dataset / "complete.json"
    if dp.is_symlink() or a.audit.is_symlink():
        raise ValueError("owned dataset and audit required")
    dh, ah = digest(dp), digest(a.audit)
    data, audit = json.loads(dp.read_text()), json.loads(a.audit.read_text())
    quality = data.get("research_data_sufficiency")
    if (
        audit.get("status") != "RECEIVER_ONPOLICY_READBACK_AND_DENSITY_NOT_TRAINED"
        or audit.get("source_head") != a.audit_head
        or audit.get("dataset_hash") != dh
        or audit.get("reward_profile") != "center-strict-full-history-v1"
        or audit.get("research_data_sufficiency") != quality
        or audit.get("exact_inputs_actions_rewards_and_source_replayed") is not True
        or any(
            audit.get(k) is not False
            for k in ("optimizer_authorized", "promotion_authorized", "hardware_authorized")
        )
        or not isinstance(quality, dict)
        or type(quality.get("research_training_data_sufficient")) is not bool
    ):
        raise ValueError("exact audited strict reward dataset required")
    a.output.mkdir(parents=True)
    if not quality["research_training_data_sufficient"]:
        write_once(
            a.output / "complete.json",
            dict(
                status="CENTER_RECEIVER_INSUFFICIENT_NO_TRAINING",
                source_head=a.expected_head,
                dataset_hash=dh,
                audit_hash=ah,
                research_data_sufficiency=quality,
                critic_updates=0,
                actor_updates=0,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        return
    results = []
    for seed in (92401, 92402, 92403):
        out = a.output / f"seed-{seed}"
        command = [
            sys.executable,
            "-m",
            "scripts.rsi_train_receiver_center_strict",
            "--dataset",
            str(a.dataset),
            "--dataset-hash",
            dh,
            "--audit",
            str(a.audit),
            "--audit-hash",
            ah,
            "--audit-head",
            a.audit_head,
            "--anchor-admission",
            str(a.anchor_admission),
            "--anchor-admission-hash",
            a.anchor_admission_hash,
            "--anchor-dataset",
            str(a.anchor_dataset),
            "--anchor-dataset-hash",
            a.anchor_dataset_hash,
            "--expected-head",
            a.expected_head,
            "--seed",
            str(seed),
            "--output",
            str(out),
        ]
        subprocess.run(command, check=True)
        results.append(
            dict(
                seed=seed,
                receipt=str(out / "complete.json"),
                receipt_hash=digest(out / "complete.json"),
            )
        )
    if digest(dp) != dh or digest(a.audit) != ah:
        raise ValueError("data or audit changed while training")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CENTER_RECEIVER_THREE_SEED_OFFLINE_RESEARCH",
            source_head=a.expected_head,
            dataset_hash=dh,
            audit_hash=ah,
            research_data_sufficiency=quality,
            results=results,
            physical_improvement_proven=False,
            online_rl=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
