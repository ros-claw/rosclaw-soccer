"""One fixed research generation: verified data -> real updates -> pure policy -> replay."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "review", "parent-train", "world", "model-root", "core-source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "expected-bank-hash",
        "expected-parent-training-hash",
        "expected-review-head",
        "review-unit",
        "unit-prefix",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != a.expected_head
        or a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", a.unit_prefix) is None
    ):
        raise ValueError("frozen cycle, unused outputs and bounded unit prefix required")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    pins = {
        str(f): digest(f)
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source)
        for f in folder.rglob("*.py")
    }
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=head,
            source_hashes=pins,
            review_unit=a.review_unit,
            review=str(a.review),
            unit_prefix=a.unit_prefix,
            planned_additional_optimizer_steps=1500,
            planned_student_native_steps=240000,
            planned_replay_native_steps=240000,
            publicity_authorized=False,
            hardware_authorized=False,
        ),
    )
    wait_for_terminal_collection(a.review_unit, wait=True)
    rd = json.loads((a.review / "declaration.json").read_text())
    rc = json.loads((a.review / "complete.json").read_text())
    if (
        rd["source_head"] != a.expected_review_head
        or rc["status"] != "ALL40_MIXED_NN_AND_NATIVE_REPLAY_VERIFIED"
    ):
        raise ValueError("all40 same-state neural/native verification required")
    check_pins(pins)
    launches = []

    def launch(stage, script, args, *, gpu=False):
        unit = a.unit_prefix + "-" + stage
        folder = a.output / stage
        command = [
            "systemd-run",
            "--user",
            "--unit=" + unit,
            "--property=Type=simple",
            "--property=RemainAfterExit=yes",
            "--property=MemoryMax=10G",
            "--property=CPUQuota=180%",
            "--property=NoNewPrivileges=yes",
            "--working-directory=" + str(Path.cwd()),
            "--setenv=PYTHONPATH=src:.:" + str(a.core_source),
            "--setenv=OMP_NUM_THREADS=1",
            "--setenv=OPENBLAS_NUM_THREADS=1",
            "--setenv=MKL_NUM_THREADS=1",
            "--setenv=CUDA_VISIBLE_DEVICES=" + ("0" if gpu else ""),
            "--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8",
            sys.executable,
            "-u",
            script,
            "--expected-head",
            head,
            "--output",
            str(folder),
        ] + args
        subprocess.run(command, check=True)
        item = dict(stage=stage, unit=unit + ".service", output=str(folder), command=command)
        launches.append(item)
        write_once(a.output / f"launch-{stage}.json", item)
        wait_for_terminal_collection(item["unit"], wait=True)
        check_pins(pins)
        return item["unit"], folder, json.loads((folder / "complete.json").read_text())

    common = ["--bank", str(a.bank), "--expected-bank-hash", a.expected_bank_hash]
    train_unit, training, tc = launch(
        "train",
        "scripts/rsi_train_dagger_motor.py",
        common
        + [
            "--review",
            str(a.review),
            "--review-unit",
            a.review_unit,
            "--expected-review-head",
            a.expected_review_head,
            "--parent-train",
            str(a.parent_train),
            "--expected-parent-training-hash",
            a.expected_parent_training_hash,
        ],
        gpu=True,
    )
    if (
        tc["additional_optimizer_steps"] != 1500
        or tc["optimizer_steps"] != 4500
        or tc["training_protocol"] != "dagger4500"
        or tc["source_head"] != head
    ):
        raise ValueError("real fixed-budget DAgger training receipt required")
    eval_unit, evaluation, ec = launch(
        "evaluate",
        "scripts/rsi_eval_motor_student.py",
        common
        + [
            "--train-root",
            str(training),
            "--train-unit",
            train_unit,
            "--expected-training-hash",
            digest(training / "complete.json"),
            "--training-protocol",
            "dagger4500",
            "--world",
            str(a.world),
            "--model-root",
            str(a.model_root),
            "--core-source",
            str(a.core_source),
        ],
    )
    _, replay, proof = launch(
        "replay",
        "scripts/rsi_verify_motor_student.py",
        [
            "--mode",
            "replay",
            "--producer",
            eval_unit,
            str(evaluation),
            "--expected-producer-head",
            head,
            "--expected-producer-hash",
            digest(evaluation / "complete.json"),
            "--world",
            str(a.world),
            "--core-source",
            str(a.core_source),
        ],
    )
    if (
        proof["status"] != "ALL80_STUDENT_NN_AND_NATIVE_REPLAY_VERIFIED"
        or proof["additional_native_steps"] != ec["new_native_steps"]
    ):
        raise ValueError(
            "all80 student neural/native proof required, execution failure is not qualification"
        )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="DAGGER_GENERATION_PHYSICALLY_VERIFIED_NOT_PROMOTED",
            launches=launches,
            training_hash=digest(training / "complete.json"),
            evaluation_hash=digest(evaluation / "complete.json"),
            replay_hash=digest(replay / "complete.json"),
            initial_qualified=ec["initial_qualified"],
            trained_qualified=ec["trained_qualified"],
            trained_causal_qualified=ec["trained_causal_qualified"],
            new_optimizer_steps=1500,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
