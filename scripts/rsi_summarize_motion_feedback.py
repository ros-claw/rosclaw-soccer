"""Read-only all-three-seed verified feedback for the next actual learning stage."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pair",
        nargs=4,
        action="append",
        required=True,
        metavar=("PRODUCER", "REPLAY", "PRODUCER_HASH", "REPLAY_HASH"),
    )
    for key in ("expected-head", "expected-experiment-head", "expected-bank-hash"):
        p.add_argument("--" + key, required=True)
    for key in ("bank", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != a.expected_head
        or a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("clean frozen feedback source and unused output required")
    from collections import Counter

    from rosclaw_soccer.rsi.motion_failure_feedback import first_touch_failure_causes
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if digest(a.bank / "complete.json") != a.expected_bank_hash or len(a.pair) != 3:
        raise ValueError("entire declared3-seed bank required")
    bd = json.loads((a.bank / "declaration.json").read_text())
    pins = dict(bd["input_file_hashes"])
    pins.update({str(f): digest(f) for f in a.bank.iterdir() if f.is_file()})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    results = []
    for producer, replay, ph, rh in a.pair:
        producer, replay = Path(producer), Path(replay)
        if digest(producer / "complete.json") != ph or digest(replay / "complete.json") != rh:
            raise ValueError("actual completed physics/replay hashes differ")
        d = json.loads((producer / "declaration.json").read_text())
        c = json.loads((producer / "complete.json").read_text())
        rd = json.loads((replay / "declaration.json").read_text())
        rc = json.loads((replay / "complete.json").read_text())
        if (
            d["source_head"] != a.expected_experiment_head
            or rd["source_head"] != a.expected_experiment_head
            or rd["producer_hash"] != ph
            or d["bank_hash"] != a.expected_bank_hash
            or rc["status"] != "ALL80_MOTION_INTENTION_MUSCLE_NATIVE_VERIFIED"
            or c["new_native_steps"] != 240000
            or rc["new_native_steps"] != 240000
            or len(c["results"]) != 80
            or c["execution_failed"] != 0
        ):
            raise ValueError("whole80 independent neural/muscle/native verification required")
        pins.update(rd["input_file_hashes"])
        for root in (producer, replay):
            pins.update({str(f): digest(f) for f in root.rglob("*.json") if f.is_file()})
        summaries = {}
        for variant in ("initial", "trained"):
            rows = [x for x in c["results"] if x["variant"] == variant]
            if [x["course"] for x in rows] != [x["course"] for x in bd["rows"]]:
                raise ValueError("whole40 original ordered course identities required")
            failures = Counter()
            successful = []
            for row in rows:
                failures.update(first_touch_failure_causes(row["score"]))
                if row["score"]["qualified"]:
                    index = row["course"]["course_index"]
                    successful.append(
                        dict(
                            course_index=index,
                            split=row["split"],
                            causal=row["causal_student_first_touch"],
                            teacher_qualified=bd["rows"][index]["selected_score"]["qualified"],
                            score=row["score"],
                        )
                    )
            summaries[variant] = dict(
                total=40,
                qualified=len(successful),
                successful=successful,
                body_safe=sum(r["score"]["physical_prefix_and_suffix_safe"] for r in rows),
                foot_touch=sum(
                    r["score"]["first_foot_contact_global_frame"] is not None for r in rows
                ),
                failure_cause_counts=dict(failures),
                cause_counts_overlap_not_episode_count=True,
            )
        results.append(dict(seed=c["seed"], summaries=summaries, producer_hash=ph, replay_hash=rh))
    if {r["seed"] for r in results} != {17, 29, 43}:
        raise ValueError("allthreeunique fixed seeds required")
    check_pins(pins)
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(source_head=head, input_file_hashes=pins, bank_hash=a.expected_bank_hash),
    )
    report = dict(
        results=results,
        new_native_steps=0,
        optimizer_steps=0,
        all_labels_independently_neural_native_verified=True,
        near_first_touch_only=True,
        task_gate_relaxed=False,
        goal_complete=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(a.output / "complete.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
