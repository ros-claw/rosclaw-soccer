"""Curate all40 verified outcomes; positive actor targets, all-outcome value data."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--expected-dispatch-hash", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != args.expected_head
        or args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("clean sealed source and unused bank required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if digest(args.dispatch / "complete.json") != args.expected_dispatch_hash:
        raise ValueError("dispatch identity differs")
    dispatch = json.loads((args.dispatch / "complete.json").read_text())
    if (
        dispatch["status"] != "ALL40_NATIVE_REVIEW_DISPATCHED_NOT_VERIFIED"
        or len(dispatch["launched_reviews"]) != 4
    ):
        raise ValueError("all40 native dispatch required, not a single good fixture")
    dispatch_declaration = json.loads((args.dispatch / "declaration.json").read_text())
    pins = dict(dispatch_declaration["input_file_hashes"])
    pins.update({str(p): digest(p) for p in args.dispatch.iterdir() if p.is_file()})
    records = []
    shards = set()
    for launch in dispatch["launched_reviews"]:
        terminal = dict(
            line.split("=", 1)
            for line in subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    launch["unit"],
                    "-p",
                    "MainPID",
                    "-p",
                    "SubState",
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                ],
                text=True,
            ).splitlines()
        )
        if terminal != dict(MainPID="0", SubState="exited", Result="success", ExecMainStatus="0"):
            raise ValueError("all native review workers must actually have completed")
        root = Path(launch["output"])
        review = json.loads((root / "complete.json").read_text())
        declaration = json.loads((root / "declaration.json").read_text())
        if (
            review["shard"] in shards
            or len(review["results"]) != 10
            or any(root.rglob("failure.json"))
            or declaration["schema"] != "soccer.rsi.expanded_bilateral_native.v1"
            or review["optimizer_updates"] != 0
            or declaration["source_head"] != dispatch_declaration["source_head"]
        ):
            raise ValueError("unique complete native ten-case review required")
        pairs = declaration["producer_complete_hashes"]
        if len(pairs) != 1:
            raise ValueError("one explicitly bound producer per native shard")
        name, expected = next(iter(pairs.items()))
        producer = Path(name).parent
        if digest(Path(name)) != expected or expected != launch["producer_complete_hash"]:
            raise ValueError("native producer binding differs")
        complete = json.loads(Path(name).read_text())
        if (
            review["additional_native_steps"]
            != complete["forecast_native_steps"] + complete["endpoint_native_steps"]
        ):
            raise ValueError("all forecasts and actual endpoints must be natively replayed")
        for row, verified in zip(complete["results"], review["results"], strict=True):
            if (
                row["course"] != verified["course"]
                or row["selected"]["score"] != verified["selected_score"]
                or verified["all_forecast_and_actual_native_fields_bit_exact"] is not True
                or verified["reference_action_mapping_and_contact_side_verified"] is not True
                or verified["prefix_safety_and_first_touch_attribution_verified"] is not True
            ):
                raise ValueError("actual outcome/target/contact review differs")
            records.append((producer, row))
        shards.add(review["shard"])
        pins.update(declaration["input_file_hashes"])
        pins.update({str(p): digest(p) for p in root.rglob("*") if p.is_file()})
    if shards != set(range(4)) or len(records) != 40:
        raise ValueError("entire40 required, no failed course drop")
    check_pins(pins)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    protocol = Path.cwd() / "docs/rsi/recurrent-motor-bc-preregister-20261020.zh-CN.md"
    pins[str(protocol)] = digest(protocol)
    import numpy as np

    from rosclaw_soccer.rsi.motor_student_bank import validation_course_indices
    from rosclaw_soccer.rsi.neural_motor_student import recorded_student_sequence
    from scripts.rsi_atomic_artifacts import write_once

    records.sort(key=lambda p: p[1]["course"]["course_index"])
    if [row["course"]["course_index"] for _, row in records] != list(range(40)):
        raise ValueError("unique ordered40 conditions required")
    # Fixed label-independent stratification:4 static,2 approaching,2 receding validation.
    validation = validation_course_indices([row["course"] for _, row in records])
    features = []
    targets = []
    eligible = []
    rows = []
    for producer, row in records:
        index = row["course"]["course_index"]
        receipt = row["selected"]
        folder = producer / f"course{index:02d}"
        path = folder / receipt["artifact"]
        if digest(path) != receipt["artifact_hash"]:
            raise ValueError("selected raw motor record identity differs")
        with np.load(path, allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        x, y = recorded_student_sequence(raw)
        actor = bool(
            receipt["score"]["qualified"]
            and row["prefix_diagnostics"]["prefix_physical_safe"]
            and row["prefix_diagnostics"]["post_boundary_control_can_cause_first_touch"]
        )
        if actor and abs(y).max() >= 8:
            raise ValueError("expert targets outside fixed student action range; no label clipping")
        features.append(x)
        targets.append(y)
        eligible.append(actor)
        rows.append(
            dict(
                course=row["course"],
                split="validation" if index in validation else "train",
                actor_eligible=actor,
                selected_score=receipt["score"],
                source_artifact=str(path),
                source_artifact_hash=receipt["artifact_hash"],
                native_reviewed=True,
                causal_first_touch_rescue=row["causal_first_touch_rescue"],
            )
        )
    train = np.array([row["split"] == "train" for row in rows], dtype=np.bool_)
    actor = np.array(eligible, dtype=np.bool_)
    if not np.any(train & actor):
        raise ValueError("no qualified training expert, cannot start imitation")
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.recurrent_motor_bank40.v1",
            source_head=head,
            input_file_hashes=pins,
            rows=rows,
            validation_course_indices=sorted(validation),
            whole_course_split=True,
            actor_targets_selected_only=True,
            negative_cases_not_dropped=True,
            broad_bilateral_teacher_promoted=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    data = args.output / "bank.npz"
    np.savez_compressed(
        data,
        features=np.stack(features),
        targets=np.stack(targets),
        train=train,
        actor_eligible=actor,
    )
    check_pins(pins)
    write_once(
        args.output / "complete.json",
        dict(
            schema="soccer.rsi.recurrent_motor_bank40.v1",
            source_head=head,
            declaration_hash=digest(args.output / "declaration.json"),
            artifact=data.name,
            artifact_hash=digest(data),
            total_courses=40,
            train_courses=int(train.sum()),
            validation_courses=int((~train).sum()),
            actor_train_courses=int((train & actor).sum()),
            actor_validation_courses=int((~train & actor).sum()),
            new_native_steps=0,
            new_neural_calls=0,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    print(
        "CURATED40_MOTOR_BANK", int((train & actor).sum()), int((~train & actor).sum()), flush=True
    )


if __name__ == "__main__":
    main()
