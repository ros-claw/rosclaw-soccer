"""Single compact source-lead experiment on the consumed fixed24, SIM only.

Before candidate physics, reproduce old6 from actual student30/case21 twice.
This is privileged teacher development, no learned-policy promotion or training.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def load_arrays(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def exact_artifact(actual, expected):
    from scripts.rsi_motor_student_rollout import exact_array

    a, b = load_arrays(actual), load_arrays(expected)
    if set(a) != set(b):
        raise ValueError("all native/neural field identities required")
    for k, v in a.items():
        exact_array(v, b[k])


def compare_old_six(actual, old, row):
    from rosclaw_soccer.rsi.source_lead_protocol import (
        ORIGINAL_NINE_ZERO_C_INDICES,
        matched_old_six_selection,
    )

    row = json.loads(json.dumps(row))
    oldrow = json.loads((old / "complete.json").read_text())
    match = matched_old_six_selection(oldrow)
    if (
        row["option_protocol"] != "zero_c_six"
        or row["source_window"] != [286, 767]
        or row["selected"] != match["six_index"]
        or len(row["candidates"]) != 6
        or row["new_native_steps"] != 19200
        or row["forecast_native_steps"] != 16200
        or row["executed_native_steps"] != 3000
        or row["selected_score"] != oldrow["selected_score"]
        or row["causal_qualified"] != oldrow["causal_qualified"]
        or row["selected_report"] != oldrow["candidates"][match["original_nine_index"]]["report"]
    ):
        raise ValueError("matched zero-C6 actual30 baseline/choice/count/report differs")
    for new, index in enumerate(ORIGINAL_NINE_ZERO_C_INDICES):
        exact_artifact(actual / f"candidate{new}.npz", old / f"candidate{index}.npz")
        keys = set(row["candidates"][new]) - {"index", "artifact", "artifact_hash"}
        a = {k: row["candidates"][new][k] for k in keys}
        b = {k: oldrow["candidates"][index][k] for k in keys}
        if json.loads(json.dumps(a)) != b:
            raise ValueError("all matched future primitive metadata required")
    for name in (
        "actual-student30.npz",
        "shadow-labels30.npz",
        "selected-actual270.npz",
        "selected-physical300.npz",
    ):
        exact_artifact(actual / name, old / name)


def compare_six_case(actual, expected, row):
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != "artifact_hash"}
        return [strip(v) for v in value] if isinstance(value, (list, tuple)) else value

    if strip(json.loads(json.dumps(row))) != strip(
        json.loads((expected / "complete.json").read_text())
    ):
        raise ValueError("all six-option repeat choice/report/score/accounting required")
    names = ["actual-student30.npz", "shadow-labels30.npz"]
    if row["status"] == "STUDENT30_PRIVILEGED_RECOVERY_EXECUTED":
        names += [f"candidate{i}.npz" for i in range(6)]
        names += ["selected-actual270.npz", "selected-physical300.npz"]
    for name in names:
        exact_artifact(actual / name, expected / name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("declare", "zero", "probe", "reproduce"), required=True)
    for k in (
        "inputs",
        "core-source",
        "old-review",
        "old-launch",
        "warm",
        "warm-repeat",
        "near-plan",
        "motion-csv",
        "default-parity",
        "output",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "expected-head",
        "old-review-unit",
        "expected-old-review-hash",
        "default-parity-unit",
        "expected-default-source",
    ):
        p.add_argument("--" + k, required=True)
    p.add_argument("--plan", type=Path)
    p.add_argument("--expected-plan-hash")
    p.add_argument("--zero", type=Path)
    p.add_argument("--zero-unit")
    p.add_argument("--part", type=int, choices=range(8))
    p.add_argument("--producer", type=Path)
    p.add_argument("--producer-unit")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_compact_source_common import admit, load_student, source_preflight
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, near, pins, runtime, courses, indices, old_cases, baseline, prior_path = admit(a)
    world, poses, prior, source_qualification = source_preflight(a, prior_path)
    check_pins(pins)
    protocol = dict(
        source_head=a.expected_head,
        input_file_hashes=pins,
        indices=list(indices),
        cases=[asdict(courses[i]) for i in indices],
        matched_old_baseline=baseline,
        source_qualification=source_qualification,
        source_student_seed=43,
        original_source_window=[286, 767],
        candidate_source_window=[466, 767],
        source_frequency_hz=120,
        entry_intervals=50,
        options=6,
        zero_c_only=True,
        original_task_horizon=300,
        original_score_frames=[20, 299],
        maximum_zero_native_steps=38400,
        maximum_candidate_paired_native_steps=921600,
        old_actual_baseline_reused_not_new_physics=True,
        optimizer_steps=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if a.mode == "declare":
        if a.plan or a.zero or a.part is not None or a.producer:
            raise ValueError("whole24 no-physics declaration required")
        a.output.mkdir(parents=True)
        write_once(
            a.output / "complete.json",
            dict(status="COMPACT_SOURCE_LEAD24_DECLARED", **protocol, new_native_steps=0),
        )
        return
    if (
        not a.plan
        or not a.expected_plan_hash
        or digest(a.plan / "complete.json") != a.expected_plan_hash
    ):
        raise ValueError("exact frozen whole24 declaration required")
    plan = json.loads((a.plan / "complete.json").read_text())
    if plan != json.loads(
        json.dumps(dict(status="COMPACT_SOURCE_LEAD24_DECLARED", **protocol, new_native_steps=0))
    ):
        raise ValueError("entire input/source/protocol commitment differs")
    pins[str(a.plan / "complete.json")] = a.expected_plan_hash
    if a.mode == "zero":
        if a.zero or a.part is not None or a.producer:
            raise ValueError("zero qualification is fixed original case21 only")
        selected = [21]
    else:
        if not a.zero or not a.zero_unit or a.part is None:
            raise ValueError("closed actual30 old-six parity prerequisite required")
        wait_for_terminal_collection(a.zero_unit, wait=True)
        z = json.loads((a.zero / "complete.json").read_text())
        if (
            z["status"] != "ORIGINAL_SOURCE_SIX_ACTUAL30_NUMERIC_PARITY"
            or z["source_head"] != a.expected_head
            or z["new_native_steps"] != 38400
            or z["all18_six_futures_and_actual_byte_exact_old"] is not True
        ):
            raise ValueError("actual30 native/NN old-six qualification must pass")
        pins[str(a.zero / "complete.json")] = digest(a.zero / "complete.json")
        selected = list(indices[a.part :: 8])
    if a.mode == "reproduce":
        if not a.producer or not a.producer_unit:
            raise ValueError("explicit actual producer/unit required")
        wait_for_terminal_collection(a.producer_unit, wait=True)
        c = json.loads((a.producer / "complete.json").read_text())
        if (
            c["status"] != "COMPACT_SOURCE_LEAD_SHARD_PRODUCED"
            or c["source_head"] != a.expected_head
            or c["indices"] != selected
        ):
            raise ValueError("matching fixed-three actual terminal0 producer required")
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
    elif a.producer or a.producer_unit:
        raise ValueError("producer root is only legal in reproduction mode")
    check_pins(pins)
    a.output.mkdir(parents=True)
    from rosclaw_soccer.sim.contracts import hash_json

    declaration = dict(
        source_head=a.expected_head,
        input_file_hashes=pins,
        runtime_file_hashes=runtime,
        indices=selected,
        part=a.part,
        mode=a.mode,
        source_window=[286, 767] if a.mode == "zero" else [466, 767],
        options=6,
        maximum_native_steps=38400 if a.mode == "zero" else 19200 * len(selected),
        optimizer_steps=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    evidence = hash_json(declaration)
    if a.mode == "reproduce":
        evidence = json.loads((a.producer / "declaration.json").read_text())[
            "teacher_evidence_hash"
        ]
    declaration["teacher_evidence_hash"] = evidence
    write_once(a.output / "declaration.json", declaration)
    import numpy as np

    from scripts.rsi_student_intervention_course import run_course

    student, muscle = load_student(spec, near, a.world)
    rows = []
    try:
        for i in selected:
            paths = [Path(k) for k in pins if k.endswith(f"/course{i:03d}/seed43/native.npz")]
            if len(paths) != 1:
                raise ValueError("one original seed43 neural/native trace required")
            with np.load(paths[0], allow_pickle=False) as z:
                expected = {k: z[k].copy() for k in z.files}
            if a.mode == "zero":
                old_evidence = json.loads((old_cases[i].parent / "declaration.json").read_text())[
                    "teacher_evidence_hash"
                ]
                trials = []
                for name in ("first", "second"):
                    trial = a.output / name
                    trial.mkdir()
                    row = run_course(
                        SimpleNamespace(**{**vars(a), "output": trial}),
                        courses[i],
                        world,
                        poses,
                        prior,
                        student,
                        muscle,
                        expected,
                        old_evidence,
                        runtime,
                        option_protocol="zero_c_six",
                    )
                    compare_old_six(trial / f"course{i:03d}", old_cases[i], row)
                    trials.append(row)
                compare_six_case(
                    a.output / "second/course021", a.output / "first/course021", trials[1]
                )
                rows.append(
                    dict(
                        course_index=i, new_native_steps=sum(r["new_native_steps"] for r in trials)
                    )
                )
            else:
                row = run_course(
                    a,
                    courses[i],
                    world,
                    poses,
                    prior,
                    student,
                    muscle,
                    expected,
                    evidence,
                    runtime,
                    source_window=(466, 767),
                    option_protocol="zero_c_six",
                )
                if a.producer:
                    compare_six_case(
                        a.output / f"course{i:03d}", a.producer / f"course{i:03d}", row
                    )
                rows.append(row)
            write_once(a.output / f"progress{i:03d}.json", rows[-1])
        check_pins(pins)
        result = dict(
            status="ORIGINAL_SOURCE_SIX_ACTUAL30_NUMERIC_PARITY"
            if a.mode == "zero"
            else "COMPACT_SOURCE_LEAD_SHARD_REPRODUCED"
            if a.mode == "reproduce"
            else "COMPACT_SOURCE_LEAD_SHARD_PRODUCED",
            source_head=a.expected_head,
            indices=selected,
            results=rows,
            new_native_steps=sum(r["new_native_steps"] for r in rows),
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        if a.mode == "zero":
            result["all18_six_futures_and_actual_byte_exact_old"] = True
        write_once(a.output / "complete.json", result)
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                completed_cases=len(rows),
                partial_charge_requires_case_receipts=True,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
