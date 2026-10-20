"""Bounded privileged nine-option teacher coverage, not a deployed actor exam.

First reproduce original nominal teacher0 numerically under today's interfaces.
Then probe24 fixed ranks of the consumed near96 and repeat ALL NN/futures/native
in new processes. This is same-implementation reproduction, not independent math.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("zero", "probe", "reproduce"), required=True)
    for k in (
        "inputs",
        "core-source",
        "near-review",
        "near-plan",
        "original-teacher",
        "motion-csv",
        "output",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "expected-head",
        "expected-inputs-hash",
        "expected-near-review-hash",
        "near-review-unit",
        "expected-original-teacher-hash",
    ):
        p.add_argument("--" + k, required=True)
    p.add_argument("--zero", type=Path)
    p.add_argument("--zero-unit")
    p.add_argument("--producer", type=Path)
    p.add_argument("--producer-unit")
    p.add_argument("--part", type=int, choices=range(8))
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, _, pins, _ = admit(a)
    wait_for_terminal_collection(a.near_review_unit)
    if (
        digest(a.near_review / "complete.json") != a.expected_near_review_hash
        or digest(a.original_teacher / "complete.json") != a.expected_original_teacher_hash
    ):
        raise ValueError("exact closed generalized failure and original teacher required")
    review = json.loads((a.near_review / "complete.json").read_text())
    if review["status"] != "CATEGORICAL_NEAR96_FOUR_ARM_PRIMITIVES_AUTHENTICATED" or any(
        c["necessary_generalization_evidence"] for c in review["summary"]["comparisons"].values()
    ):
        raise ValueError("this failure diagnostic requires completed unsuccessful near96 exam")
    for k, h in review["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting past experiment lineage")
        pins[k] = h
    pins[str(a.near_review / "complete.json")] = a.expected_near_review_hash
    old = json.loads((a.original_teacher / "declaration.json").read_text())
    old_pins = old["runtime_file_hashes"]
    a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
    assets = [a.world, a.motion_csv] + [
        a.model_root / s
        for s in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]
    for f in assets:
        if old_pins.get(str(f)) != digest(f):
            raise ValueError("teacher world/motion/NN assets changed")
        pins[str(f)] = digest(f)
    pins[str(a.original_teacher / "declaration.json")] = digest(
        a.original_teacher / "declaration.json"
    )
    pins[str(a.original_teacher / "complete.json")] = a.expected_original_teacher_hash
    from rosclaw_soccer.rsi.teacher_coverage_cohort import teacher_coverage_indices
    from rosclaw_soccer.training.categorical_motion_holdout import (
        categorical_motion_holdout_courses,
    )
    from rosclaw_soccer.training.expanded_entry_courses import expanded_entry_courses
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = categorical_motion_holdout_courses()
    bank_hash = preflight_receiving_courses(courses)
    near = json.loads((a.near_plan / "complete.json").read_text())
    if (
        near["preflight_hash"] != bank_hash
        or digest(a.near_plan / "courses.json") != near["courses_hash"]
    ):
        raise ValueError("original consumed near96 population differs")
    for f in (a.near_plan / "complete.json", a.near_plan / "courses.json"):
        pins[str(f)] = digest(f)
    if a.mode == "zero":
        if a.part is not None or a.zero or a.producer:
            raise ValueError("zero admission is original nominal0 only")
        original = expanded_entry_courses()
        preflight_receiving_courses(original)
        items = [original[0]]
    else:
        if a.part is None or not a.zero or not a.zero_unit:
            raise ValueError("complete zero-teacher parity prerequisite required")
        wait_for_terminal_collection(a.zero_unit, wait=True)
        z = json.loads((a.zero / "complete.json").read_text())
        if (
            z["status"] != "NOMINAL_ZERO_TEACHER_NEURAL_NATIVE_REPRODUCED"
            or z["source_head"] != a.expected_head
            or z["all18_array_fields_byte_exact_to_original"] is not True
        ):
            raise ValueError("zero parity must really have succeeded")
        pins[str(a.zero / "complete.json")] = digest(a.zero / "complete.json")
        items = [courses[i] for i in teacher_coverage_indices()[a.part :: 8]]
    producer_d = producer_c = None
    if a.mode == "reproduce":
        if not a.producer or not a.producer_unit:
            raise ValueError("explicit actual terminal producer required")
        wait_for_terminal_collection(a.producer_unit, wait=True)
        producer_d = json.loads((a.producer / "declaration.json").read_text())
        producer_c = json.loads((a.producer / "complete.json").read_text())
        if (
            producer_c["status"] != "PRIVILEGED_TEACHER_COVERAGE_PRODUCED"
            or producer_d["source_head"] != a.expected_head
            or producer_c["source_head"] != a.expected_head
            or producer_d["indices"] != [i.course_index for i in items]
        ):
            raise ValueError("same fixed three-course producer shard required")
        for f in a.producer.rglob("*"):
            if f.is_file():
                pins[str(f)] = digest(f)
    check_pins(pins)
    runtime = {
        str(f): digest(f)
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source)
        for f in folder.rglob("*.py")
    }
    runtime.update({str(f): digest(f) for f in assets})
    prior_paths = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(prior_paths) != 1:
        raise ValueError("one historical entry prior required")
    runtime[str(prior_paths[0])] = digest(prior_paths[0])
    pins.update(runtime)
    a.output.mkdir(parents=True)
    decl = dict(
        source_head=a.expected_head,
        input_file_hashes=pins,
        runtime_file_hashes=runtime,
        mode=a.mode,
        part=a.part,
        indices=[i.course_index for i in items],
        cases=[asdict(i) for i in items],
        population="original40_nominal0"
        if a.mode == "zero"
        else "consumed_near96_fixed24_diagnostic",
        near_review_hash=a.expected_near_review_hash,
        candidates_per_case=9,
        maximum_native_steps=62000 if a.mode == "zero" else 31000 * len(items),
        new_optimizer_steps=0,
        student_actor_executed=False,
        privileged_teacher_uses_bounded_futures=True,
        original_far_or_private_fresh=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    from rosclaw_soccer.sim.contracts import hash_json

    evidence = producer_d["teacher_evidence_hash"] if producer_d else hash_json(decl)
    decl["teacher_evidence_hash"] = evidence
    write_once(a.output / "declaration.json", decl)
    progress = []
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
        from scripts.rsi_expanded_bilateral_course import run_course

        model = mujoco.MjModel.from_binary_path(str(a.world))
        poses = np.loadtxt(a.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        stored = json.loads(prior_paths[0].read_text())
        affordance = SkillEntryAffordance(
            tuple(stored["ball_offset_xy_m"]),
            stored["historical_frame"],
            stored["historical_forward_speed_mps"],
            stored["historical_artifact_hash"],
        )
        if affordance.contract_hash != stored["contract_hash"]:
            raise ValueError("historical prior contract changed")
        for item in items:
            if a.mode == "zero":
                first_root, second_root = a.output / "first", a.output / "second"
                first_root.mkdir()
                second_root.mkdir()
                first = run_course(
                    SimpleNamespace(world=a.world, model_root=a.model_root, output=first_root),
                    item,
                    model,
                    poses,
                    affordance,
                    evidence,
                    runtime,
                )
                compare_case(
                    first_root / f"course{item.course_index:02d}",
                    a.original_teacher / f"course{item.course_index:02d}",
                    first,
                    original=True,
                )
                second = run_course(
                    SimpleNamespace(world=a.world, model_root=a.model_root, output=second_root),
                    item,
                    model,
                    poses,
                    affordance,
                    evidence,
                    runtime,
                )
                compare_case(
                    second_root / f"course{item.course_index:02d}",
                    first_root / f"course{item.course_index:02d}",
                    second,
                )
                charged = 2 * (first["forecast_native_steps"] + first["endpoint_native_steps"])
                progress.append(dict(course=asdict(item), new_native_steps=charged))
            else:
                row = run_course(a, item, model, poses, affordance, evidence, runtime)
                if a.mode == "reproduce":
                    compare_case(
                        a.output / f"course{item.course_index:02d}",
                        a.producer / f"course{item.course_index:02d}",
                        row,
                    )
                progress.append(row)
            write_once(a.output / f"progress{item.course_index:03d}.json", progress[-1])
        check_pins(pins)
        if a.mode == "zero":
            result = dict(
                status="NOMINAL_ZERO_TEACHER_NEURAL_NATIVE_REPRODUCED",
                new_native_steps=sum(r["new_native_steps"] for r in progress),
                all18_array_fields_byte_exact_to_original=True,
            )
        else:
            result = dict(
                status="PRIVILEGED_TEACHER_COVERAGE_REPRODUCED"
                if a.mode == "reproduce"
                else "PRIVILEGED_TEACHER_COVERAGE_PRODUCED",
                results=progress,
                new_native_steps=sum(
                    r["forecast_native_steps"] + r["endpoint_native_steps"] for r in progress
                ),
                same_implementation_neural_and_physics_reexecuted=a.mode == "reproduce",
            )
        write_once(
            a.output / "complete.json",
            dict(
                source_head=a.expected_head,
                **result,
                optimizer_steps=0,
                original_far_or_private_fresh=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                completed_courses=len(progress),
                partial_native_charge_requires_case_receipts=True,
                automatic_retry_authorized=False,
            ),
        )
        raise


def compare_case(actual, expected, row, *, original=False):
    import numpy as np

    from scripts.rsi_motor_student_rollout import exact_array

    old = json.loads((expected / "complete.json").read_text())
    # Producer JSON turns tuples into lists; normalize the in-memory row.
    if (
        json.loads(json.dumps(row["course"])) != old["course"]
        or row["selected"]["selected_index"] != old["selected"]["selected_index"]
    ):
        raise ValueError("teacher case/selection identity changed")
    paths = (
        ["selected-actual.npz"]
        if original
        else ["own-prefix20.npz"]
        + [f"candidate{i}.npz" for i in range(9)]
        + ["selected-actual.npz"]
    )
    if not original and not row["shared_endpoint"]:
        paths.append("baseline-actual.npz")
    for name in paths:
        with (
            np.load(actual / name, allow_pickle=False) as a,
            np.load(expected / name, allow_pickle=False) as b,
        ):
            if a.files != b.files:
                raise ValueError("all recorded numerical field identities required")
            for k in a.files:
                exact_array(a[k], b[k])
    if (
        row["selected"]["score"] != old["selected"]["score"]
        or row["forecast_native_steps"] != old["forecast_native_steps"]
        or row["endpoint_native_steps"] != old["endpoint_native_steps"]
    ):
        raise ValueError("teacher task outcome/native accounting differs")
    if not original:

        def strip(value):
            if isinstance(value, dict):
                return {k: strip(v) for k, v in value.items() if k != "artifact_hash"}
            return [strip(v) for v in value] if isinstance(value, (list, tuple)) else value

        if strip(json.loads(json.dumps(row))) != strip(old):
            raise ValueError("teacher predictions/report/contact/selection differ on reproduction")


if __name__ == "__main__":
    main()
