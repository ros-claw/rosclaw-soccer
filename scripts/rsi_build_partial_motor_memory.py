"""Freeze all successful partial options from two bounded, independently audited SIM pilots."""

import argparse
import json
from pathlib import Path


def admit_memory(path, expected, pins, context, backbone_model):
    from rosclaw_soccer.rsi.partial_motor_memory import decode_motor_options
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_stable_plastic_sources import merge_pins

    if path.is_symlink() or digest(path) != expected:
        raise ValueError("exact immutable partial motor memory required")
    report = json.loads(path.read_text())
    if (
        report["status"] != "AUDITED_PARTIAL_MOTOR_OPTIONS_RECORDED_NOT_QUALIFIED"
        or report["teacher_qualified"] is not False
        or report["fresh_verified"] is not False
        or report["independently_validated_combination"] is not False
        or report["new_physical_steps"] != 0
        or report["neural_optimizer_updates"] != 0
        or report["backbone_model_hash"] != digest(backbone_model)
        or report["compiled_model_hash"] != compiled_model_hash(context["model"])
        or report["source_outcomes_are_not_runtime_inputs"] is not True
        or report["source_clock_is_not_replayed"] is not True
        or report["known_source_states_are_consumed_development"] is not True
        or any(
            report[k] is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError(
            "unqualified bounded two-option memory on the identical actual body required"
        )
    options = decode_motor_options(report["runtime_options"], context["basis"])
    merge_pins(pins, report["input_file_hashes"])
    pins[str(path)] = expected
    return options, report


def admit_source_retrieval(path, expected, pins, memory_path, memory_hash, memory_report):
    from rosclaw_soccer.rsi.partial_motor_memory import require_source_retrieval_report
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_stable_plastic_sources import merge_pins

    if path is None or path.is_symlink() or digest(path) != expected:
        raise ValueError(
            "exact completed independent source-retrieval audit before transfer required"
        )
    report = json.loads(path.read_text())
    require_source_retrieval_report(report, memory_report)
    if report["input_file_hashes"].get(str(memory_path)) != memory_hash:
        raise ValueError("transfer must use the SAME frozen motor memory, not another option bank")
    merge_pins(pins, report["input_file_hashes"])
    pins[str(path)] = expected
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for kind in ("aim", "target"):
        p.add_argument(f"--{kind}-audit", type=Path, required=True)
        p.add_argument(f"--{kind}-audit-hash", required=True)
        p.add_argument(f"--{kind}-root", type=Path, required=True)
    p.add_argument("--backbone-model", type=Path, required=True)
    p.add_argument("--backbone-model-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_stable_plastic_sources import merge_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused immutable partial motor memory output required")
    pins, options, sources = {}, [], []
    for prefix, kind, status in (
        ("aim", "aim_command", "FULL_NN_AIM_COMMAND_RAW_AND_PHYSICAL_EXAM_VERIFIED"),
        (
            "target",
            "executed_target_synergy",
            "FULL_NN_EXECUTED_TARGET_RAW_AND_PHYSICAL_EXAM_VERIFIED",
        ),
    ):
        path, expected, root = (
            getattr(a, f"{prefix}_{k}") for k in ("audit", "audit_hash", "root")
        )
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact completed independent partial motor audit required")
        audit = json.loads(path.read_text())
        if (
            audit["status"] != status
            or audit["actual_worlds"] != 12
            or audit["new_physical_steps"] != 0
            or audit["regressions"]
            or len(audit["rescues"]) != 1
            or audit["qualified_for_expansion"] is not False
            or [r["course_index"] for r in audit["results"]] != [0, 2, 8, 22, 27, 30]
            or (prefix == "target" and audit["executed_target_projection_mode"] != "per_axis")
        ):
            raise ValueError(
                "two failed-family pilots with distinct audited partial successes required"
            )
        merge_pins(pins, audit["input_file_hashes"])
        pins[str(path)] = expected
        cp = root / "complete.json"
        if pins.get(str(cp)) != digest(cp):
            raise ValueError("exact original physical pilot bound by independent audit required")
        producer = json.loads(cp.read_text())
        index = audit["rescues"][0]
        refs = [r["arms"]["C"] for r in producer["results"] if r["course_index"] == index]
        if len(refs) != 1:
            raise ValueError("one original physical record for every partial gain required")
        record = root / refs[0]["record"]
        if (
            not record.resolve().is_relative_to(root.resolve())
            or pins.get(str(record)) != refs[0]["hash"]
            or digest(record) != refs[0]["hash"]
        ):
            raise ValueError("confined bound actual successful trajectory record required")
        from rosclaw_soccer.rsi.partial_motor_memory import partial_option_from_success

        row = json.loads(record.read_text())
        options.append(partial_option_from_success(kind, row))
        sources.append(
            dict(
                kind=kind,
                audit_hash=expected,
                source_record_hash=refs[0]["hash"],
                source_course_index=index,
                actual_outcome=row["outcome"],
                all_source_results=audit["results"],
            )
        )
    if sources[0]["source_course_index"] == sources[1]["source_course_index"]:
        raise ValueError(
            "two distinct actually rescued source states required, not duplicate examples"
        )
    if (
        pins.get(str(a.backbone_model)) != a.backbone_model_hash
        or digest(a.backbone_model) != a.backbone_model_hash
    ):
        raise ValueError("same explicitly bound successful HG29 subspace required")
    source_pins(pins)
    check_pins(pins)
    from rosclaw_soccer.rsi.partial_motor_memory import decode_motor_options
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from scripts.rsi_teacher_physics_context import load_teacher_context

    context = load_teacher_context(pins, a.backbone_model)
    decode_motor_options(options, context["basis"])
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="AUDITED_PARTIAL_MOTOR_OPTIONS_RECORDED_NOT_QUALIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            backbone_model_hash=a.backbone_model_hash,
            compiled_model_hash=compiled_model_hash(context["model"]),
            runtime_options=options,
            sources=sources,
            source_outcomes_are_not_runtime_inputs=True,
            source_clock_is_not_replayed=True,
            known_source_states_are_consumed_development=True,
            independently_validated_combination=False,
            teacher_qualified=False,
            fresh_verified=False,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
