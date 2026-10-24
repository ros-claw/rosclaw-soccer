"""Retain current skills and learn only audited rescues, never fit consumed Fresh."""

from typing import Any

from rosclaw_soccer.rsi.pass_teacher_coverage import teacher_coverage
from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label


def refinement_curriculum(
    coverage: dict[str, Any],
    native_records: list[dict[str, Any]],
    old_records: list[dict[str, Any]],
) -> dict[str, Any]:
    measured = teacher_coverage(
        coverage["baseline"], coverage["selected_sources"], coverage["transfers"]
    )
    if (
        measured != coverage["coverage"]
        or measured["functional_correction_teacher_qualified"] is not True
    ):
        raise ValueError("whole independently audited functional teacher coverage required")
    if len(native_records) != 80 or len(old_records) != 274:
        raise ValueError("whole current80 and original274 TRAIN histories required")
    by_key = {}
    for index, r in enumerate(native_records):
        c = r["course"]
        key = (c["state_id"], c["task"])
        if (
            r["index"] != index
            or c["assisted"] is not False
            or key in by_key
            or type(c["state_id"]) is not int
            or type(c["task"]) is not int
            or not 0 <= c["state_id"] < 40
            or c["task"] not in (0, 1)
            or c["context"] != (1, 4, 20, 21)[c["state_id"] // 10]
        ):
            raise ValueError("unchanged complete unassisted current TRAIN80 grid required")
        by_key[key] = r
    protected = [r["index"] for r in native_records if controlled_pass_label(r["outcome"])]
    if len(protected) != 41:
        raise ValueError("exact105025 acquired41-skill frontier required for this generation")
    old = [r["index"] for r in old_records if r["role"] == "retained_neural17"]
    if len(old) != 17 or any(
        old_records[i]["index"] != i or old_records[i]["task_success"] is not True for i in old
    ):
        raise ValueError("all original17 retained motion anchors required")
    for base in coverage["baseline"]:
        key = (base["state_id"], base["task"])
        if controlled_pass_label(base["outcome"]) != controlled_pass_label(by_key[key]["outcome"]):
            raise ValueError("calibrated nominal baseline must match actual current candidate")
    rescue_keys = {tuple(k) for k in measured["rescues"]}
    teachers = []
    for r in coverage["selected_sources"]:
        key = (r["state_id"], r["task"])
        if key not in rescue_keys:
            continue
        baseline = by_key[key]
        if controlled_pass_label(baseline["outcome"]) or not controlled_pass_label(r["outcome"]):
            raise ValueError(
                "only genuine current-failure to controlled-teacher-success corrections"
            )
        for field in (
            "context",
            "state_id",
            "task",
            "side",
            "target",
            "center",
            "foot",
            "ball_planar_offset_world",
        ):
            if r["course"][field] != baseline["course"][field]:
                raise ValueError("same physical initial course/goal required for every rescue")
        teachers.append(r)
    teachers.sort(key=lambda r: (r["state_id"], r["task"]))
    if len(teachers) != len(rescue_keys) or len({r["state_id"] for r in teachers}) < 2:
        raise ValueError("all rescued tasks across at least two physical states required")
    return dict(
        schema="soccer.qualified_pass_refinement_curriculum.v1",
        original_motion_anchor_indices=old,
        current_pass_anchor_indices=protected,
        correction_sources=teachers,
        current_failed_indices=[r["index"] for r in native_records if r["index"] not in protected],
        correction_teacher_count=len(teachers),
        fit_histories=17 + 41 + len(teachers),
        success_targets_replaced_by_teacher=False,
        original_negative_histories_retained=True,
        teacher_hidden_state_supplied=False,
        observation_width=160,
        source_policy="current105025",
        student_optimizer_updates=0,
        Fresh_fit_rows=0,
        promotion_authorized=False,
    )
