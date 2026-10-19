"""Independent sealed-array readback of prospective full-task zero predictions.

No new physics, NN, learning or temporal attestation: persistence-before-main
is a producer execution-order commitment, not proved by filesystem timestamps.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS, PREFIX
from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_full_task_branch_inputs import require_scope
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def same_arrays(actual, expected):
    if actual.keys() != expected.keys() or any(
        a.shape != expected[k].shape
        or a.dtype != expected[k].dtype
        or a.tobytes() != expected[k].tobytes()
        for k, a in actual.items()
    ):
        raise ValueError("all physical/control/NN tensors must match bitwise")


def review_case(row, job, load_arrays):
    """Recompute labels and every future tensor, without running producer logic."""
    main, gold = load_arrays(row["arrays"]), load_arrays(job["gold_arrays"])
    original_arrays(main)
    if set(main) != set(original_arrays(main)) | {PREFIX + k for k in FIELDS}:
        raise ValueError("all43 full body/control/NN fields required")
    same_arrays(main, gold)
    outcome = label_first_touch(main).to_dict()
    start = outcome["first_foot_contact_frame"] + 1
    if (
        row["index"] != job["index"]
        or row["course"] != job["course"]
        or row["source_case_hash"] != job["source_case_hash"]
        or start != job["comparison_start_after_measured_foot_bin"]
        or row["outcome"] != outcome
        or len(row["predictions"]) != 2
        or any(
            row[k] is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or type(row["new_optimizer_updates"]) is not int
        or row["new_optimizer_updates"] != 0
        or any(
            type(row["main_review"][k]) is not int or row["main_review"][k] != 3000
            for k in ("native_attempted_steps", "native_integrated_steps")
        )
    ):
        raise ValueError("same complete ordered baseline case required")
    planner_calls = 0
    for bi, pred in enumerate(row["predictions"]):
        review = pred["review"]
        if (
            pred["index"] != row["index"]
            or pred["branch_index"] != bi
            or pred["start_frame"] != start
            or pred["persisted_before_main_advances_frame"] != start
            or pred["source_future_used_for_prediction"] is not False
            or review["schema"] != "soccer.rsi.full_task_branch_review.v1"
            or review["action_name"] != "baseline"
            or review["start_frame"] != start
            or review["end_exclusive"] != 300
            or review["borrowed_prefix_frames"] != start
            or any(
                type(review[k]) is not int or review[k] != (300 - start) * 10
                for k in ("new_native_attempted_steps", "new_native_integrated_steps")
            )
            or any(
                type(review[k]) is not int or review[k] != 300 - start
                for k in (
                    "recomputed_control_frames",
                    "recomputed_foundation_encoder_decoder_pairs",
                )
            )
            or review["future_observations_from_own_simulation"] is not True
            or any(
                review[k] is not False
                for k in (
                    "recorded_future_targets_or_states_consumed",
                    "prefix_steps_counted_as_new",
                    "teacher_qualified",
                    "hardware_authorized",
                    "promotion_authorized",
                )
            )
            or type(review["new_optimizer_updates"]) is not int
            or review["new_optimizer_updates"] != 0
        ):
            raise ValueError("exact baseline remainder accounting and commitments required")
        expected_calls = sum(f % 20 == 0 for f in range(start, 300))
        if (
            type(review["new_planner_onnx_calls"]) is not int
            or review["new_planner_onnx_calls"] != expected_calls
        ):
            raise ValueError("every remaining scheduled planner refresh required")
        planner_calls += expected_calls
        if (
            review["donor_integration_hash"]
            != row["predictions"][0]["review"]["donor_integration_hash"]
        ):
            raise ValueError("both predictions must fork the same measured donor")
        future = load_arrays(pred["arrays"])
        same_arrays(future, {k: a[start:] for k, a in main.items()})
        measured = {k: np.concatenate((main[k][:start], future[k])) for k in main}
        if (
            label_first_touch(measured).to_dict() != pred["predicted_outcome"]
            or pred["predicted_outcome"] != outcome
        ):
            raise ValueError("own future plus measured past M0 prediction must match")
    return dict(
        index=row["index"],
        start_frame=start,
        new_main_steps=3000,
        new_prediction_steps=2 * (300 - start) * 10,
        predicted_planner_calls=planner_calls,
        outcome=outcome,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--declaration-hash", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.root / "failure.json").exists():
        raise ValueError("unused readback output and no producer failure required")
    require_terminal(args.unit)
    pins = {}

    def pin(name, expected=None):
        path = Path(name)
        value = digest(path)
        if (
            path.is_symlink()
            or (expected is not None and value != expected)
            or (str(path) in pins and pins[str(path)] != value)
        ):
            raise ValueError("immutable exact whole-task input required")
        pins[str(path)] = value

    def sealed(path):
        pin(path)
        value = json.loads(Path(path).read_text())
        if value["report_hash"] != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("whole-task artifact seal mismatch")
        return value

    def arrays(witness):
        pin(witness["path"], witness["sha256"])
        with np.load(witness["path"], allow_pickle=False) as archive:
            return dict(archive)

    declaration = sealed(args.root / "declaration.json")
    complete = sealed(args.root / "complete.json")
    require_scope(declaration["config"])
    if (
        declaration["report_hash"] != args.declaration_hash
        or complete["declaration_hash"] != args.declaration_hash
        or declaration["schema"] != "soccer.rsi.full_task_zero18_declaration.v1"
        or complete["schema"] != "soccer.rsi.full_task_zero18_complete.v1"
        or len(declaration["jobs"]) != 18
        or len(complete["rows"]) != 18
        or Path(declaration["config"]["output_root"]) != args.root
        or any(
            complete.get(k) is not False
            for k in (
                "nonzero_action_prediction_qualified",
                "teacher_qualified",
                "training_authorized",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
        or complete.get("baseline_whole_task_prediction_qualified_on_this_consumed_scope")
        is not True
        or any(
            type(complete.get(k)) is not int or complete[k] != v
            for k, v in dict(
                main_reconstruction_runs=18,
                prediction_runs=36,
                executed_control_foundation_pairs=9484,
                initialization_foundation_parity_pairs=4,
                new_optimizer_updates=0,
            ).items()
        )
    ):
        raise ValueError("prebound whole18 declaration and completion required")
    for name, value in declaration["input_file_hashes"].items():
        pin(name, value)
    rows = []
    for i, (row, job) in enumerate(zip(complete["rows"], declaration["jobs"], strict=True)):
        if row != sealed(args.root / f"case-{i}-complete.json") or row["index"] != i:
            raise ValueError("all18 separately sealed ordered rows required")
        for bi, pred in enumerate(row["predictions"]):
            if pred != sealed(args.root / f"case-{i}-branch-{bi}-prediction.json"):
                raise ValueError("both separately sealed predictions required")
        rows.append(review_case(row, job, arrays))
    totals = dict(
        new_main_native_steps=sum(r["new_main_steps"] for r in rows),
        new_prediction_native_steps=sum(r["new_prediction_steps"] for r in rows),
    )
    totals["combined_new_native_steps"] = sum(totals.values())
    if totals != dict(
        new_main_native_steps=54000,
        new_prediction_native_steps=40840,
        combined_new_native_steps=94840,
    ) or any(complete[k] != v for k, v in totals.items()):
        raise ValueError("exact whole94840 native-step accounting required")
    if any(digest(name) != value for name, value in pins.items()):
        raise ValueError("inputs changed during independent readback")
    report = dict(
        schema="soccer.rsi.full_task_zero18_raw_readback.v1",
        producer_complete_hash=complete["report_hash"],
        input_file_hashes=pins,
        rows=rows,
        producer_steps_reviewed=totals,
        new_physical_steps=0,
        new_nn_inferences=0,
        new_optimizer_updates=0,
        nonzero_action_prediction_qualified=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        temporal_execution_order_independently_attested=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print("FULL_TASK_ZERO18_RAW_READBACK_COMPLETE", report["report_hash"], flush=True)


if __name__ == "__main__":
    main()
