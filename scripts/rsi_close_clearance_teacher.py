"""Close the fixed pilot from original full actual arrays and saved predictions."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", nargs=3, action="append", required=True, metavar=("INDEX", "ROOT", "HASH")
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_pilot_clearance_teacher import INDICES

    clean(a.expected_head)
    if a.output.exists() or tuple(int(r[0]) for r in a.case) != INDICES:
        raise ValueError("exact ordered six-state bank and unused output required")
    pins, cases = {}, []
    for index, root, expected in a.case:
        root = Path(root)
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact closed case required")
        report = json.loads(path.read_text())
        if report["status"] != "FINITE_CLEARANCE_TEACHER_ACTUAL_CASE_COMPLETE" or report[
            "case_index"
        ] != int(index):
            raise ValueError("closed declared physical case required")
        for name, commitment in report["input_file_hashes"].items():
            if name in pins and pins[name] != commitment:
                raise ValueError("source commitment conflict")
            pins[name] = commitment
        pins[str(path)] = expected
        cases.append((int(index), root, report))
    source_pins(pins)
    check_pins(pins)
    import numpy as np

    from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    results = []
    integrated = attempted = native = 0
    for index, root, report in cases:
        old_json = a.source_root / f"course-{index:02d}" / "C.json"
        if pins.get(str(old_json)) != digest(old_json) or old_json.is_symlink():
            raise ValueError("exact independently audited old teacher record required")
        old_record = json.loads(old_json.read_text())
        old_path = old_json.parent / old_record["artifact"]["path"]
        with (
            np.load(old_path, allow_pickle=False) as old,
            np.load(root / "actual-unvalidated.npz", allow_pickle=False) as new,
        ):
            actual = {k: new[k].copy() for k in new.files}
            if set(old.files) != set(actual) or any(
                not np.array_equal(v, old[k]) for k, v in actual.items()
            ):
                raise ValueError("no-adoption pilot must match the complete old actual path")
        row = report["actual_record"]
        outcome = relabel_arrays(
            actual,
            controls=360,
            ball_address=row["ball_qpos_address"],
            root_address=row["robot_qpos_address"],
        )
        if hash_json(outcome) != hash_json(row["outcome"]) or hash_json(outcome) != hash_json(
            old_record["outcome"]
        ):
            raise ValueError("original H300 full physical labels differ")
        passing = bool(old_record["outcome"]["task_pass"])
        if report["clearance_enabled"] or report["rescue"] or report["regression"]:
            raise ValueError(
                "this no-gain closing audit cannot hide candidate adoption or a changed outcome"
            )
        count = 37 if passing else 38
        consumed = 0
        for j in range(count):
            path = root / f"forecast-{j:02d}.npz"
            meta = json.loads(path.with_suffix(".json").read_text())
            with np.load(path, allow_pickle=False) as raw:
                values = {k: raw[k].copy() for k in raw.files}
            if (
                any(not np.isfinite(v).all() for v in values.values())
                or len(values["qpos"]) != meta["native_steps"]
                or len(values["joint_targets"]) * 10 != meta["native_steps"]
            ):
                raise ValueError("unfilled finite raw forecast steps required")
            consumed += meta["native_steps"]
            if j == 37:
                if meta["name"] != "measured-clearance" or meta["controls"] != 128:
                    raise ValueError("one complete measured-feedback candidate required")
                observed = predicted_touch_window(
                    dict(values, controls=128),
                    ball_address=row["ball_qpos_address"],
                    root_address=row["robot_qpos_address"],
                    start_control=row["entry_control"],
                    diagnostic_tail=True,
                )
                if observed["window_pass"] or hash_json(observed) != hash_json(
                    report["clearance_events"][0]["window"]
                ):
                    raise ValueError("raw candidate outcome differs from recorded rejection")
        if passing and (root / "forecast-37.npz").exists():
            raise ValueError("protected old passing branch evaluated an unnecessary new candidate")
        if consumed != report["forecasts"]["integrated_native_steps"]:
            raise ValueError("forecast step accounting differs from every saved raw future")
        integrated += consumed
        attempted += report["forecasts"]["attempted_native_steps"]
        native += row["native_steps"]
        results.append(
            dict(
                index=index,
                old_pass=passing,
                actual_pass=outcome["task_pass"],
                whole_actual_bitexact=True,
                candidate_tested=not passing,
                candidate_accepted=False,
                rescue=False,
                regression=False,
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="FINITE_CLEARANCE_TEACHER_REJECTED_ZERO_GAIN",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            old_passes=sum(r["old_pass"] for r in results),
            new_passes=sum(r["actual_pass"] for r in results),
            rescues=0,
            regressions=0,
            actual_native_steps_readback=native,
            forecast_native_steps_readback=integrated,
            forecast_attempted_steps=attempted,
            new_physical_steps=0,
            new_neural_calls=0,
            optimizer_updates=0,
            independent_nn_semantics_scope=(
                "separately qualified source interface, not all pilot futures"
            ),
            expansion_rejected=True,
            stop_activation_lead_amplitude_seed_sweeps=True,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
