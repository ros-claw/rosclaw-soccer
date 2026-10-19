"""Recompute every saved impact proxy and authenticate the original terminal journal."""

import argparse
import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecast
from rosclaw_soccer.rsi.impact_torque_screen import impact_prediction_score
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_impact_screen_inputs import require_scope
from scripts.rsi_stage_release_contact_replay import digest


def require_terminal_journal(events, unit):
    """Preserve the original transient unit's history; never recreate/relaunch it."""
    messages = [e for e in events if e.get("_SYSTEMD_USER_UNIT") == unit]
    ids = {e.get("_SYSTEMD_INVOCATION_ID") for e in messages}
    if len(ids) != 1 or None in ids:
        raise ValueError("one original invocation required")
    invocation = next(iter(ids))
    completions = [
        e
        for e in events
        if e.get("USER_UNIT") == unit
        and e.get("USER_INVOCATION_ID") == invocation
        and e.get("MESSAGE_ID") == "39f53479d3a045ac8e11786248231fbf"
        and e.get("JOB_TYPE") == "start"
        and e.get("JOB_RESULT") == "done"
    ]
    if len(completions) != 1 or {e.get("MESSAGE") for e in messages} != {
        f"IMPACT_CASE_COMPLETE {i} FORECAST_STEPS 140" for i in range(6)
    }:
        raise ValueError("all six original case completions and manager terminal success required")
    if max(int(e["__REALTIME_TIMESTAMP"]) for e in messages) >= int(
        completions[0]["__REALTIME_TIMESTAMP"]
    ):
        raise ValueError("journal terminal order differs")
    return invocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--complete-file-hash", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("exact clean frozen readback source required")
    if args.output.exists() or (args.root / "failure.json").exists():
        raise ValueError("unused readback and no source failure required")
    pins = {
        str(path): digest(path)
        for folder in (source / "src", source / "scripts")
        for path in folder.rglob("*.py")
    }

    def sealed(path):
        if path.is_symlink():
            raise ValueError("ordinary source evidence required")
        h = digest(path)
        value = json.loads(path.read_text())
        if value["report_hash"] != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("whole artifact seal differs")
        pins[str(path)] = h
        return value

    if digest(args.root / "complete.json") != args.complete_file_hash:
        raise ValueError("declared actual complete bytes required")
    complete = sealed(args.root / "complete.json")
    declaration = sealed(args.root / "declaration.json")
    require_scope(declaration["config"])
    if (
        complete["declaration_hash"] != declaration["report_hash"]
        or complete["schema"] != "soccer.rsi.retrospective_impact_torque_screen_complete.v1"
        or complete["all_six_source_cases_retained"] is not True
        or len(complete["rows"]) != 6
        or any(
            complete[k] is not False
            for k in (
                "final_m0_improvement_proven",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            type(complete[k]) is not int or complete[k] != n
            for k, n in (
                ("recorded_attempted_steps", 18000),
                ("recorded_integrated_steps", 18000),
                ("forecast_attempted_steps", 840),
                ("forecast_integrated_steps", 840),
                ("combined_attempted_steps", 18840),
                ("combined_integrated_steps", 18840),
                ("new_closed_loop_policy_trials", 0),
                ("new_neural_inferences", 0),
                ("new_optimizer_updates", 0),
            )
        )
    ):
        raise ValueError("unchanged whole six bounded diagnostic required")
    for name, h in declaration["input_file_hashes"].items():
        if digest(name) != h:
            raise ValueError("original complete input/source chain differs")
        pins[name] = h
    raw = subprocess.check_output(
        ["journalctl", "--user", "-u", args.unit, "-o", "json", "--no-pager"], text=True
    )
    events = [json.loads(line) for line in raw.splitlines()]
    invocation = require_terminal_journal(events, args.unit)
    rows = []
    for index, row in enumerate(complete["rows"]):
        if sealed(args.root / f"case-{index}-complete.json") != row:
            raise ValueError("ordered complete case differs")
        if (
            row["index"] != index
            or len(row["predictions"]) != 13
            or row["repeat_zero_bit_equal"] is not True
            or row["zero_forecast_matches_original_ten_substeps"] is not True
            or row["live_action_applied"] is not False
            or row["forecast_substeps"] != 140
        ):
            raise ValueError("whole original prediction bank and parity required")
        donors, feasible = set(), []
        for number, prediction in enumerate(row["predictions"]):
            path = args.root / f"case-{index}-candidate-{number}.json"
            if (
                json.loads(path.read_text()) != prediction
                or prediction["candidate_index"] != number
            ):
                raise ValueError("all ordered original candidate metadata required")
            pins[str(path)] = digest(path)
            arrays = args.root / f"case-{index}-candidate-{number}.npz"
            if (
                str(arrays) != prediction["arrays"]["path"]
                or digest(arrays) != prediction["arrays"]["sha256"]
            ):
                raise ValueError("whole raw prediction arrays required")
            pins[str(arrays)] = digest(arrays)
            with np.load(arrays, allow_pickle=False) as archive:
                values = dict(archive)
            offset = values.pop("offset_nm")
            np.testing.assert_array_equal(
                offset, np.asarray(prediction["offset_nm"], dtype=np.float64)
            )
            values.update(
                donor_integration_state_hash=prediction["donor_integration_state_hash"],
                nonfoot_robot_contact_over_1n=prediction["nonfoot_robot_contact_over_1n"],
            )
            if set(values) != {f.name for f in dataclasses.fields(G1PDTorqueForecast)}:
                raise ValueError("every original forecast field required")
            score = impact_prediction_score(G1PDTorqueForecast(**values), offset)
            if score != prediction["score"]:
                raise ValueError("saved proxy score not reproduced")
            donors.add(prediction["donor_integration_state_hash"])
            if score["feasible_predicted_impact"]:
                feasible.append(prediction)
        best = min(feasible, key=lambda r: r["score"]["proxy_cost"]) if feasible else None
        selected = None if best is None else best["candidate_index"]
        if len(donors) != 1 or selected != row["best_proxy_candidate_index"]:
            raise ValueError("same donor and stable all-candidate selection required")
        baseline = row["predictions"][0]["score"]
        rows.append(
            dict(
                index=index,
                job=row["job"],
                original_outcome=row["original_outcome"],
                baseline=baseline,
                selected_index=selected,
                selected=None if best is None else best["score"],
                proxy_improved=best is not None
                and baseline["proxy_cost"] is not None
                and best["score"]["proxy_cost"] < baseline["proxy_cost"],
            )
        )
    if any(digest(name) != h for name, h in pins.items()):
        raise ValueError("source/input changed during readback")
    report = dict(
        schema="soccer.rsi.impact_torque_screen_readback.v1",
        source_head=args.source_head,
        complete_hash=complete["report_hash"],
        input_file_hashes=pins,
        terminal_journal=events,
        original_invocation=invocation,
        all78_saved_proxy_scores_recomputed=True,
        rows=rows,
        proxy_improved_source_cases=sum(r["proxy_improved"] for r in rows),
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        final_m0_improvement_proven=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(
        "WHOLE_IMPACT_SCREEN_READBACK",
        report["report_hash"],
        "PROXY_IMPROVED",
        report["proxy_improved_source_cases"],
        flush=True,
    )


if __name__ == "__main__":
    main()
