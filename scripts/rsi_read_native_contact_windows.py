"""Authenticate existing native pilots and measure censored ball outcome windows.

Read-only analysis: no simulation, policy inference, learning, or old exam edits.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", nargs=3, action="append", required=True, metavar=("UNIT", "ROOT", "HASH")
    )
    for key in ("plan-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("plan-unit", "plan-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or not 1 <= len(a.source) <= 32
    ):
        raise ValueError("frozen read-only analyzer, bounded sources and unused output required")
    wait_for_terminal_collection(a.plan_unit)
    plan_path = a.plan_root / "complete.json"
    if digest(plan_path) != a.plan_hash:
        raise ValueError("actual immutable paired plan required")
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("force_decoder") != "addressed"
        or plan.get("all_initial_states_paired_exact") is not True
    ):
        raise ValueError("actual paired addressed plan required")
    pins = dict(plan["input_file_hashes"])
    pins[str(plan_path)] = a.plan_hash
    reports = []
    worlds = set()
    for unit, name, expected in a.source:
        wait_for_terminal_collection(unit)
        root = Path(name)
        path = root / "complete.json"
        if digest(path) != expected:
            raise ValueError("actual source receipt identity differs")
        d = json.loads(path.read_text())
        if (
            d.get("status") != "BALL_AWARE_NATIVE_CONTACT_NOT_GOAL_OR_TEAM_QUALIFIED"
            or d.get("all_full_independent_repeats_exact") is not True
            or d["input_file_hashes"].get(str(plan_path)) != a.plan_hash
            or len(d["results"]) != 8
        ):
            raise ValueError("complete paired native pilot bound to same exogenous plans required")
        world_ids = {r["world_index"] for r in d["results"]}
        if len(world_ids) != 1 or worlds & world_ids:
            raise ValueError("each independent initial world may be analyzed only once")
        worlds |= world_ids
        pins.update(d["input_file_hashes"])
        pins[str(path)] = expected
        for r in d["results"]:
            raw = root / f"source{r['source']}-{r['arm']}-repeat{r['repeat']}" / "native.npz"
            pins[str(raw)] = r["raw_hash"]
        reports.append((root, d))
    for index in range(2):
        source_name = ("soccer-standard-007_left.npz", "soccer-standard-010_right.npz")[index]
        row = next(
            r for r in plan["results"] if r["arm"] == "initial" and r["source"] == source_name
        )
        pins[str(a.plan_root / f"{index}-initial-raw.npz")] = row["raw_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            new_native_steps=0,
            new_neural_calls=0,
            new_optimizer_steps=0,
            profile="native_reference_motion.contact10N.post1p2s.v1",
            historical_reports_unchanged=True,
            formal_exam_unchanged=True,
            hardware_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.native_contact_window import native_contact_window

    rows = []
    for root, report in reports:
        for r in report["results"]:
            source, world = r["source"], r["world_index"]
            index = 0 if source == 7 else 1
            with np.load(a.plan_root / f"{index}-initial-raw.npz", allow_pickle=False) as z:
                goal = z["goal_target_world"][0, world].copy()
            raw = root / f"source{source}-{r['arm']}-repeat{r['repeat']}" / "native.npz"
            with np.load(raw, allow_pickle=False) as z:
                values = native_contact_window(
                    force=z["foot_ball_contact_force_n"].reshape(-1, 2),
                    canonical_qpos=z["canonical_qpos_after"].reshape(-1, 43),
                    canonical_qvel=z["canonical_qvel_after"].reshape(-1, 41),
                    goal=goal,
                    intended_foot=index,
                )
            rows.append(
                dict(source=source, world=world, arm=r["arm"], repeat=r["repeat"], **values)
            )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="READ_ONLY_NATIVE_CONTACT_WINDOW_NOT_TASK_OR_FRESH_QUALIFICATION",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=rows,
            new_native_steps=0,
            new_neural_calls=0,
            new_optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
