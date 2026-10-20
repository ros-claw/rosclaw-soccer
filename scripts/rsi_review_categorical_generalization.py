"""Authenticate all96 four-arm native records and compute paired comparisons."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("launch", "plan", "output", "worker-source", "inputs"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "expected-launch-hash", "expected-inputs-hash"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or subprocess.check_output(["git", "status", "--porcelain"])
        or digest(a.launch) != a.expected_launch_hash
        or digest(a.inputs) != a.expected_inputs_hash
    ):
        raise ValueError("immutable review source, exact inputs/launch and unused output required")
    launch = json.loads(a.launch.read_text())
    if (
        len(launch["jobs"]) != 16
        or len({j["unit"] for j in launch["jobs"]}) != 16
        or len({j["output"] for j in launch["jobs"]}) != 16
        or subprocess.check_output(
            ["git", "-C", str(a.worker_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != launch["source_head"]
    ):
        raise ValueError("exact sixteen disjoint declared workers required")
    for folder in ("src", "scripts"):
        for old in (a.worker_source / folder).rglob("*.py"):
            if digest(old) != digest(Path.cwd() / old.relative_to(a.worker_source)):
                raise ValueError("review cannot alter frozen generator or actor/physics")
    if digest(a.plan / "complete.json") != launch["plan_complete_hash"]:
        raise ValueError("whole sealed population differs")
    plan = json.loads((a.plan / "complete.json").read_text())
    if (
        plan["status"] != "CATEGORICAL_POSTFREEZE96_DECLARED"
        or plan["source_head"] != launch["source_head"]
        or digest(a.plan / "courses.json") != plan["courses_hash"]
    ):
        raise ValueError("declared complete96 population required")
    from rosclaw_soccer.training.categorical_motion_holdout import (
        categorical_motion_holdout_courses,
    )
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = categorical_motion_holdout_courses()
    table = json.loads(json.dumps([asdict(c) for c in courses]))
    if (
        preflight_receiving_courses(courses) != plan["preflight_hash"]
        or json.loads((a.plan / "courses.json").read_text())["courses"] != table
    ):
        raise ValueError("preflight or entire new-case identity differs")
    pins = dict(plan["input_file_hashes"])
    for folder in ("src", "scripts"):
        pins.update({str(f.resolve()): digest(f) for f in Path(folder).rglob("*.py")})
    pins[str(a.launch)] = a.expected_launch_hash
    pins[str(a.inputs)] = a.expected_inputs_hash
    pins[str(a.plan / "complete.json")] = launch["plan_complete_hash"]
    pins[str(a.plan / "courses.json")] = plan["courses_hash"]
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            launch_hash=a.expected_launch_hash,
            plan_hash=launch["plan_complete_hash"],
            new_native_steps=0,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        for j in launch["jobs"]:
            wait_for_terminal_collection(j["unit"] + ".service", wait=True)
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.categorical_pair_summary import summarize_categorical_pairs
        from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
        from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry
        from scripts.rsi_motion_student_rollout import validate_motion_recording
        from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array

        spec = json.loads(a.inputs.read_text())
        world = mujoco.MjModel.from_binary_path(spec["world"])
        rows = []
        for part, j in enumerate(launch["jobs"]):
            expected_indices = list(range(96))[part::16]
            root = Path(j["output"])
            d = json.loads((root / "declaration.json").read_text())
            c = json.loads((root / "complete.json").read_text())
            if (
                j["indices"] != expected_indices
                or d["indices"] != expected_indices
                or d["roster"] != plan["roster"]
                or d["plan_hash"] != launch["plan_complete_hash"]
                or d["source_head"] != launch["source_head"]
                or c["source_head"] != launch["source_head"]
                or c["status"] != "CATEGORICAL_NEAR96_FOUR_ARM_NATIVE_VERIFIED"
                or c["new_native_steps"] != 24000 * len(expected_indices)
                or c["optimizer_steps"] != 0
                or c["hardware_authorized"] is not False
                or c["promotion_authorized"] is not False
                or len(c["results"]) != len(expected_indices)
            ):
                raise ValueError("whole exact worker accounting/roster/partition required")
            for k, h in d["input_file_hashes"].items():
                if k in pins and pins[k] != h:
                    raise ValueError("worker commitment conflict")
                pins[k] = h
            for name in ("declaration.json", "complete.json"):
                pins[str(root / name)] = digest(root / name)
            for i, row in zip(expected_indices, c["results"], strict=True):
                folder = root / f"course{i:03d}"
                if (
                    row["course"] != table[i]
                    or row["new_native_steps"] != 24000
                    or len(row["variants"]) != 4
                ):
                    raise ValueError("complete paired case identity required")
                for f in (folder / "complete.json", root / f"progress{i:03d}.json"):
                    if json.loads(f.read_text()) != row:
                        raise ValueError("case progress/terminal mismatch")
                    pins[str(f)] = digest(f)
                common = None
                for roster, v in zip(plan["roster"], row["variants"], strict=True):
                    sub = folder / v["variant"]
                    if (
                        v["variant"] != roster["variant"]
                        or v["checkpoint_hash"] != roster["hash"]
                        or json.loads((sub / "complete.json").read_text()) != v
                        or digest(sub / "native.npz") != v["native_hash"]
                        or v["native_steps"] != 6000
                    ):
                        raise ValueError("exact paired model/raw trajectory required")
                    for f in (sub / "complete.json", sub / "replay.json"):
                        pins[str(f)] = digest(f)
                    pins[str(sub / "native.npz")] = v["native_hash"]
                    with np.load(sub / "native.npz", allow_pickle=False) as z:
                        trace = {k: z[k].copy() for k in z.files}
                    validate_motion_recording(trace, nbody=world.nbody)
                    if (
                        score_full_entry(world, {k: q[20:] for k, q in trace.items()}) != v["score"]
                        or diagnose_entry_prefix(world, {k: q[:20] for k, q in trace.items()})
                        != v["prefix_diagnostics"]
                    ):
                        raise ValueError("native primitives disagree with outcome gate")
                    fields = (
                        *PHYSICAL_FIELDS,
                        "student_features",
                        "muscle_proprioception",
                        "consumed_action_isaac",
                        "hidden_after",
                        "value_estimate",
                    )
                    if common is None:
                        common = {k: trace[k][:20].copy() for k in fields}
                    else:
                        for k in fields:
                            exact_array(trace[k][:20], common[k])
                    proof = v["replay"]
                    if (
                        json.loads((sub / "replay.json").read_text()) != proof
                        or proof["additional_native_steps"] != 3000
                        or proof["student_neural_and_muscle_native_bit_exact"] is not True
                        or proof["intention_network_calls"]
                        != (600 if v["variant"] == "anchor" else 900)
                        or proof["muscle_decoder_calls"] != 280
                        or proof["teacher_encoder_calls"] != 0
                        or proof["teacher_planner_calls"] != 0
                        or proof["hardware_authorized"] is not False
                        or proof["promotion_authorized"] is not False
                    ):
                        raise ValueError("complete independent policy/muscle/native proof required")
                rows.append(row)
        summary = summarize_categorical_pairs(rows)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="CATEGORICAL_NEAR96_FOUR_ARM_PRIMITIVES_AUTHENTICATED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                summary=summary,
                reviewed_native_steps=2304000,
                new_native_steps=0,
                optimizer_steps=0,
                independently_reintegrated_by_this_review=False,
                original_far_or_private_fresh_pass=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
        print("CATEGORICAL_NEAR96_REVIEWED", summary, flush=True)
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(error=repr(error), automatic_retry_authorized=False, new_native_steps=0),
        )
        raise


if __name__ == "__main__":
    main()
