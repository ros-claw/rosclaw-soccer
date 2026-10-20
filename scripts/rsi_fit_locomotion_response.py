"""Wait for the complete bounded experiment, raw audit, then train-only model fit.

No new physics/NN calls or actor updates. A failed prediction gate never launches
an MPC or promotes a motor. Output includes every fixed validation intervention.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("launch", "worker-source", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "expected-launch-hash"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or digest(a.launch) != a.expected_launch_hash
    ):
        raise ValueError("clean frozen fitter and exact launched physical protocol required")
    launch = json.loads(a.launch.read_text())
    if (
        len(launch["jobs"]) != 8
        or len({j["unit"] for j in launch["jobs"]}) != 8
        or launch["maximum_new_native_steps"] != 96000
        or Path(launch["source"]) != a.worker_source
        or subprocess.check_output(
            ["git", "-C", str(a.worker_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != launch["source_head"]
        or subprocess.check_output(["git", "-C", str(a.worker_source), "status", "--porcelain"])
    ):
        raise ValueError("exact pristine locomotion scientific worker source required")
    for directory in ("src", "scripts"):
        for f in (a.worker_source / directory).rglob("*.py"):
            if digest(f) != digest(Path.cwd() / f.relative_to(a.worker_source)):
                raise ValueError("fitter cannot change original numerical/control math")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            worker_source_head=launch["source_head"],
            launch_hash=a.expected_launch_hash,
            training_commands=list(range(9)),
            validation_commands=list(range(9, 15)),
            delay_candidates=list(range(21)),
            ridge=1e-4,
            new_native_steps=0,
            new_neural_calls=0,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    pins = {str(a.launch): a.expected_launch_hash}
    pins.update(
        {
            str(f): digest(f)
            for directory in (Path.cwd() / "src", Path.cwd() / "scripts")
            for f in directory.rglob("*.py")
        }
    )
    parts = {}
    try:
        for i, j in enumerate(launch["jobs"]):
            wait_for_terminal_collection(j["unit"], wait=True)
            root = Path(j["root"])
            c = json.loads((root / "complete.json").read_text())
            if c["source_head"] != launch["source_head"]:
                raise ValueError("scientific job source differs")
            if i == 0:
                status = "CALIBRATED_ENTRY_INITIATION_FAILURE_DIAGNOSED"
                declared = c
            else:
                declared = json.loads((root / "declaration.json").read_text())
                if i == 1:
                    status = "LOCOMOTION_RUNNER_ORIGINAL_FROM0_PARITY"
                    if c["native_steps"] != 6000:
                        raise ValueError("full original six thousand qualification required")
                else:
                    part = (i - 2) // 2
                    repeated = bool((i - 2) % 2)
                    status = (
                        "LOCOMOTION_RESPONSE_PART_REPRODUCED"
                        if repeated
                        else "LOCOMOTION_RESPONSE_PART_COLLECTED"
                    )
                    if c["part"] != part or c["native_steps"] != 15000 or len(c["results"]) != 5:
                        raise ValueError("whole declared part native budget required")
                    parts[part, repeated] = (root, c)
            if c["status"] != status:
                raise ValueError("actual terminal artifact is not the declared scientific result")
            for k, h in declared["input_file_hashes"].items():
                if k in pins and pins[k] != h:
                    raise ValueError("conflicting original numerical source/assets")
                pins[k] = h
            pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
        check_pins(pins)
        import numpy as np

        from rosclaw_soccer.rsi.locomotion_response_model import fit_training, validate_prediction
        from rosclaw_soccer.rsi.locomotion_response_protocol import (
            preflight_response_courses,
            response_courses,
            summarize_response,
        )
        from scripts.rsi_collect_locomotion_response import compare_response_row
        from scripts.rsi_probe_compact_source import exact_artifact, load_arrays

        courses = response_courses()
        bank = preflight_response_courses(courses)
        records = {}
        physical = []
        for part in range(3):
            root, c = parts[part, False]
            repeat, rc = parts[part, True]
            if c["bank_hash"] != bank or rc["bank_hash"] != bank:
                raise ValueError("complete exact command/split bank required")
            for course, row, rerow in zip(
                courses[part::3], c["results"], rc["results"], strict=True
            ):
                folder = f"command{course.index:02d}"
                if row != json.loads(
                    (root / folder / "complete.json").read_text()
                ) or rerow != json.loads((repeat / folder / "complete.json").read_text()):
                    raise ValueError("part summary differs from actual episode")
                compare_response_row(row, rerow)
                exact_artifact(root / folder / "native300.npz", repeat / folder / "native300.npz")
                for receipt, directory in ((row, root), (rerow, repeat)):
                    if (
                        receipt["native_hash"] != digest(directory / folder / "native300.npz")
                        or receipt["native_steps"] != 3000
                        or receipt["new_optimizer_steps"] != 0
                        or receipt["body_report"]["frames"] != 300
                        or receipt["body_report"]["integrated_steps"] != 3000
                        or receipt["body_report"]["attempted_steps"] != 3000
                    ):
                        raise ValueError("full actual native trajectory commitment required")
                raw = load_arrays(root / folder / "native300.npz")
                summary = summarize_response(raw, course)
                if any(row.get(k) != json.loads(json.dumps(v)) for k, v in summary.items()):
                    raise ValueError("recomputed primitive response differs")
                records[course.index] = raw
                physical.append(summary)
        if set(records) != set(range(15)):
            raise ValueError("all fifteen actual interventions required")
        fit = fit_training(
            [
                (
                    records[i]["navigation_command"][:, :2].copy(),
                    records[i]["canonical_qvel"][:, :2].copy(),
                )
                for i in range(9)
            ]
        )
        w = np.asarray(fit["selected"]["weights"], dtype=np.float64)
        validation = []
        for course in courses[9:]:
            raw = records[course.index]
            validation.append(
                dict(
                    command_index=course.index,
                    **validate_prediction(
                        w,
                        fit["selected"]["delay_frames"],
                        raw["navigation_command"][:, :2].copy(),
                        raw["canonical_qvel"][:, :2].copy(),
                        raw["canonical_qpos"][:, :2].copy(),
                        onset_frame=75,
                    ),
                )
            )
        all_predict = all(r["predictor_research_gate"] for r in validation)
        all_stable = all(
            r["stable_height_and_upright"] and r["no_robot_ball_contact"] for r in physical
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="LOCOMOTION_RESPONSES_AUDITED_AND_DELAYED_MODEL_FITTED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                reviewed_native_steps=96000,
                physical_responses=physical,
                model=fit,
                validation=validation,
                all_prediction_gates_pass=all_predict,
                all_physical_response_guards_pass=all_stable,
                predictive_control_research_eligible=all_predict and all_stable,
                actual_motor_control_validated=False,
                football_skill_proven=False,
                goal_complete=False,
                new_native_steps=0,
                new_neural_calls=0,
                optimizer_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                new_native_steps=0,
                new_neural_calls=0,
                optimizer_steps=0,
                automatic_physics_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
