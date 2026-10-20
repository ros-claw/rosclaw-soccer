"""Await all six real terminal jobs, then audit all thirty native recordings."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("launch", "output"):
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
        raise ValueError("clean pinned read-only reviewer and exact launch required")
    launch = json.loads(a.launch.read_text())
    source = Path(launch["source"])
    if (
        subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"])
        or subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != launch["source_head"]
        or len(launch["jobs"]) != 6
        or launch["maximum_native_steps"] != 90000
        or launch["maximum_neural_calls"] != 9000
    ):
        raise ValueError("complete immutable launched six-job physical protocol required")
    for directory in ("src", "scripts"):
        for f in (source / directory).rglob("*.py"):
            if digest(f) != digest(Path.cwd() / f.relative_to(source)):
                raise ValueError("read-only reviewer must preserve all existing worker math")
    pins = {str(a.launch): a.expected_launch_hash}
    pins.update({str(f): digest(f) for d in ("src", "scripts") for f in Path(d).rglob("*.py")})
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            launch_hash=a.expected_launch_hash,
            worker_source_head=launch["source_head"],
            new_native_steps=0,
            new_neural_calls=0,
            new_optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        parts = {}
        for j in launch["jobs"]:
            wait_for_terminal_collection(j["unit"], wait=True)
            root = Path(j["root"])
            c = json.loads((root / "complete.json").read_text())
            d = json.loads((root / "declaration.json").read_text())
            repeated = j["mode"] == "reproduce"
            expected = (
                "G1_VELOCITY_NATIVE_PART_REPRODUCED"
                if repeated
                else "G1_VELOCITY_NATIVE_PART_COLLECTED"
            )
            key = (j["part"], repeated)
            if (
                key in parts
                or j["part"] not in (0, 1, 2)
                or c["status"] != expected
                or c["source_head"] != launch["source_head"]
                or d["source_head"] != launch["source_head"]
                or c["part"] != j["part"]
                or d["part"] != j["part"]
                or c["native_steps"] != 15000
                or c["neural_calls"] != 1500
                or c["new_optimizer_steps"] != 0
                or len(c["results"]) != 5
            ):
                raise ValueError("all exact complete native/NN parts required")
            for path, h in d["input_file_hashes"].items():
                if path in pins and pins[path] != h:
                    raise ValueError("conflicting input commitments")
                pins[path] = h
            pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
            parts[key] = (root, c, d)
        check_pins(pins)
        import mujoco

        from rosclaw_soccer.physics.g1_velocity_observation import G1VelocityObservation
        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.locomotion_response_protocol import (
            preflight_response_courses,
            response_courses,
        )
        from rosclaw_soccer.rsi.velocity_response_bank_review import review_episode, summarize_bank
        from scripts.rsi_collect_velocity_policy_response import compare_velocity_result
        from scripts.rsi_probe_compact_source import exact_artifact, load_arrays

        courses = response_courses()
        bank = preflight_response_courses(courses)
        rows = []
        for part in range(3):
            root, c, d = parts[part, False]
            repeat, rc, rd = parts[part, True]
            for item in (d, rd):
                if (
                    item["bank_hash"] != bank
                    or item["maximum_native_steps"] != 15000
                    or item["maximum_neural_calls"] != 1500
                    or item["optimizer_steps"] != 0
                ):
                    raise ValueError("unaltered full bank and budget required")
            config = Path(d["config_file"])
            worldfile = Path(d["world_file"])
            params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
            sensor = G1VelocityObservation(
                mujoco.MjModel.from_binary_path(str(worldfile)),
                expected_world_hash=pins[str(worldfile)],
            )
            for course, row, rerow in zip(
                courses[part::3], c["results"], rc["results"], strict=True
            ):
                folder = f"command{course.index:02d}"
                compare_velocity_result(row, rerow)
                exact_artifact(root / folder / "native300.npz", repeat / folder / "native300.npz")
                for directory, receipt in ((root, row), (repeat, rerow)):
                    path = directory / folder / "native300.npz"
                    if (
                        digest(path) != receipt["native_hash"]
                        or receipt != json.loads((directory / folder / "complete.json").read_text())
                        or receipt["parameter_contract_hash"] != params.contract_hash
                        or receipt["policy_contract_hash"] != c["policy_contract_hash"]
                    ):
                        raise ValueError("whole exact episode/hash/parameter identity required")
                    raw = load_arrays(path)
                    result = review_episode(raw, receipt, params, sensor, course)
                    if directory == root:
                        rows.append(result)
                print("VELOCITY_RESPONSE_RAW_AUDITED", course.index, flush=True)
        summary = summarize_bank(rows)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="G1_VELOCITY_NATIVE_90K_BANK_AUDITED_NOT_FOOTBALL",
                source_head=a.expected_head,
                input_file_hashes=pins,
                reviewed_native_steps=90000,
                reviewed_neural_calls=9000,
                new_native_steps=0,
                new_neural_calls=0,
                all_30_typed_recordings_reviewed=True,
                all_15_independent_nn_native_repeats_exact=True,
                source_gain_and_initial_pose_differ_from_sonic=True,
                publication_license_cleared=False,
                results=sorted(rows, key=lambda r: r["course_index"]),
                **summary,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                automatic_retry_authorized=False,
                new_native_steps=0,
                new_neural_calls=0,
            ),
        )
        raise


if __name__ == "__main__":
    main()
