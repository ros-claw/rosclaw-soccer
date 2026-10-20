"""Bounded frozen-Foundation locomotion interventions with actual NN/native repeats."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace


def compare_response_row(actual, expected):
    a = {k: v for k, v in actual.items() if k != "native_hash"}
    b = {k: v for k, v in expected.items() if k != "native_hash"}
    if json.loads(json.dumps(a)) != json.loads(json.dumps(b)):
        raise ValueError("actual independent NN/native response report differs")


def execute(a, world, course, runtime):
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.full_entry_forecast import _tick
    from rosclaw_soccer.rsi.locomotion_response_protocol import summarize_response
    from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    check_pins(runtime)
    root = a.output / f"command{course.index:02d}"
    root.mkdir(parents=True, exist_ok=False)
    body, rows = None, []
    try:
        nav = G1SonicNavigation(
            a.model_root,
            "tracker",
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        body = G1TargetEpisode(
            world,
            world_hash=digest(a.world),
            kp=nav.backend.kp.copy(),
            kd=nav.backend.kd.copy(),
            course=course.ball_course,
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )
        for frame in range(300):
            q, v = body.canonical()
            cmd = course.command(frame)
            obs = TeamMotorObservation(
                agent_id="tracker",
                frame=frame,
                time_sec=float(body.data.time),
                intent="other",
                prospective_owner=False,
                qpos=tuple(float(x) for x in q),
                qvel=tuple(float(x) for x in v),
                target_position_m=tuple(float(x) for x in q[:3]),
                navigation_command=cmd,
            )
            row = _tick(nav, body, obs)
            row["navigation_command"] = np.asarray([cmd], dtype=np.float64)
            rows.append(row)
        if body.integrated_steps != 3000 or body.attempted_steps != 3000 or body.next_frame != 300:
            raise ValueError("whole actual H300 navigation episode required")
        raw = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        result = summarize_response(raw, course)
        np.savez_compressed(root / "native300.npz", **raw)
        check_pins(runtime)
        result.update(
            native_hash=digest(root / "native300.npz"),
            body_report=body.review(),
            native_steps=3000,
            new_optimizer_steps=0,
        )
        write_once(root / "complete.json", result)
        print(
            "LOCOMOTION_RESPONSE",
            course.index,
            course.split,
            result["pulse_last_second_mean_velocity_mps"],
            result["stable_height_and_upright"],
            flush=True,
        )
        return result
    except Exception as error:
        if rows:
            np.savez_compressed(
                root / "partial-native.npz",
                **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]},
            )
        write_once(
            root / "failure.json",
            dict(
                error=repr(error),
                integrated_steps=body.integrated_steps if body else 0,
                attempted_steps=body.attempted_steps if body else 0,
                automatic_retry_authorized=False,
            ),
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("qualify", "collect", "reproduce"), required=True)
    p.add_argument("--part", type=int, choices=(0, 1, 2), default=0)
    for k in ("review", "original", "inputs", "core-source", "output", "diagnostic"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "expected-head",
        "expected-review-hash",
        "review-unit",
        "diagnostic-unit",
        "expected-pilot-hash",
    ):
        p.add_argument("--" + k, required=True)
    for k in ("qualification", "producer"):
        p.add_argument("--" + k, type=Path)
        p.add_argument("--" + k + "-unit")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact frozen source required")
    wait_for_terminal_collection(a.review_unit)
    if digest(a.review / "complete.json") != a.expected_review_hash:
        raise ValueError("exact raw from0 review required")
    review = json.loads((a.review / "complete.json").read_text())
    if review["status"] != "INTENT_FROM0_CASE21_RAW_REPRODUCTION_AUTHENTICATED":
        raise ValueError("closed paired actual original evidence required")
    pins = dict(review["input_file_hashes"])
    pins[str(a.review / "complete.json")] = a.expected_review_hash
    wait_for_terminal_collection(a.diagnostic_unit, wait=True)
    diagnostic = json.loads((a.diagnostic / "complete.json").read_text())
    if (
        diagnostic["status"] != "CALIBRATED_ENTRY_INITIATION_FAILURE_DIAGNOSED"
        or diagnostic["source_head"] != a.expected_head
        or diagnostic["pilot_hash"] != a.expected_pilot_hash
        or len(diagnostic["results"]) != 3
        or any(r["source_handoff_frame"] is not None for r in diagnostic["results"])
    ):
        raise ValueError(
            "same-source exact failed-initiation diagnostic required before locomotion physics"
        )
    for k, h in diagnostic["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting diagnostic/history lineage")
        pins[k] = h
    pins[str(a.diagnostic / "complete.json")] = digest(a.diagnostic / "complete.json")
    if pins.get(str(a.inputs)) != digest(a.inputs) or pins.get(
        str(a.original / "complete.json")
    ) != digest(a.original / "complete.json"):
        raise ValueError("bound assets/original producer required")
    if a.mode != "qualify":
        if a.qualification is None or not a.qualification_unit:
            raise ValueError("explicit baseline qualification required")
        wait_for_terminal_collection(a.qualification_unit, wait=True)
        qual = json.loads((a.qualification / "complete.json").read_text())
        if (
            qual["status"] != "LOCOMOTION_RUNNER_ORIGINAL_FROM0_PARITY"
            or qual["source_head"] != a.expected_head
            or qual["review_hash"] != a.expected_review_hash
            or qual["native_steps"] != 6000
        ):
            raise ValueError("same-source actual original from0 qualification required")
        pins.update({str(f): digest(f) for f in a.qualification.rglob("*") if f.is_file()})
    producer = None
    if a.mode == "reproduce":
        if a.producer is None or not a.producer_unit:
            raise ValueError("explicit actual producer required")
        wait_for_terminal_collection(a.producer_unit, wait=True)
        producer = json.loads((a.producer / "complete.json").read_text())
        if (
            producer["status"] != "LOCOMOTION_RESPONSE_PART_COLLECTED"
            or producer["source_head"] != a.expected_head
            or producer["part"] != a.part
            or producer["native_steps"] != 15000
        ):
            raise ValueError("exact completed same part producer required")
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
    check_pins(pins)
    spec = json.loads(a.inputs.read_text())
    a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
    a.motion_csv = next(Path(k) for k in pins if k.endswith(".csv"))
    prior_path = next(Path(k) for k in pins if Path(k).name == "entry-affordance.json")
    runtime = {
        str(f): digest(f)
        for directory in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source)
        for f in directory.rglob("*.py")
    }
    for f in [a.world, a.motion_csv] + [
        a.model_root / n
        for n in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if pins.get(str(f)) != digest(f):
            raise ValueError("unchanged world/source/frozen Foundation required")
        runtime[str(f)] = digest(f)
    pins.update(runtime)
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.locomotion_response_protocol import (
        preflight_response_courses,
        response_courses,
    )

    courses = response_courses()
    bank_hash = preflight_response_courses(courses)
    world = mujoco.MjModel.from_binary_path(str(a.world))
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            mode=a.mode,
            part=a.part,
            bank_hash=bank_hash,
            input_file_hashes=pins,
            maximum_native_steps=6000 if a.mode == "qualify" else 15000,
            optimizer_steps=0,
            independent_initial_worlds=1,
            football_exam=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    try:
        if a.mode == "qualify":
            from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
            from rosclaw_soccer.training.categorical_motion_holdout import (
                categorical_motion_holdout_courses,
            )
            from rosclaw_soccer.training.receiving_course_preflight import (
                preflight_receiving_courses,
            )
            from scripts.rsi_motion_entry_episode import run_episode
            from scripts.rsi_pilot_motion_entry import compare_episode

            old = json.loads((a.original / "declaration.json").read_text())
            received = categorical_motion_holdout_courses()
            preflight_receiving_courses(received)
            saved = json.loads(prior_path.read_text())
            prior = SkillEntryAffordance(
                tuple(saved["ball_offset_xy_m"]),
                saved["historical_frame"],
                saved["historical_forward_speed_mps"],
                saved["historical_artifact_hash"],
            )
            if prior.contract_hash != saved["contract_hash"]:
                raise ValueError("original historical prior identity differs")
            poses = np.loadtxt(a.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
            for attempt in (0, 1):
                local = SimpleNamespace(**{**vars(a), "output": a.output / f"pass{attempt}"})
                row = run_episode(
                    local,
                    received[21],
                    world,
                    poses,
                    prior,
                    arm="original",
                    side="right",
                    law_index=0,
                    evidence=old["effective_motion_reference_evidence_hash"],
                    runtime=runtime,
                )
                compare_episode(local.output / "right-law0", a.original / "right-law0", row)
            status = "LOCOMOTION_RUNNER_ORIGINAL_FROM0_PARITY"
            results = []
        else:
            from scripts.rsi_probe_compact_source import exact_artifact

            results = []
            for course in courses[a.part :: 3]:
                row = execute(a, world, course, runtime)
                if producer is not None:
                    original = json.loads(
                        (a.producer / f"command{course.index:02d}" / "complete.json").read_text()
                    )
                    compare_response_row(row, original)
                    exact_artifact(
                        a.output / f"command{course.index:02d}" / "native300.npz",
                        a.producer / f"command{course.index:02d}" / "native300.npz",
                    )
                results.append(row)
            status = (
                "LOCOMOTION_RESPONSE_PART_REPRODUCED"
                if producer
                else "LOCOMOTION_RESPONSE_PART_COLLECTED"
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=status,
                source_head=a.expected_head,
                review_hash=a.expected_review_hash,
                bank_hash=bank_hash,
                part=a.part,
                results=results,
                native_steps=6000 if a.mode == "qualify" else 15000,
                optimizer_steps=0,
                learned_policy_improvement_proven=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                actual_partial_steps_in_episode_receipts=True,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
