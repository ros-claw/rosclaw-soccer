"""Three-arm, intent-known-from0 case21 pilot; SIM-only teacher development.

Every intent gets an independent native body, full NN inference and all300
frames. This is not the earlier student30 protocol or an autonomous best-of6
actor. Reference side is declared before navigation, not inferred from outcomes.
"""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path


def compare_episode(actual, expected, row):
    from scripts.rsi_probe_compact_source import exact_artifact

    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != "native_hash"}
        return [strip(v) for v in value] if isinstance(value, (list, tuple)) else value

    if strip(json.loads(json.dumps(row))) != strip(
        json.loads((expected / "complete.json").read_text())
    ):
        raise ValueError("intent episode report/score/causal/accounting repeat differs")
    exact_artifact(actual / "native300.npz", expected / "native300.npz")
    exact_artifact(actual / "own-prefix20.npz", expected / "own-prefix20.npz")


def compare_original_right(actual, old_case, row):
    import numpy as np

    from scripts.rsi_motor_student_rollout import exact_array

    index = {0: 0, 3: 2, 4: 4}[row["law_index"]]
    old = json.loads((old_case / "complete.json").read_text())
    candidate = old["candidates"][index]
    if (
        row["reference_side"] != "right"
        or row["source_window"] != [286, 767]
        or json.loads(json.dumps(row["report"])) != candidate["report"]
        or row["score"] != candidate["score"]
        or row["new_native_steps"] != 3000
    ):
        raise ValueError("original RHS neural/native/report parity differs")
    with (
        np.load(actual / "native300.npz", allow_pickle=False) as a,
        np.load(old_case / "own-prefix20.npz", allow_pickle=False) as p,
        np.load(old_case / f"candidate{index}.npz", allow_pickle=False) as f,
    ):
        if set(a.files) != set(p.files) or set(a.files) != set(f.files):
            raise ValueError("all18 old prefix20/future280 fields required")
        for k in a.files:
            exact_array(a[k], np.concatenate((p[k], f[k])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("declare", "produce", "reproduce"), required=True)
    p.add_argument("--arm", choices=("original", "compact_unadapted", "compact_transported"))
    for k in (
        "geometry",
        "teacher-review",
        "old-case",
        "inputs",
        "core-source",
        "motion-csv",
        "prior",
        "output",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "expected-head",
        "expected-geometry-hash",
        "geometry-unit",
        "expected-teacher-review-hash",
        "teacher-review-unit",
    ):
        p.add_argument("--" + k, required=True)
    for k in ("plan", "baseline", "producer"):
        p.add_argument("--" + k, type=Path)
    for k in ("expected-plan-hash", "baseline-unit", "producer-unit"):
        p.add_argument("--" + k)
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
        raise ValueError("clean frozen pilot source and unused output required")
    wait_for_terminal_collection(a.geometry_unit)
    wait_for_terminal_collection(a.teacher_review_unit)
    if (
        digest(a.geometry / "complete.json") != a.expected_geometry_hash
        or digest(a.teacher_review / "complete.json") != a.expected_teacher_review_hash
    ):
        raise ValueError("exact geometry and original teacher raw audit required")
    g = json.loads((a.geometry / "complete.json").read_text())
    t = json.loads((a.teacher_review / "complete.json").read_text())
    if (
        g["status"] != "ENTRY_FRAME_POINT_TRANSPORT_GEOMETRY_QUALIFIED_NOT_DYNAMIC"
        or t["status"] != "PRIVILEGED_TEACHER_FIXED24_REPRODUCTION_AUTHENTICATED"
    ):
        raise ValueError("qualified geometry is not already dynamic admission")
    pins = dict(g["input_file_hashes"])
    for k, h in t["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting original source/assets lineage")
        pins[k] = h
    for f in (a.geometry / "complete.json", a.teacher_review / "complete.json"):
        pins[str(f)] = digest(f)
    for f in (
        a.inputs,
        a.motion_csv,
        a.prior,
        a.old_case / "complete.json",
        a.old_case.parent / "declaration.json",
    ):
        if pins.get(str(f)) != digest(f):
            raise ValueError("unchanged asset/past case binding required: " + str(f))
    check_pins(pins)
    spec = json.loads(a.inputs.read_text())
    a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
    from rosclaw_soccer.training.categorical_motion_holdout import (
        categorical_motion_holdout_courses,
    )
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = categorical_motion_holdout_courses()
    preflight_receiving_courses(courses)
    course = courses[21]
    old = json.loads((a.old_case / "complete.json").read_text())
    if old["course"] != json.loads(json.dumps(asdict(course))):
        raise ValueError("fixed consumed case21 exactly as declared required")
    runtime = {
        str(f): digest(f)
        for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source)
        for f in root.rglob("*.py")
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
            raise ValueError("unchanged physical world/ONNX/motion asset required")
        runtime[str(f)] = digest(f)
    pins.update(runtime)
    import numpy as np

    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.motion_entry_profile import build_motion_entry
    from scripts.rsi_motion_entry_episode import ARMS, run_episode

    poses = np.loadtxt(a.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
    stored = json.loads(a.prior.read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("unchanged past entry contract required")
    profiles = {}
    for arm, (window, transport) in ARMS.items():
        for side in ("right", "left"):
            b, e, _ = build_motion_entry(
                poses,
                prior,
                world_hash=digest(a.world),
                motion_artifact_hash=digest(a.motion_csv),
                reference_side=side,
                source_window=window,
                transport=transport,
            )
            profiles[arm + ":" + side] = dict(
                binding=b.report(),
                binding_hash=b.contract_hash,
                offset_xy_m=list(e.ball_offset_xy_m),
            )
            if transport:
                np.testing.assert_allclose(
                    e.ball_offset_xy_m,
                    g[side]["report"]["transported_ball_offset_xy_m"],
                    atol=1e-12,
                    rtol=0,
                )
    protocol = dict(
        source_head=a.expected_head,
        input_file_hashes=pins,
        course=asdict(course),
        profiles=profiles,
        arms=list(ARMS),
        laws=[0, 3, 4],
        reference_sides=["right", "left"],
        maximum_native_steps_per_arm=18000,
        maximum_total_native_steps_for_producers_and_repeats=108000,
        profile_known_before_frame0=True,
        prior_student43_not_executed=True,
        original_task_horizon=300,
        original_score_frames=[20, 299],
        optimizer_steps=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if a.mode == "declare":
        if a.arm or a.plan or a.baseline or a.producer:
            raise ValueError("whole three-arm declaration only")
        a.output.mkdir(parents=True)
        write_once(
            a.output / "complete.json",
            dict(status="INTENT_FROM0_CASE21_PILOT_DECLARED", **protocol, new_native_steps=0),
        )
        return
    if (
        not a.arm
        or not a.plan
        or not a.expected_plan_hash
        or digest(a.plan / "complete.json") != a.expected_plan_hash
        or json.loads((a.plan / "complete.json").read_text())
        != json.loads(
            json.dumps(
                dict(status="INTENT_FROM0_CASE21_PILOT_DECLARED", **protocol, new_native_steps=0)
            )
        )
    ):
        raise ValueError("exact immutable pilot/source declaration required")
    pins[str(a.plan / "complete.json")] = a.expected_plan_hash
    if a.arm != "original":
        if not a.baseline or not a.baseline_unit:
            raise ValueError("original from0 right-side parity producer required")
        wait_for_terminal_collection(a.baseline_unit, wait=True)
        c = json.loads((a.baseline / "complete.json").read_text())
        if (
            c["status"] != "INTENT_FROM0_ARM_PRODUCED"
            or c["arm"] != "original"
            or c["source_head"] != a.expected_head
            or c["all_three_right_episodes_byte_exact_to_old"] is not True
        ):
            raise ValueError("baseline3RHS numeric parity must actually pass")
        pins[str(a.baseline / "complete.json")] = digest(a.baseline / "complete.json")
    if a.mode == "reproduce":
        if not a.producer or not a.producer_unit:
            raise ValueError("explicit actual producer required")
        wait_for_terminal_collection(a.producer_unit, wait=True)
        c = json.loads((a.producer / "complete.json").read_text())
        if (
            c["status"] != "INTENT_FROM0_ARM_PRODUCED"
            or c["arm"] != a.arm
            or c["source_head"] != a.expected_head
            or c["new_native_steps"] != 18000
        ):
            raise ValueError("same arm/source six-episode producer required")
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
    check_pins(pins)
    a.output.mkdir(parents=True)
    from rosclaw_soccer.sim.contracts import hash_json

    declaration = dict(
        source_head=a.expected_head,
        input_file_hashes=pins,
        runtime_file_hashes=runtime,
        arm=a.arm,
        mode=a.mode,
        course=asdict(course),
        profiles=profiles,
        maximum_native_steps=18000,
        new_optimizer_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    generated = hash_json(declaration)
    effective = generated
    if a.arm == "original":
        effective = json.loads((a.old_case.parent / "declaration.json").read_text())[
            "teacher_evidence_hash"
        ]
    elif a.producer:
        effective = json.loads((a.producer / "declaration.json").read_text())[
            "effective_motion_reference_evidence_hash"
        ]
    declaration["generated_declaration_evidence_hash"] = generated
    declaration["effective_motion_reference_evidence_hash"] = effective
    write_once(a.output / "declaration.json", declaration)
    import mujoco

    world = mujoco.MjModel.from_binary_path(str(a.world))
    rows = []
    try:
        for side in ("right", "left"):
            for law in (0, 3, 4):
                row = run_episode(
                    a,
                    course,
                    world,
                    poses,
                    prior,
                    arm=a.arm,
                    side=side,
                    law_index=law,
                    evidence=effective,
                    runtime=runtime,
                )
                folder = a.output / f"{side}-law{law}"
                if a.arm == "original" and side == "right":
                    compare_original_right(folder, a.old_case, row)
                if a.producer:
                    compare_episode(folder, a.producer / folder.name, row)
                rows.append(row)
                write_once(a.output / f"progress-{side}-law{law}.json", row)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="INTENT_FROM0_ARM_PRODUCED"
                if a.mode == "produce"
                else "INTENT_FROM0_ARM_REPRODUCED",
                source_head=a.expected_head,
                arm=a.arm,
                results=rows,
                new_native_steps=sum(r["new_native_steps"] for r in rows),
                all_three_right_episodes_byte_exact_to_old=a.arm == "original",
                optimizer_steps=0,
                candidate_outcomes_not_online_actor_inputs=True,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                completed_episodes=len(rows),
                partial_charge_requires_episode_receipts=True,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
