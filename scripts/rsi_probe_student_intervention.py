"""Declare/probe/reproduce privileged teacher30 recovery on fixed consumed24.

No learning or model promotion. Teacher30 recovery is not a deployable actor or
evidence for original far/private Fresh football. Failed histories stay in24.
"""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path


def validate_warm_pair(a, pins, indices):
    """Read every warm field, not just a trusted-looking completion status."""
    import numpy as np

    from scripts.rsi_motor_student_rollout import exact_array

    for root in (a.warm, a.warm_repeat):
        c = json.loads((root / "complete.json").read_text())
        if c["indices"] != list(indices) or len(c["results"]) != 24:
            raise ValueError("all24 ordered warm rows required")
        for i, row in zip(indices, c["results"], strict=True):
            if row != dict(
                course_index=i,
                actual_student_native_steps=300,
                actual_attempted_steps=300,
                actual_student_prefix30_byte_exact=True,
                shadow_actual_history930_byte_exact=True,
                new_student_policy_calls=30,
                new_student_muscle_decoder_calls=10,
                shadow_encoder_calls=30,
                shadow_decoder_calls=30,
                recovery_executed=False,
                optimizer_steps=0,
            ):
                raise ValueError("actual complete warm NN/native count/checks required")
    for i in indices:
        names = [Path(k) for k in pins if k.endswith(f"/course{i:03d}/seed43/native.npz")]
        if len(names) != 1:
            raise ValueError("one prior seed43 physical endpoint required")
        with (
            np.load(names[0], allow_pickle=False) as old,
            np.load(a.warm / f"course{i:03d}/actual-student30.npz", allow_pickle=False) as first,
            np.load(
                a.warm_repeat / f"course{i:03d}/actual-student30.npz", allow_pickle=False
            ) as repeat,
        ):
            if set(old.files) != set(first.files) or set(old.files) != set(repeat.files):
                raise ValueError("all student30 numerical fields required")
            for k in old.files:
                exact_array(first[k], old[k][:30])
                exact_array(first[k], repeat[k])
        with (
            np.load(a.warm / f"course{i:03d}/shadow-labels30.npz", allow_pickle=False) as first,
            np.load(
                a.warm_repeat / f"course{i:03d}/shadow-labels30.npz", allow_pickle=False
            ) as repeat,
        ):
            shapes = dict(
                teacher_motion_token=(30, 64),
                teacher_advice_action=(30, 29),
                teacher_encoder_input=(30, 1247),
            )
            if set(first.files) != set(shapes) or set(repeat.files) != set(shapes):
                raise ValueError("all original teacher30 network label fields required")
            for k, shape in shapes.items():
                x = first[k]
                if x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all():
                    raise ValueError("finite complete original teacher30 label required")
                exact_array(x, repeat[k])


def compare_case(actual, expected, row):
    import numpy as np

    from scripts.rsi_motor_student_rollout import exact_array

    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != "artifact_hash"}
        return [strip(v) for v in value] if isinstance(value, (list, tuple)) else value

    if strip(json.loads(json.dumps(row))) != strip(
        json.loads((expected / "complete.json").read_text())
    ):
        raise ValueError("same-implementation recovery report/choices/score/native charges differ")
    names = ["actual-student30.npz", "shadow-labels30.npz"]
    if row["status"] == "STUDENT30_PRIVILEGED_RECOVERY_EXECUTED":
        names += [f"candidate{i}.npz" for i in range(9)]
        names += ["selected-actual270.npz", "selected-physical300.npz"]
    for name in names:
        with (
            np.load(actual / name, allow_pickle=False) as a,
            np.load(expected / name, allow_pickle=False) as b,
        ):
            if set(a.files) != set(b.files):
                raise ValueError("complete independent recovery array identities required")
            for k in a.files:
                exact_array(a[k], b[k])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("declare", "probe", "reproduce"), required=True)
    for k in ("inputs", "core-source", "warm", "warm-repeat", "near-plan", "motion-csv", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "warm-unit", "warm-repeat-unit", "expected-warm-source"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--part", type=int, choices=range(8))
    p.add_argument("--plan", type=Path)
    p.add_argument("--expected-plan-hash")
    p.add_argument("--producer", type=Path)
    p.add_argument("--producer-unit")
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
        raise ValueError("clean frozen source and unused output required")
    for root, unit, repeated in (
        (a.warm, a.warm_unit, False),
        (a.warm_repeat, a.warm_repeat_unit, True),
    ):
        wait_for_terminal_collection(unit, wait=True)
        c = json.loads((root / "complete.json").read_text())
        if (
            c["status"] != "STUDENT30_SHADOW_HISTORY_NATIVE_QUALIFIED"
            or c["source_head"] != a.expected_warm_source
            or c["new_native_steps"] != 7200
            or c["separate_process_same_implementation_repeat"] is not repeated
        ):
            raise ValueError("both actual qualified24 warm histories must really close")
    d = json.loads((a.warm / "declaration.json").read_text())
    rd = json.loads((a.warm_repeat / "declaration.json").read_text())
    pins = dict(d["input_file_hashes"])
    for k, h in rd["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting warm-history lineage")
        pins[k] = h
    for root in (a.warm, a.warm_repeat):
        pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
    if pins.get(str(a.inputs)) != digest(a.inputs) or pins.get(
        str(a.near_plan / "complete.json")
    ) != digest(a.near_plan / "complete.json"):
        raise ValueError("same frozen inputs/near roster required")
    spec = json.loads(a.inputs.read_text())
    near = json.loads((a.near_plan / "complete.json").read_text())
    a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
    from rosclaw_soccer.rsi.teacher_coverage_cohort import teacher_coverage_indices
    from rosclaw_soccer.training.categorical_motion_holdout import (
        categorical_motion_holdout_courses,
    )
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    indices = teacher_coverage_indices()
    courses = categorical_motion_holdout_courses()
    if (
        list(indices) != d["indices"]
        or list(indices) != rd["indices"]
        or preflight_receiving_courses(courses) != near["preflight_hash"]
    ):
        raise ValueError("all fixed24 consumed indices/full96 preflight must agree")
    roster = near["roster"]
    if [r["variant"] for r in roster] != ["anchor", "seed17", "seed29", "seed43"]:
        raise ValueError("existing fixed categorical roster required")
    assets = [a.world, a.motion_csv] + [
        a.model_root / n
        for n in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]
    # CSV was not used by warm-only qualification; require its already-bound
    # original teacher hash from the exact closed shadow20 input declaration.
    for f in assets:
        if pins.get(str(f)) != digest(f):
            raise ValueError("original world/NN/motion identities required: " + str(f))
    priors = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(priors) != 1:
        raise ValueError("one original affordance required")
    check_pins(pins)
    validate_warm_pair(a, pins, indices)
    runtime = {
        str(f): digest(f)
        for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source)
        for f in root.rglob("*.py")
    }
    runtime.update({str(f): digest(f) for f in assets})
    pins.update(runtime)
    if a.mode == "declare":
        if a.part is not None or a.producer or a.plan:
            raise ValueError("whole24 no-physics declaration required")
        a.output.mkdir(parents=True)
        write_once(
            a.output / "complete.json",
            dict(
                status="ACTUAL_STUDENT30_TEACHER_RECOVERY24_DECLARED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                indices=list(indices),
                cases=[asdict(courses[i]) for i in indices],
                source_student_seed=43,
                maximum_native_steps_per_eligible_case=27300,
                maximum_native_steps_for_producers_and_repeats=1310400,
                preintervention_history_veto_kept_in_denominator=True,
                original_task_horizon=300,
                original_score_frames=[20, 299],
                teacher_options=9,
                optimizer_steps=0,
                new_native_steps=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        return
    if (
        a.part is None
        or not a.plan
        or not a.expected_plan_hash
        or digest(a.plan / "complete.json") != a.expected_plan_hash
    ):
        raise ValueError("frozen whole24 declaration and explicit worker part required")
    plan = json.loads((a.plan / "complete.json").read_text())
    if (
        plan["status"] != "ACTUAL_STUDENT30_TEACHER_RECOVERY24_DECLARED"
        or plan["source_head"] != a.expected_head
        or plan["indices"] != list(indices)
        or plan["cases"] != json.loads(json.dumps([asdict(courses[i]) for i in indices]))
        or plan["input_file_hashes"] != pins
    ):
        raise ValueError("declared exact source/population/input commitment differs")
    pins[str(a.plan / "complete.json")] = a.expected_plan_hash
    selected_indices = list(indices[a.part :: 8])
    if a.mode == "reproduce":
        if not a.producer or not a.producer_unit:
            raise ValueError("explicit actual producer required")
        wait_for_terminal_collection(a.producer_unit, wait=True)
        c = json.loads((a.producer / "complete.json").read_text())
        if (
            c["status"] != "ACTUAL_STUDENT30_RECOVERY_SHARD_PRODUCED"
            or c["source_head"] != a.expected_head
            or c["indices"] != selected_indices
        ):
            raise ValueError("matching actual closed three-case producer required")
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
    elif a.producer or a.producer_unit:
        raise ValueError("probe is not a producer reuse path")
    check_pins(pins)
    a.output.mkdir(parents=True)
    from rosclaw_soccer.sim.contracts import hash_json

    declaration = dict(
        source_head=a.expected_head,
        input_file_hashes=pins,
        runtime_file_hashes=runtime,
        indices=selected_indices,
        part=a.part,
        mode=a.mode,
        source_student_seed=43,
        maximum_native_steps=27300 * len(selected_indices),
        optimizer_steps=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    evidence = hash_json(declaration)
    if a.producer:
        evidence = json.loads((a.producer / "declaration.json").read_text())[
            "teacher_evidence_hash"
        ]
    declaration["teacher_evidence_hash"] = evidence
    write_once(a.output / "declaration.json", declaration)
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.categorical_motion import create_categorical_motion
    from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree
    from scripts.rsi_online_kinematic_motion import load_model
    from scripts.rsi_student_intervention_course import run_course

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parent = load_model(spec, Path(roster[0]["path"])).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    student = create_categorical_motion(parent)
    cp = torch.load(roster[3]["path"], map_location="cpu", weights_only=True)
    if (
        cp["model_kind"] != "categorical_fsq32_motion64"
        or cp["anchor_hash"] != roster[0]["hash"]
        or cp["world_hash"] != digest(a.world)
    ):
        raise ValueError("fixed categorical43 world/parent identities required")
    student.load_state_dict(cp["state_dict"], strict=True)
    for k, v in parent.state_dict().items():
        exact_tensor_tree(v, before[k], path=k)
    student.eval().requires_grad_(False)
    muscle = G1SonicRunupController(
        a.model_root, G1SonicRunupConfig(model_variant="low_latency"), inference_threads=1
    )
    world = mujoco.MjModel.from_binary_path(str(a.world))
    poses = np.loadtxt(a.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
    stored = json.loads(priors[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("historical entry contract changed")
    rows = []
    try:
        for i in selected_indices:
            paths = [Path(k) for k in pins if k.endswith(f"/course{i:03d}/seed43/native.npz")]
            if len(paths) != 1:
                raise ValueError("one old seed43 raw trace per fixed course required")
            with np.load(paths[0], allow_pickle=False) as z:
                expected = {k: z[k].copy() for k in z.files}
            row = run_course(
                a, courses[i], world, poses, prior, student, muscle, expected, evidence, runtime
            )
            if a.producer:
                compare_case(a.output / f"course{i:03d}", a.producer / f"course{i:03d}", row)
            rows.append(row)
            write_once(a.output / f"progress{i:03d}.json", row)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="ACTUAL_STUDENT30_RECOVERY_SHARD_REPRODUCED"
                if a.producer
                else "ACTUAL_STUDENT30_RECOVERY_SHARD_PRODUCED",
                source_head=a.expected_head,
                indices=selected_indices,
                results=rows,
                new_native_steps=sum(r["new_native_steps"] for r in rows),
                same_implementation_neural_native_repeat=bool(a.producer),
                optimizer_steps=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                completed_cases=len(rows),
                partial_physics_charge_requires_case_receipts=True,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
