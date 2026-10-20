"""One fit from two past pairs, then three from0 native intents and NN repeats.

No parameter sweep, gate relaxation, body snapshot adoption, or online reward
feedback. Law4 is withheld from the fit, not an independent world holdout.
"""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def paired_sample(world, original, compact, law):
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_motor_student_rollout import exact_array
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_probe_compact_source import load_arrays

    qi = world.jnt_qposadr[[world.joint(n).id for n in G1_DDS_JOINT_NAMES]]
    body = world.body("right_ankle_roll_link").id
    spheres = np.flatnonzero(
        (world.geom_bodyid == body)
        & (world.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
        & (world.geom_contype != 0)
    )
    if len(spheres) != 4 or not np.all(world.geom_size[spheres, 0] == 0.005):
        raise ValueError("unchanged four actual toe support landmarks required")
    toes = spheres[world.geom_pos[spheres, 0] == world.geom_pos[spheres, 0].max()]
    if len(toes) != 2:
        raise ValueError("two original front support spheres required")
    private = mujoco.MjData(world)
    points, samples, states = [], [], []
    for root, phase in ((original, 140), (compact, 65)):
        folder = root / f"right-law{law}"
        row = json.loads((folder / "complete.json").read_text())
        raw = load_arrays(folder / "native300.npz")
        handoff = row["report"]["handoff_frame"]
        if type(handoff) is not int or not 30 <= handoff <= 90:
            raise ValueError("actual source handoff required in each fit pair")
        frame = handoff + phase
        if (
            row["reference_side"] != "right"
            or row["law_index"] != law
            or row["native_hash"] != digest(folder / "native300.npz")
            or np.any(raw["ball_contact_force_by_body_n"][: frame + 1, :, 1:] > 1)
            or row["score"]["first_foot_contact_global_frame"] is None
            or frame >= row["score"]["first_foot_contact_global_frame"]
        ):
            raise ValueError("fit landmarks must precede all robot-ball contact")
        q = raw["canonical_qpos"][frame]
        private.qpos[:7], private.qpos[qi] = q[:7], q[7:36]
        mujoco.mj_kinematics(world, private)
        # Both source-only arms have the exact same handoff physical frame;
        # fixed world+X task axes agree with the entry command's XY axes.
        points.append(
            private.geom_xpos[toes].mean(axis=0).copy() - raw["canonical_qpos"][handoff, :3]
        )
        states.append(raw["canonical_qpos"][handoff])
        samples.append(
            dict(
                raw_hash=row["native_hash"], global_frame=frame, phase=phase, handoff_frame=handoff
            )
        )
    exact_array(states[0], states[1])
    record = dict(
        law=law,
        samples=samples,
        original_toe_displacement_m=points[0].tolist(),
        compact_toe_displacement_m=points[1].tolist(),
        delta_xy_m=(points[1] - points[0])[:2].tolist(),
    )
    return (points[1] - points[0])[:2], hash_json(record), record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("review", "launch", "inputs", "motion-csv", "prior", "core-source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "expected-review-hash", "review-unit"):
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
    ):
        raise ValueError("unused output and clean frozen calibrated pilot source required")
    wait_for_terminal_collection(a.review_unit)
    if digest(a.review / "complete.json") != a.expected_review_hash:
        raise ValueError("exact closed from0 paired pilot raw audit required")
    review = json.loads((a.review / "complete.json").read_text())
    if review["status"] != "INTENT_FROM0_CASE21_RAW_REPRODUCTION_AUTHENTICATED":
        raise ValueError("all three source-only/geometry arms must be audited")
    pins = dict(review["input_file_hashes"])
    pins[str(a.review / "complete.json")] = a.expected_review_hash
    for f in (a.launch, a.inputs, a.motion_csv, a.prior):
        if pins.get(str(f)) != digest(f):
            raise ValueError("historical calibrated fit input commitment differs")
    check_pins(pins)
    spec = json.loads(a.inputs.read_text())
    a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
    launch = json.loads(a.launch.read_text())
    roots = {j["arm"]: Path(j["root"]) for j in launch["jobs"] if j["mode"] == "produce"}
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
            raise ValueError("unchanged world/ONNX/source motion required")
        runtime[str(f)] = digest(f)
    pins.update(runtime)
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.motion_entry_profile import build_motion_entry
    from rosclaw_soccer.rsi.native_entry_calibration import bind_native_entry, fit_displacements
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.categorical_motion_holdout import (
        categorical_motion_holdout_courses,
    )
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_motion_entry_episode import run_episode
    from scripts.rsi_pilot_motion_entry import compare_episode
    from scripts.rsi_probe_compact_source import load_arrays
    from scripts.rsi_review_motion_entry_pilot import review_episode

    courses = categorical_motion_holdout_courses()
    preflight_receiving_courses(courses)
    course = courses[21]
    old = json.loads((roots["original"] / "declaration.json").read_text())
    if json.loads(json.dumps(asdict(course))) != old["course"]:
        raise ValueError("same consumed course21; no initial-world change permitted")
    world = mujoco.MjModel.from_binary_path(str(a.world))
    poses = np.loadtxt(a.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
    stored = json.loads(a.prior.read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("historical parent prior identity differs")
    binding, parent, source = build_motion_entry(
        poses,
        prior,
        world_hash=digest(a.world),
        motion_artifact_hash=digest(a.motion_csv),
        reference_side="right",
        source_window=(466, 767),
        transport=False,
    )
    pairs = [
        paired_sample(world, roots["original"], roots["compact_unadapted"], law) for law in (0, 3)
    ]
    fit = fit_displacements(
        np.stack([x[0] for x in pairs]),
        configuration_hash=binding.configuration.contract_hash,
        parent_affordance_hash=parent.contract_hash,
        audit_hash=a.expected_review_hash,
        sample_hashes=(pairs[0][1], pairs[1][1]),
    )
    bound, entry = bind_native_entry(binding, parent, fit)
    bound.validate(
        source, entry, world_hash=digest(a.world), source_window=(466, 767), reference_side="right"
    )
    reference_evidence = hash_json(
        dict(
            calibration_hash=fit.contract_hash,
            motion_entry_binding_hash=bound.contract_hash,
            source_motion_artifact_hash=digest(a.motion_csv),
        )
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            fit=fit.report(),
            fit_hash=fit.contract_hash,
            effective_motion_reference_evidence_hash=reference_evidence,
            paired_samples=[x[2] for x in pairs],
            entry_ball_offset_xy_m=entry.ball_offset_xy_m,
            maximum_native_steps=18000,
            input_file_hashes=pins,
            optimizer_steps=0,
            independent_contexts=1,
            paired_training_laws=[0, 3],
            withheld_law=4,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    rows, repeats = [], []
    try:
        for mode, target in (("produce", rows), ("repeat", repeats)):
            for law in (0, 3, 4):
                local = SimpleNamespace(**{**vars(a), "output": a.output / mode})
                target.append(
                    run_episode(
                        local,
                        course,
                        world,
                        poses,
                        prior,
                        arm="compact_unadapted",
                        side="right",
                        law_index=law,
                        evidence=reference_evidence,
                        runtime=runtime,
                        entry_calibration=fit,
                    )
                )
                if mode == "repeat":
                    compare_episode(
                        local.output / f"right-law{law}",
                        a.output / "produce" / f"right-law{law}",
                        target[-1],
                    )
        summary = [
            review_episode(
                world,
                load_arrays(a.output / "produce" / f"right-law{r['law_index']}" / "native300.npz"),
                r,
            )
            for r in rows
        ]
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_ENTRY_FIT_THREE_INTENTS_REPRODUCED",
                source_head=a.expected_head,
                fit_hash=fit.contract_hash,
                rows=summary,
                results=rows,
                independent_contexts=1,
                native_steps=18000,
                optimizer_steps=0,
                repeat_implementation="same_code_full_neural_native_second_execution_not_independent_algorithm",
                new_statistical_fit_coefficients=2,
                neural_policy_improvement_proven=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                completed_episodes=len(rows) + len(repeats),
                partial_native_counts_in_episode_receipts=True,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
