"""No-search feedback ablation of the closed public40 constant-program oracle."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-audit", "experimental-approach-config", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-audit-hash", "experimental-approach-config-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shard", type=int, choices=range(4), required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.walk_kick_teacher_plan import search_course_bank
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    courses = search_course_bank(expanded_public=True)
    bank_hash = preflight_receiving_courses(courses)
    path = a.source_audit
    if a.output.exists() or path.is_symlink() or digest(path) != a.source_audit_hash:
        raise ValueError("unused output and exact completed source audit required")
    source = json.loads(path.read_text())
    if (
        source["status"]
        != "PUBLIC40_CEM_ALL760_RAW_LABELS_AND_REEXECUTION_AUTHENTICATED_NOT_PROMOTED"
        or source["course_bank_hash"] != bank_hash
        or [r["course_index"] for r in source["results"]] != list(range(40))
        or source["baseline_uses_original_strict_codec"] is not True
        or source["promotion_authorized"] is not False
    ):
        raise ValueError("complete public40 constant-program authority study required")
    pins = dict(source["input_file_hashes"])
    pins[str(path)] = a.source_audit_hash
    fast_assets, point_pins = admit_heading_evidence(
        a.experimental_approach_config,
        a.experimental_approach_config_hash,
    )
    for key, value in point_pins.items():
        if key in pins and pins[key] != value:
            raise ValueError("conflicting experimental locomotion identity")
        pins[key] = value
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique bound physical asset required: " + suffix)
        return paths[0]

    # Join selected records by authenticated course state, never first path hit.
    selected = {}
    for index, c in enumerate(courses):
        summary = source["results"][index]
        suffix = f"/course-{index:02d}/{summary['selected']}.json"
        matches = []
        for key in pins:
            if key.endswith(suffix):
                row = json.loads(Path(key).read_text())
                if row["course"] == list(c.course):
                    matches.append((Path(key), row))
        if len(matches) != 1:
            raise ValueError("unique full-state matched original teacher required")
        rp, row = matches[0]
        artifact = rp.parent / row["artifact"]["path"]
        if (
            rp.is_symlink()
            or artifact.parent != rp.parent
            or artifact.is_symlink()
            or (digest(artifact) != row["artifact"]["hash"])
        ):
            raise ValueError("exact original selected physical arrays required")
        selected[index] = (row, artifact)
    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.sim.contracts import hash_json

    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    assets = dict(
        velocity_model=vm,
        velocity_config=vc,
        velocity_model_hash=pins[str(vm)],
        velocity_config_hash=pins[str(vc)],
    )
    assets.update(fast_assets)
    types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(),
            prefixes=("a/",),
        )
    )
    arms = ("constant", "velocity-feedback", "velocity-and-phase-feedback")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.public40_measured_velocity_phase_feedback_ablation.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            course_bank_hash=bank_hash,
            arms=arms,
            shard=a.shard,
            assigned_indices=list(range(a.shard, 40, 4)),
            search_or_refit=False,
            no_new_neural_gradients=True,
            fixed_longitudinal_gain=1.5,
            longitudinal_command="clip(ball_vx+1.5*(gap-entrygap+.20),0,cruise)",
            admission_runway_m=0.20,
            lateral_gain=2.0,
            lateral_ball_velocity_feedforward=True,
            phase_feedback="measured q/dq nearest reference180..250 at actual handoff",
            same_selected_parameters_and_velocity_domain_as_original=True,
            new_course_distribution=False,
            public_consumed_optimization_states=True,
            per_arm_fresh_nn_reexecution=True,
            worlds=60,
            native_step_ceiling=216000,
            horizon_controls=360,
            formal_horizon_controls=300,
            optimizer_updates=0,
            physical_teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    results, steps = [], 0
    try:
        for index in range(a.shard, 40, 4):
            c = courses[index]
            original, original_artifact = selected[index]
            root = a.output / f"course-{index:02d}"
            root.mkdir()
            records, originals = [], {}
            for arm in arms:
                params = dict(original["parameters"])
                if arm != "constant":
                    params["approach_feedback"] = True
                if arm == "velocity-and-phase-feedback":
                    params["entry_phase_feedback"] = True
                owned = dict(assets, experimental_fast=original["experimental_fast_approach"])
                for repeat in range(2):
                    row, arrays = evaluate_episode(
                        model=model,
                        assets=owned,
                        policy_types=types,
                        course=c.course,
                        parameters=params,
                        hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
                    )
                    steps += row["native_steps"]
                    relabel = relabel_arrays(
                        arrays,
                        controls=row["controls"],
                        ball_address=row["ball_qpos_address"],
                        root_address=row["robot_qpos_address"],
                    )
                    if hash_json(relabel) != hash_json(row["outcome"]):
                        raise ValueError("native outcome reconstruction differs")
                    reference = original_artifact if arm == "constant" else None
                    if repeat:
                        reference = originals[arm]
                    exact = None
                    if reference is not None:
                        with np.load(reference) as old:
                            exact = {k: bool(np.array_equal(v, old[k])) for k, v in arrays.items()}
                        if not all(exact.values()):
                            raise ValueError(
                                "original constant or fresh NN physical reexecution differs"
                            )
                    artifact = root / f"{arm}-{repeat}.npz"
                    np.savez_compressed(artifact, **arrays)
                    if not repeat:
                        originals[arm] = artifact
                    row.update(
                        arm=arm,
                        repeat=repeat,
                        course_index=index,
                        exact_reexecution=exact,
                        artifact=dict(path=artifact.name, hash=digest(artifact)),
                    )
                    records.append(row)
                    write_once(root / f"{arm}-{repeat}.json", row)
            write_once(
                root / "complete.json",
                dict(course_index=index, records=records, search_or_refit=False),
            )
            results.append(
                dict(
                    course_index=index,
                    source_hash=digest(root / "complete.json"),
                    outcomes={
                        r["arm"]: dict(outcome=r["outcome"], healthy=r["healthy_full_horizon"])
                        for r in records
                        if not r["repeat"]
                    },
                )
            )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PUBLIC40_FEEDBACK_ABLATION_SHARD_MEASURED_NOT_LEARNED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=results,
                course_bank_hash=bank_hash,
                worlds=60,
                native_steps=steps,
                optimizer_updates=0,
                physical_teacher_qualified=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                native_steps=steps,
                completed_courses=len(results),
                no_automatic_physics_retry=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
