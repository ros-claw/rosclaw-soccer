"""Freeze a post-pilot teacher route, test new40 states without further search."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("slow-audit", "fast-audit", "experimental-approach-config", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "slow-audit-hash",
        "fast-audit-hash",
        "experimental-approach-config-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shard", type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.walk_kick_hybrid_teacher import (
        compose_teacher,
        expanded_validation_courses,
        select_teacher,
    )
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused validation output required")
    courses = expanded_validation_courses()
    bank_hash = preflight_receiving_courses(courses)
    reports, pins = [], {}
    for path, expected in ((a.slow_audit, a.slow_audit_hash), (a.fast_audit, a.fast_audit_hash)):
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact completed teacher audit required")
        report = json.loads(path.read_text())
        for key, value in {**report["input_file_hashes"], str(path): expected}.items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting source asset ancestry")
            pins[key] = value
        reports.append(report)
    table = compose_teacher(*reports)
    fast_assets, point_pins = admit_heading_evidence(
        a.experimental_approach_config,
        a.experimental_approach_config_hash,
    )
    for key, value in point_pins.items():
        if key in pins and pins[key] != value:
            raise ValueError("conflicting motion point ancestry")
        pins[key] = value
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique actual policy/scene required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.sim.contracts import hash_json

    model_path = one("/velocity/v0/exported/policy.onnx")
    config_path = one("/velocity/v0/params/deploy.yaml")
    assets = dict(
        velocity_model=model_path,
        velocity_config=config_path,
        velocity_model_hash=pins[str(model_path)],
        velocity_config_hash=pins[str(config_path)],
    )
    types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(),
            prefixes=("a/",),
        )
    )
    baseline = dict(speed_mps=1.0, entry_gap_m=0.9, reference_frame=None, root_lateral_offset_m=0.0)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.public40_frozen_context_hybrid_teacher_validation.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            frozen_teacher=table,
            course_bank_hash=bank_hash,
            courses=[dict(index=i, course=c.course, seed=c.seed) for i, c in enumerate(courses)],
            shard=a.shard,
            assigned_indices=list(range(a.shard, 40, 2)),
            no_search_or_refit=True,
            baseline=baseline,
            source16_training_contexts=16,
            new_public_validation_initial_states=40,
            independent_scenarios_claimed=40,
            sealed_fresh_exam=False,
            all_baseline_and_teacher_fresh_nn_reexecuted=True,
            worlds=80,
            native_step_ceiling=288000,
            horizon_controls=360,
            formal_horizon_controls=300,
            minimum_passes_for_teacher_review=32,
            minimum_net_gain_over_baseline=8,
            maximum_paired_safety_regression=0,
            thresholds_not_policy_promotion=True,
            physical_teacher_qualified=False,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    results, total_steps = [], 0
    try:
        for index in range(a.shard, 40, 2):
            c = courses[index]
            choice = select_teacher(table, c.course)
            root = a.output / f"course-{index:02d}"
            root.mkdir()
            records, originals = [], {}
            for arm in ("baseline", "hybrid"):
                owned = dict(assets)
                if arm == "hybrid" and choice["experimental_fast"]:
                    owned.update(fast_assets)
                params = baseline if arm == "baseline" else choice["parameters"]
                for repeat in range(2):
                    row, arrays = evaluate_episode(
                        model=model,
                        assets=owned,
                        policy_types=types,
                        course=c.course,
                        parameters=params,
                        hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
                    )
                    total_steps += row["native_steps"]
                    recomputed = relabel_arrays(
                        arrays,
                        controls=row["controls"],
                        ball_address=row["ball_qpos_address"],
                        root_address=row["robot_qpos_address"],
                    )
                    if hash_json(recomputed) != hash_json(row["outcome"]):
                        raise ValueError("formal label disagrees with native state readback")
                    exact = None
                    if repeat:
                        with np.load(originals[arm]) as prior:
                            exact = {
                                k: bool(np.array_equal(v, prior[k])) for k, v in arrays.items()
                            }
                        if not all(exact.values()):
                            raise ValueError("new neural-instance reexecution differed")
                    artifact = root / f"{arm}-{repeat}.npz"
                    np.savez_compressed(artifact, **arrays)
                    if not repeat:
                        originals[arm] = artifact
                    row.update(
                        arm=arm,
                        repeat=repeat,
                        course_index=index,
                        choice=choice,
                        fresh_reexecution_exact=exact,
                        artifact=dict(path=artifact.name, hash=digest(artifact)),
                    )
                    records.append(row)
                    write_once(root / f"{arm}-{repeat}.json", row)
            write_once(
                root / "complete.json",
                dict(records=records, course_index=index, no_search_or_refit=True),
            )
            results.append(
                dict(
                    course_index=index,
                    source_hash=digest(root / "complete.json"),
                    baseline=records[0]["outcome"],
                    hybrid=records[2]["outcome"],
                    baseline_healthy=records[0]["healthy_full_horizon"],
                    hybrid_healthy=records[2]["healthy_full_horizon"],
                )
            )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PUBLIC40_FROZEN_HYBRID_TEACHER_SHARD_MEASURED_NOT_TRAINED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                table_hash=table["table_hash"],
                course_bank_hash=bank_hash,
                results=results,
                worlds=80,
                native_steps=total_steps,
                no_search_or_refit=True,
                physical_teacher_qualified=False,
                optimizer_updates=0,
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
                native_steps=total_steps,
                completed_courses=len(results),
                no_automatic_physics_retry=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
