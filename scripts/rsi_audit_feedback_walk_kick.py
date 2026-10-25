"""Close four feedback ablation shards from native data, not a best-case video."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-root", type=Path, nargs=4, required=True)
    p.add_argument("--producer-unit", nargs=4, required=True)
    p.add_argument("--expected-producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_close_walk_kick_teacher import wait_for_producers
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or len({p.resolve() for p in a.shard_root}) != 4:
        raise ValueError("unused output and four unique producer roots required")
    wait_for_producers(a.producer_unit)
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.rsi.walk_kick_teacher_plan import search_course_bank
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = search_course_bank(expanded_public=True)
    bank_hash = preflight_receiving_courses(courses)
    arms = ("constant", "velocity-feedback", "velocity-and-phase-feedback")
    pins, results, total_steps, layout = {}, [], 0, None
    for shard_index, root in enumerate(a.shard_root):
        cp = root / "complete.json"
        if cp.is_symlink():
            raise ValueError("ordinary completed shard required")
        source = json.loads(cp.read_text())
        if (
            source["status"] != "PUBLIC40_FEEDBACK_ABLATION_SHARD_MEASURED_NOT_LEARNED"
            or source["source_head"] != a.expected_producer_head
            or source["worlds"] != 60
            or source["course_bank_hash"] != bank_hash
            or source["optimizer_updates"] != 0
            or source["promotion_authorized"] is not False
            or [r["course_index"] for r in source["results"]] != list(range(shard_index, 40, 4))
        ):
            raise ValueError("entire fixed feedback ablation protocol required")
        for key, value in {**source["input_file_hashes"], str(cp): digest(cp)}.items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting inherited evidence identity")
            pins[key] = value
        if layout is None:
            scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
            if len(scenes) != 1:
                raise ValueError("unique actual model required")
            model = mujoco.MjModel.from_xml_string(
                shared_native_scene(
                    scenes[0].read_text(),
                    prefixes=("a/",),
                )
            )
            layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
        for item in source["results"]:
            index = item["course_index"]
            course_root = root / f"course-{index:02d}"
            path = course_root / "complete.json"
            if path.is_symlink() or digest(path) != item["source_hash"]:
                raise ValueError("exact original case completion required")
            pins[str(path)] = item["source_hash"]
            records = json.loads(path.read_text())["records"]
            if [(r["arm"], r["repeat"]) for r in records] != [
                (arm, repeat) for arm in arms for repeat in range(2)
            ]:
                raise ValueError("all six paired physical trials required")
            original = records[0]
            paths, measured = {}, {}
            for row in records:
                arm = row["arm"]
                params = dict(original["parameters"])
                if arm != "constant":
                    params["approach_feedback"] = True
                if arm == "velocity-and-phase-feedback":
                    params["entry_phase_feedback"] = True
                if row["parameters"] != params or (
                    row["experimental_fast_approach"] != original["experimental_fast_approach"]
                    or row["course"] != list(courses[index].course)
                    or row["ball_qpos_address"] != layout.bq
                    or row["robot_qpos_address"] != layout.rq
                ):
                    raise ValueError("matched parameters, actual body and source course required")
                artifact = course_root / row["artifact"]["path"]
                if (
                    artifact.is_symlink()
                    or artifact.parent.resolve() != course_root.resolve()
                    or (digest(artifact) != row["artifact"]["hash"])
                ):
                    raise ValueError("exact confined native trace required")
                pins[str(artifact)] = row["artifact"]["hash"]
                with np.load(artifact) as saved:
                    label = relabel_arrays(
                        dict(saved),
                        controls=row["controls"],
                        ball_address=layout.bq,
                        root_address=layout.rq,
                    )
                    pose = saved["qpos"][:, layout.rq : layout.rq + 7]
                    healthy = row["controls"] == 360 and not bool(
                        np.any(
                            (pose[:, 2] < 0.65)
                            | (1 - 2 * (pose[:, 4] ** 2 + pose[:, 5] ** 2) < 0.9)
                        )
                    )
                    if (
                        healthy != row["healthy_full_horizon"]
                        or np.any(
                            np.abs(saved["torque"][:, layout.ai])
                            > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
                        )
                        or hash_json(label) != hash_json(row["outcome"])
                    ):
                        raise ValueError("native label, health or torque evidence disagrees")
                paths[(arm, row["repeat"])] = artifact
                total_steps += row["native_steps"]
                if not row["repeat"]:
                    measured[arm] = dict(
                        outcome=label,
                        healthy=healthy,
                        entry_control=row["entry_control"],
                        phase=row["admitted_reference_phase"],
                        nonfoot_native_samples=row["native_nonfoot_contact_samples"],
                    )
            for arm in arms:
                with np.load(paths[(arm, 0)]) as old, np.load(paths[(arm, 1)]) as new:
                    if set(old.files) != set(new.files) or any(
                        not np.array_equal(old[k], new[k]) for k in old.files
                    ):
                        raise ValueError("fresh neural-instance reexecution differs")
            results.append(dict(course_index=index, course=courses[index].course, arms=measured))
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    results.sort(key=lambda r: r["course_index"])
    if [r["course_index"] for r in results] != list(range(40)):
        raise ValueError("all forty public states required")

    def passed(row, arm):
        x = row["arms"][arm]
        return x["healthy"] and x["outcome"]["task_pass"]

    summaries = {}
    for arm in arms:
        classes = [sum(passed(r, arm) for r in results[k * 10 : (k + 1) * 10]) for k in range(4)]
        regressions = sum(passed(r, "constant") and not passed(r, arm) for r in results)
        safety = sum(
            r["arms"]["constant"]["outcome"]["safe"] and not r["arms"][arm]["outcome"]["safe"]
            for r in results
        )
        summaries[arm] = dict(
            passes=sum(classes),
            class_passes=classes,
            rescue=sum(passed(r, arm) and not passed(r, "constant") for r in results),
            regression=regressions,
            safety_regression=safety,
            eligible_for_new_distribution_review=(
                sum(classes) >= 32 and min(classes) >= 6 and not regressions and not safety
            ),
        )
    write_once(
        a.output,
        dict(
            status="PUBLIC40_FEEDBACK_ALL240_NATIVE_LABELS_AND_REEXECUTIONS_VERIFIED_NOT_PROMOTED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            course_bank_hash=bank_hash,
            results=results,
            summaries=summaries,
            worlds=240,
            native_steps=total_steps,
            selected_best_arm_per_state=False,
            original52_verified=False,
            sealed_fresh_verified=False,
            optimizer_updates=0,
            physical_teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
