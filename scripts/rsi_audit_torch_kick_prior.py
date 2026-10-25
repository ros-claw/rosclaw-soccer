"""Authenticate all80 native backend trials; never promote or train a policy."""

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
        raise ValueError("new audit and four unique complete producer roots required")
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
    preflight_receiving_courses(courses)
    pins, results, steps, layout = {}, [], 0, None
    for shard, root in enumerate(a.shard_root):
        cp = root / "complete.json"
        if cp.is_symlink():
            raise ValueError("ordinary complete physical shard required")
        source = json.loads(cp.read_text())
        if (
            source["status"] != "PUBLIC40_FROZEN_KICK_BACKEND_SHARD_EXECUTED_NOT_TRAINED"
            or source["source_head"] != a.expected_producer_head
            or source["worlds"] != 20
            or source["actor_optimizer_updates"] != 0
            or source["promotion_authorized"] is not False
            or [r["course_index"] for r in source["results"]] != list(range(shard, 40, 4))
        ):
            raise ValueError("entire frozen paired backend shard required")
        for key, value in {**source["input_file_hashes"], str(cp): digest(cp)}.items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting immutable evidence identity")
            pins[key] = value
        if layout is None:
            scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
            if len(scenes) != 1:
                raise ValueError("unique bound native body required")
            model = mujoco.MjModel.from_xml_string(
                shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
            )
            layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
        shard_steps = 0
        for item in source["results"]:
            index, rows = item["course_index"], item["rows"]
            folder = root / f"course-{index:02d}"
            case_path = folder / "complete.json"
            if case_path.is_symlink() or hash_json(json.loads(case_path.read_text())) != hash_json(
                item
            ):
                raise ValueError("original case completion differs")
            pins[str(case_path)] = digest(case_path)
            if set(rows) != {"onnx", "torch"}:
                raise ValueError("both full backend arms required")
            measured = {}
            for arm in ("onnx", "torch"):
                row = rows[arm]
                rp = folder / (arm + ".json")
                if rp.is_symlink() or hash_json(json.loads(rp.read_text())) != hash_json(row):
                    raise ValueError("original physical row differs")
                pins[str(rp)] = digest(rp)
                if (
                    row["course"] != list(courses[index].course)
                    or row["parameters"] != rows["onnx"]["parameters"]
                    or (
                        row["experimental_fast_approach"]
                        != rows["onnx"]["experimental_fast_approach"]
                        or row["ball_qpos_address"] != layout.bq
                        or row["robot_qpos_address"] != layout.rq
                    )
                ):
                    raise ValueError("same initial state, body and whole-body program required")
                artifact = folder / row["artifact"]["path"]
                if (
                    artifact.is_symlink()
                    or artifact.parent.resolve() != folder.resolve()
                    or (digest(artifact) != row["artifact"]["hash"])
                ):
                    raise ValueError("exact confined physical arrays required")
                pins[str(artifact)] = row["artifact"]["hash"]
                with np.load(artifact) as saved:
                    arrays = dict(saved)
                label = relabel_arrays(
                    arrays, controls=row["controls"], ball_address=layout.bq, root_address=layout.rq
                )
                pose = arrays["qpos"][:, layout.rq : layout.rq + 7]
                healthy = row["controls"] == 360 and not bool(
                    np.any(
                        (pose[:, 2] < 0.65) | (1 - 2 * (pose[:, 4] ** 2 + pose[:, 5] ** 2) < 0.9)
                    )
                )
                if (
                    healthy != row["healthy_full_horizon"]
                    or hash_json(label) != hash_json(row["outcome"])
                    or (
                        row["native_steps"] != row["controls"] * 10
                        or np.any(
                            np.abs(arrays["torque"][:, layout.ai])
                            > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
                        )
                    )
                ):
                    raise ValueError("recomputed labels, clock, health or effort disagree")
                if arm == "onnx":
                    suffix = f"/course-{index:02d}/"
                    matches = []
                    for name in source["input_file_hashes"]:
                        if (
                            suffix in name
                            and name.endswith(".json")
                            and ("/rsi-walk-kick-public40-authority-20261025-shard" in name)
                        ):
                            old_row = json.loads(Path(name).read_text())
                            if (
                                old_row.get("parameters") == row["parameters"]
                                and old_row.get("course") == row["course"]
                            ):
                                matches.append((Path(name), old_row))
                    if not matches:
                        raise ValueError("original physical source program missing")
                    exact = False
                    for original_path, old in matches:
                        old_path = original_path.parent / old["artifact"]["path"]
                        if digest(old_path) != old["artifact"]["hash"]:
                            raise ValueError("source raw array hash changed")
                        with np.load(old_path) as original:
                            exact |= set(original.files) == set(arrays) and all(
                                np.array_equal(original[k], v) for k, v in arrays.items()
                            )
                    if not exact:
                        raise ValueError("ONNX arm is not an exact source reexecution")
                shard_steps += row["native_steps"]
                measured[arm] = dict(outcome=label, healthy=healthy)
            results.append(dict(course_index=index, arms=measured))
        if shard_steps != source["native_steps"]:
            raise ValueError("charged native-step accounting differs")
        steps += shard_steps
    results.sort(key=lambda r: r["course_index"])
    if [r["course_index"] for r in results] != list(range(40)):
        raise ValueError("all40 paired contexts required")

    def passed(r, arm):
        return r["arms"][arm]["healthy"] and r["arms"][arm]["outcome"]["task_pass"]

    def safe(r, arm):
        return r["arms"][arm]["healthy"] and r["arms"][arm]["outcome"]["safe"]

    lost = [r["course_index"] for r in results if passed(r, "onnx") and not passed(r, "torch")]
    unsafe = [r["course_index"] for r in results if safe(r, "onnx") and not safe(r, "torch")]
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="ALL80_KICK_BACKEND_TRIALS_NATIVE_LABELS_AND_SOURCE_IDENTITY_VERIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            worlds=80,
            native_steps=steps,
            results=results,
            passes={arm: sum(passed(r, arm) for r in results) for arm in ("onnx", "torch")},
            class_passes={
                arm: [sum(passed(r, arm) for r in results[k * 10 : (k + 1) * 10]) for k in range(4)]
                for arm in ("onnx", "torch")
            },
            lost_success_indices=lost,
            safety_regression_indices=unsafe,
            rescue_indices=[
                r["course_index"] for r in results if passed(r, "torch") and not passed(r, "onnx")
            ],
            eligible_for_motor_training_design=not lost and not unsafe,
            actor_optimizer_updates=0,
            physical_football_gain_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
