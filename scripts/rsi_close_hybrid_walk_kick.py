"""Read-only successor for both public40 producers; never launch training/physics."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-root", type=Path, nargs=2, required=True)
    p.add_argument("--producer-unit", nargs=2, required=True)
    p.add_argument("--expected-producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_close_walk_kick_teacher import wait_for_producers
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or len({p.resolve() for p in a.shard_root}) != 2:
        raise ValueError("new output and two unique declared producers required")
    wait_for_producers(a.producer_unit)
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    pins, rows, table_hash, bank_hash = {}, [], None, None
    total_steps = 0
    layout = None
    for shard_index, root in enumerate(a.shard_root):
        path = root / "complete.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing completed producer; no automatic physics retry")
        source = json.loads(path.read_text())
        if (
            source["source_head"] != a.expected_producer_head
            or source["status"] != "PUBLIC40_FROZEN_HYBRID_TEACHER_SHARD_MEASURED_NOT_TRAINED"
            or source["worlds"] != 80
            or source["no_search_or_refit"] is not True
            or source["promotion_authorized"] is not False
            or [r["course_index"] for r in source["results"]] != list(range(shard_index, 40, 2))
            or (table_hash is not None and table_hash != source["table_hash"])
            or (bank_hash is not None and bank_hash != source["course_bank_hash"])
        ):
            raise ValueError("entire unchanged two-shard validation protocol required")
        table_hash, bank_hash = source["table_hash"], source["course_bank_hash"]
        for key, value in {**source["input_file_hashes"], str(path): digest(path)}.items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting validation ancestry")
            pins[key] = value
        if layout is None:
            scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
            if len(scenes) != 1:
                raise ValueError("unique physical scene required")
            model = mujoco.MjModel.from_xml_string(
                shared_native_scene(
                    scenes[0].read_text(),
                    prefixes=("a/",),
                )
            )
            layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
        for summary in source["results"]:
            index = summary["course_index"]
            course_root = root / f"course-{index:02d}"
            cp = course_root / "complete.json"
            if cp.is_symlink() or digest(cp) != summary["source_hash"]:
                raise ValueError("exact course completion required")
            pins[str(cp)] = summary["source_hash"]
            records = json.loads(cp.read_text())["records"]
            if [(r["arm"], r["repeat"]) for r in records] != [
                (arm, repeat) for arm in ("baseline", "hybrid") for repeat in range(2)
            ]:
                raise ValueError("all four original physical trials required")
            paths = {}
            for row in records:
                artifact = course_root / row["artifact"]["path"]
                if (
                    artifact.is_symlink()
                    or artifact.parent.resolve() != course_root.resolve()
                    or (digest(artifact) != row["artifact"]["hash"])
                ):
                    raise ValueError("exact confined native trace required")
                if row["arm"] == "baseline" and row["experimental_fast_approach"] is not False:
                    raise ValueError("baseline must use ORIGINAL strict velocity domain")
                pins[str(artifact)] = row["artifact"]["hash"]
                if row["ball_qpos_address"] != layout.bq or row["robot_qpos_address"] != layout.rq:
                    raise ValueError("record state addresses differ from actual model")
                with np.load(artifact) as saved:
                    label = relabel_arrays(
                        dict(saved),
                        controls=row["controls"],
                        ball_address=row["ball_qpos_address"],
                        root_address=row["robot_qpos_address"],
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
                        or (row["native_steps"] != row["controls"] * 10)
                        or np.any(
                            np.abs(saved["torque"][:, layout.ai])
                            > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
                        )
                    ):
                        raise ValueError("actual body health, clock or torque bounds disagree")
                    active = saved["velocity_control_active"]
                    if (
                        active.dtype != np.bool_
                        or active.shape != (row["controls"],)
                        or (int(active.sum()) != row["velocity_neural_calls"])
                    ):
                        raise ValueError("actual velocity control mask differs from actor calls")
                if hash_json(label) != hash_json(row["outcome"]):
                    raise ValueError("saved task result differs from raw native arrays")
                paths[(row["arm"], row["repeat"])] = artifact
                total_steps += row["native_steps"]
            for arm in ("baseline", "hybrid"):
                with np.load(paths[(arm, 0)]) as old, np.load(paths[(arm, 1)]) as new:
                    if set(old.files) != set(new.files) or any(
                        not np.array_equal(old[k], new[k]) for k in old.files
                    ):
                        raise ValueError("fresh NN physical reexecution differed")
            rows.append(
                dict(
                    course_index=index,
                    course=records[0]["course"],
                    baseline=records[0]["outcome"],
                    hybrid=records[2]["outcome"],
                    baseline_healthy=records[0]["healthy_full_horizon"],
                    hybrid_healthy=records[2]["healthy_full_horizon"],
                    hybrid_nonfoot_native_samples=records[2]["native_nonfoot_contact_samples"],
                    actual_context_choice=records[2]["choice"],
                )
            )
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    rows.sort(key=lambda r: r["course_index"])
    if [r["course_index"] for r in rows] != list(range(40)):
        raise ValueError("entire balanced public40 required")

    def passed(row, arm):
        return row[arm + "_healthy"] and row[arm]["task_pass"]

    baseline = sum(passed(r, "baseline") for r in rows)
    hybrid = sum(passed(r, "hybrid") for r in rows)
    regression = sum(passed(r, "baseline") and not passed(r, "hybrid") for r in rows)
    safety = sum(r["baseline"]["safe"] and not r["hybrid"]["safe"] for r in rows)
    ready = hybrid >= 32 and hybrid - baseline >= 8 and not safety and not regression
    write_once(
        a.output,
        dict(
            status="PUBLIC40_FROZEN_HYBRID_ALL160_REEXECUTIONS_AND_RAW_LABELS_VERIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            table_hash=table_hash,
            course_bank_hash=bank_hash,
            results=rows,
            worlds=160,
            native_steps=total_steps,
            baseline_passes=baseline,
            hybrid_passes=hybrid,
            task_regression=regression,
            rescue=sum(passed(r, "hybrid") and not passed(r, "baseline") for r in rows),
            paired_safety_regression=safety,
            preregistered_thresholds_plus_retention_met=ready,
            next_required_action="MULTISEED_STUDENT_DESIGN_AND_RETENTION_REVIEW"
            if ready
            else "PRESERVE_FAILURES_REPAIR_TEACHER_NO_BLIND_STUDENT_TRAINING",
            automatic_training_launched=False,
            sealed_fresh_verified=False,
            physical_teacher_qualified=False,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
