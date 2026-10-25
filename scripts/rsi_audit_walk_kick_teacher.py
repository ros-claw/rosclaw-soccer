"""Authenticate all four CEM shards and all304 raw trials; no policy promotion."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", nargs=2, action="append", metavar=("ROOT", "HASH"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or len(a.shard) != 4:
        raise ValueError("new output and all four immutable shards required")
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.rsi.walk_kick_teacher_plan import (
        candidate_parameters,
        elite_candidates,
        outcome_rank,
        teacher_courses,
        uniform_candidates,
    )
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = teacher_courses()
    bank_hash = preflight_receiving_courses(courses)
    pins, summaries, seen = {}, [], set()
    model_layout = None
    total_steps = 0
    experimental_fast = None
    for shard_index, (name, expected_hash) in enumerate(a.shard):
        root = Path(name)
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != expected_hash:
            raise ValueError("exact completed shard required")
        shard = json.loads(path.read_text())
        fast = shard.get("experimental_fast_approach", False)
        if type(fast) is not bool or (experimental_fast is not None and fast != experimental_fast):
            raise ValueError("one explicit velocity-domain protocol across all shards required")
        experimental_fast = fast
        if (
            shard["status"] != "PUBLIC16_TEACHER_CEM_SHARD_MEASURED_NOT_TRAINED"
            or shard["course_bank_hash"] != bank_hash
            or [r["course_index"] for r in shard["results"]] != list(range(shard_index, 16, 4))
            or shard["worlds"] != 76
            or shard["promotion_authorized"] is not False
        ):
            raise ValueError("whole declared shard with no outcome omission required")
        additions = dict(shard["input_file_hashes"])
        additions[str(path)] = expected_hash
        for key, value in additions.items():
            if key in pins and pins[key] != value:
                raise ValueError("inherited asset identity conflict")
            pins[key] = value
        check_pins(additions)
        if model_layout is None:
            scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
            if len(scenes) != 1:
                raise ValueError("one qualified model required")
            model = mujoco.MjModel.from_xml_string(
                shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
            )
            model_layout = NativeVelocityLayout(
                model, robot_prefix="a/", ball_body_name="soccer_ball"
            )
        lo = model_layout
        for item in shard["results"]:
            index = item["course_index"]
            c = courses[index]
            course_root = root / f"course-{index:02d}"
            cp = course_root / "complete.json"
            if index in seen or cp.is_symlink() or digest(cp) != item["source_hash"]:
                raise ValueError("unique exact course completion required")
            seen.add(index)
            summary = json.loads(cp.read_text())
            pins[str(cp)] = item["source_hash"]
            if hash_json(summary) != hash_json(item["summary"]) or summary["worlds"] != 19:
                raise ValueError("all nineteen declared physical trials required")
            rows, files = {}, {}
            names = [
                "baseline",
                *[f"candidate-{i:02d}" for i in range(16)],
                "baseline-reexecution",
                "selected-reexecution",
            ]
            if set(summary["raw_record_hashes"]) != set(names):
                raise ValueError("all candidates and both reexecutions required")
            for record_name in names:
                rp = course_root / (record_name + ".json")
                if rp.is_symlink() or digest(rp) != summary["raw_record_hashes"][record_name]:
                    raise ValueError("exact original physical record required")
                row = json.loads(rp.read_text())
                if row.get("experimental_fast_approach", False) != fast:
                    raise ValueError("record domain differs from campaign")
                artifact = course_root / row["artifact"]["path"]
                if (
                    artifact.parent.resolve() != course_root.resolve()
                    or artifact.is_symlink()
                    or digest(artifact) != row["artifact"]["hash"]
                    or row["ball_qpos_address"] != lo.bq
                    or row["robot_qpos_address"] != lo.rq
                    or row["course"] != list(c.course)
                    or row["course_index"] != index
                    or row["native_steps"] != row["controls"] * 10
                ):
                    raise ValueError("model-bound original raw trial required")
                with np.load(artifact) as saved:
                    recomputed = relabel_arrays(
                        dict(saved),
                        controls=row["controls"],
                        ball_address=lo.bq,
                        root_address=lo.rq,
                    )
                    pose = saved["qpos"][:, lo.rq : lo.rq + 7]
                    healthy = row["controls"] == 360 and not bool(
                        np.any(
                            (pose[:, 2] < 0.65)
                            | (1 - 2 * (pose[:, 4] ** 2 + pose[:, 5] ** 2) < 0.9)
                        )
                    )
                    if healthy != row["healthy_full_horizon"] or np.any(
                        np.abs(saved["torque"][:, lo.ai])
                        > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
                    ):
                        raise ValueError("actual body health or hard torque bound mismatch")
                    if (
                        int((saved["ball_forces"][:, 0, 2] > 1).sum())
                        != row["native_nonfoot_contact_samples"]
                    ):
                        raise ValueError("actual nonfoot contact count mismatch")
                if hash_json(recomputed) != hash_json(row["outcome"]):
                    raise ValueError("stored score differs from actual native trace")
                rows[record_name], files[record_name] = row, artifact
                pins[str(rp)] = summary["raw_record_hashes"][record_name]
                pins[str(artifact)] = row["artifact"]["hash"]
                total_steps += row["native_steps"]
            first = uniform_candidates(c.seed, experimental_fast=fast)
            second = elite_candidates(
                first,
                tuple(outcome_rank(rows[f"candidate-{i:02d}"]) for i in range(8)),
                seed=c.seed + 1000,
                experimental_fast=fast,
            )
            for i, vector in enumerate(np.concatenate((first, second))):
                if rows[f"candidate-{i:02d}"]["parameters"] != candidate_parameters(
                    vector, experimental_fast=fast
                ):
                    raise ValueError("search differs from frozen two-generation rule")
            selected = max(names[:17], key=lambda n: outcome_rank(rows[n]))
            if selected != summary["selected"]:
                raise ValueError("best candidate/fallback differs from declared rank")
            for label, original in (
                ("baseline-reexecution", "baseline"),
                ("selected-reexecution", selected),
            ):
                if rows[label]["parameters"] != rows[original]["parameters"]:
                    raise ValueError("reexecution changed control parameters")
                with np.load(files[original]) as old, np.load(files[label]) as new:
                    if set(old.files) != set(new.files) or any(
                        not np.array_equal(old[k], new[k]) for k in old.files
                    ):
                        raise ValueError("fresh neural physical reexecution differs")
            base, chosen = rows["baseline"], rows[selected]
            summaries.append(
                dict(
                    course_index=index,
                    course=c.course,
                    selected=selected,
                    baseline_outcome=base["outcome"],
                    selected_outcome=chosen["outcome"],
                    baseline_healthy=base["healthy_full_horizon"],
                    selected_healthy=chosen["healthy_full_horizon"],
                    candidates_with_formal_pass=sum(
                        rows[n]["outcome"]["task_pass"] for n in names[1:17]
                    ),
                    candidates_with_nonfoot=sum(
                        rows[n]["native_nonfoot_contact_samples"] > 0 for n in names[1:17]
                    ),
                    per_state_privileged_oracle=True,
                )
            )
    if seen != set(range(16)):
        raise ValueError("complete balanced sixteen-state bank required")
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    summaries.sort(key=lambda row: row["course_index"])

    def passes(r, key):
        return r[key + "_healthy"] and r[key + "_outcome"]["task_pass"]

    write_once(
        a.output,
        dict(
            status="PUBLIC16_CEM_ALL304_RAW_LABELS_AND_REEXECUTION_AUTHENTICATED_NOT_PROMOTED",
            experimental_fast_approach=experimental_fast,
            source_head=a.expected_head,
            input_file_hashes=pins,
            course_bank_hash=bank_hash,
            results=summaries,
            worlds=304,
            native_steps=total_steps,
            baseline_passes=sum(passes(r, "baseline") for r in summaries),
            per_state_oracle_passes=sum(passes(r, "selected") for r in summaries),
            rescue=sum(passes(r, "selected") and not passes(r, "baseline") for r in summaries),
            regression=sum(passes(r, "baseline") and not passes(r, "selected") for r in summaries),
            per_state_oracle_is_not_a_generalizing_controller=True,
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
