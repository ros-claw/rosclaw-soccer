"""Build causal549 entry features only after exact frozen first-action recovery."""

import argparse
import contextlib
import io
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", nargs=2, action="append", metavar=("PATH", "HASH"), required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or len(a.audit) != 3 or len({p for p, _ in a.audit}) != 3:
        raise ValueError("new output and three distinct closed audit sources required")
    pins, declarations, episodes = {}, [], []
    allowed = {
        "PUBLIC16_CEM_ALL304_RAW_LABELS_AND_REEXECUTION_AUTHENTICATED_NOT_PROMOTED": 16,
        "PUBLIC40_CEM_ALL760_RAW_LABELS_AND_REEXECUTION_AUTHENTICATED_NOT_PROMOTED": 40,
    }
    for name, expected in a.audit:
        path = Path(name)
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact completed raw-data audit required")
        report = json.loads(path.read_text())
        count = allowed.get(report["status"])
        if (
            count is None
            or [r["course_index"] for r in report["results"]] != list(range(count))
            or (report["promotion_authorized"] is not False)
        ):
            raise ValueError("entire admitted public teacher search required")
        for key, value in {**report["input_file_hashes"], str(path): expected}.items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting source file identity")
            pins[key] = value
        declarations.append(
            dict(
                path=str(path),
                hash=expected,
                contexts=count,
                experimental=report.get("experimental_fast_approach", False),
            )
        )
        for item in report["results"]:
            index, course = item["course_index"], item["course"]
            suffix = f"/course-{index:02d}/complete.json"
            cases = []
            for key in report["input_file_hashes"]:
                if key.endswith(suffix):
                    case = json.loads(Path(key).read_text())
                    if case.get("course") == course:
                        cases.append((Path(key), case))
            if len(cases) != 1:
                raise ValueError("unique audited physical course join required")
            case_path, case = cases[0]
            names = ["baseline", *[f"candidate-{i:02d}" for i in range(16)]]
            if len(case["raw_record_hashes"]) != 19:
                raise ValueError("all physical trials including reexecutions must remain present")
            for record_name in names:
                rp = case_path.parent / (record_name + ".json")
                if rp.is_symlink() or digest(rp) != case["raw_record_hashes"][record_name]:
                    raise ValueError("exact original physical proposal required")
                row = json.loads(rp.read_text())
                if row["course"] != course or row["name"] != record_name:
                    raise ValueError("physical record identity differs from declaration")
                artifact = rp.parent / row["artifact"]["path"]
                if (
                    artifact.parent.resolve() != rp.parent.resolve()
                    or artifact.is_symlink()
                    or (digest(artifact) != row["artifact"]["hash"])
                ):
                    raise ValueError("exact confined original native arrays required")
                episodes.append((rp, artifact, row))
    if sorted(r["contexts"] for r in declarations) != [16, 16, 40] or len(episodes) != 1224:
        raise ValueError("exact original72 course-executions/1224 non-replay proposals required")
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique unchanged actual asset required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo

    from rosclaw_soccer.rsi.first_touch_mechanism_expansion import validate_public_course
    from rosclaw_soccer.rsi.kick_entry_reconstruction import KickEntryReconstructor
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.sim.contracts import hash_json

    root = one("/policy/robonaldo/FreeKick.py").parents[2]
    state_type, output_type, policy_type, mapping = _load_robonaldo(root)
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(),
            prefixes=("a/",),
        )
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    with contextlib.redirect_stdout(io.StringIO()):
        policy = policy_type(state_type(29), output_type(29))
    reconstructor = KickEntryReconstructor(model, layout, policy, mapping)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.g1.causal_kick_entry_outcome_bank.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            sources=declarations,
            expected_original_records=1224,
            expected_distinct_initial_states=56,
            reexecutions_are_not_training_rows=True,
            encoder="actual547firstkickobs,refphase/(T-1),remainingH300time",
            feature_dimension=549,
            outcome_order=["safe_and_ready", "foot_observed_H300", "quality_and_ready"],
            input_contains_outcome_or_course_id=False,
            exact_first_target_required=True,
            missing_entry_is_not_a_zero_filled_sample=True,
            physical_steps_added=0,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    records, features, labels, groups, classes = [], [], [], [], []
    unique, duplicate_count, no_entry = {}, 0, 0
    try:
        for i, (rp, artifact, row) in enumerate(episodes):
            group = hash_json(dict(initial_course=row["course"]))
            bucket = validate_public_course(tuple(row["course"]))
            record = dict(
                index=i,
                record=str(rp),
                record_hash=digest(rp),
                artifact_hash=row["artifact"]["hash"],
                group=group,
                course=row["course"],
                stratum=bucket,
                outcome=row["outcome"],
                healthy_full_horizon=row["healthy_full_horizon"],
            )
            if row["entry_control"] is None:
                no_entry += 1
                record.update(status="NO_ADMISSION_PRESERVED_NOT_FEATURED", feature_index=None)
            else:
                with np.load(artifact) as saved:
                    x = reconstructor.reconstruct(row, dict(saved))
                outcome = row["outcome"]
                y = np.array(
                    [
                        outcome["safe"] and row["healthy_full_horizon"],
                        outcome["first_foot_contact_frame"] is not None,
                        outcome["task_pass"] and row["healthy_full_horizon"],
                    ],
                    np.float32,
                )
                key = hash_json(dict(feature=x.tolist()))
                if key in unique:
                    old_i = unique[key]
                    if groups[old_i] != group or not np.array_equal(labels[old_i], y):
                        raise ValueError(
                            "identical input aliases different group/labels; no silent dedup"
                        )
                    duplicate_count += 1
                    record.update(status="EXACT_FEATURE_DUPLICATE_PRESERVED", feature_index=old_i)
                else:
                    unique[key] = len(features)
                    record.update(
                        status="FIRST_TARGET_EXACT_FEATURE_ADMITTED", feature_index=len(features)
                    )
                    features.append(x)
                    labels.append(y)
                    groups.append(group)
                    classes.append(bucket)
                record["first_target_exact"] = True
            records.append(record)
            write_once(a.output / f"record-{i:04d}.json", record)
        if len({r["group"] for r in records}) != 56:
            raise ValueError("original56 initial-context groups required")
        artifact = a.output / "entry-outcomes.npz"
        np.savez_compressed(
            artifact,
            features=np.stack(features),
            labels=np.stack(labels),
            groups=np.asarray(groups),
            strata=np.asarray(classes, np.int64),
        )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="CAUSAL_KICK_ENTRY_BANK_RECONSTRUCTED_FIRST_ACTIONS_EXACT_NOT_TRAINED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                sources=declarations,
                records=records,
                total_original_records=len(records),
                no_admission_records=no_entry,
                admitted_reconstructions=len(records) - no_entry,
                exact_feature_duplicates=duplicate_count,
                unique_training_rows=len(features),
                independent_initial_states=56,
                body_hash=hash_json(layout.profile()),
                policy_model_hash=digest(one("/policy/robonaldo/model/policy-obs-aic.onnx")),
                artifact=dict(path=artifact.name, hash=digest(artifact)),
                class_row_counts=np.bincount(classes, minlength=4).tolist(),
                label_rates=np.stack(labels).mean(axis=0).tolist(),
                physical_steps_added=0,
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
                completed_records=len(records),
                next_record=str(episodes[len(records)][0])
                if len(records) < len(episodes)
                else None,
                physical_steps_added=0,
                no_training_authorized=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
