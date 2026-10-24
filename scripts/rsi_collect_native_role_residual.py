"""Bounded parallel real-native residual exploration after exact parent identity.

4 declared trials per course, all5 courses, source and fresh NN execution each.
The12sec baseline is re-executed by the pinned worker; duplicates stay evidence,
never extra training rows. No retries, private Fresh, ball reset, or hardware.
"""

import argparse
import concurrent.futures
import fcntl
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(926061, 926062, 926063), required=True)
    a = p.parse_args()
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.plan.is_symlink() or digest(a.plan) != a.plan_hash:
        raise ValueError("unused collection output and exact exploration plan required")
    plan = json.loads(a.plan.read_text())
    if (
        plan["schema"] != "soccer.native_role_residual_exploration.v1"
        or plan["source_head"] != a.expected_head
        or plan["trials_per_course"] != 4
        or plan["workers_per_seed"] != 4
        or plan["hardware_authorized"] is not False
    ):
        raise ValueError("bounded predeclared4-trial/4-worker SIM collection required")
    base_path = Path(plan["identity_plan"])
    if base_path.is_symlink() or digest(base_path) != plan["identity_plan_hash"]:
        raise ValueError("exact qualified identity plan required")
    base = json.loads(base_path.read_text())
    identity = plan["identities"][str(a.seed)]
    identity_root = Path(identity["root"])
    path = identity_root / "complete.json"
    if path.is_symlink() or digest(path) != identity["hash"]:
        raise ValueError("completed exact own parent identity required before exploration")
    identity_report = json.loads(path.read_text())
    if (
        identity_report["seed"] != a.seed
        or len(identity_report["results"]) != 5
        or any(
            not r["healthy_full_horizon"]
            or (
                r["arm"] == "role-command-residual"
                and not all(r["zero_residual_parent_identity"].values())
            )
            or (r["repeat"] and not all(r["fresh_reexecution_exact"].values()))
            for case in identity_report["results"]
            for r in case["records"]
        )
    ):
        raise ValueError("all5 zero residual identities and fresh executions required")
    init = base["residual_initializations"][str(a.seed)]
    init_root = Path(init["root"])
    path = init_root / "complete.json"
    if digest(path) != init["hash"] or path.is_symlink():
        raise ValueError("exact initialized residual required")
    initialized = json.loads(path.read_text())
    if (
        initialized["optimizer_updates"] != 0
        or initialized["historical_encoder_optimizer_updates"] != 1024
    ):
        raise ValueError("untrained head and retained encoder required for this collection")
    pins = dict(initialized["input_file_hashes"])
    pins[str(path)] = init["hash"]
    pins[str(identity_root / "complete.json")] = identity["hash"]
    pins[str(base_path)] = plan["identity_plan_hash"]
    pins[str(a.plan)] = a.plan_hash
    for case in identity_report["results"]:
        root = identity_root / case["course"]
        path = root / "complete.json"
        if (
            root.parent.resolve() != identity_root.resolve()
            or path.is_symlink()
            or digest(path) != case["source_hash"]
        ):
            raise ValueError("exact confined identity stage required")
        verified = json.loads(path.read_text())
        for key, value in verified["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting qualified identity ancestry")
            pins[key] = value
        pins[str(path)] = case["source_hash"]
        for record in verified["records"]:
            path = root / record["artifact"]["path"]
            if (
                path.parent.resolve() != root.resolve()
                or path.is_symlink()
                or digest(path) != record["artifact"]["hash"]
            ):
                raise ValueError("qualified identity raw arrays changed")
            pins[str(path)] = record["artifact"]["hash"]
    for item in (initialized["policy"], initialized["encoder"]):
        path = init_root / item["path"]
        if (
            path.parent.resolve() != init_root.resolve()
            or path.is_symlink()
            or digest(path) != item["hash"]
        ):
            raise ValueError("exact confined residual/encoder transfer required")
        pins[str(path)] = item["hash"]
    pin_active_core(Path(plan["core_root"]), plan["core_head"], pins)
    source_pins(pins)
    check_pins(pins)
    from rosclaw_soccer.rsi.native_role_residual_task import (
        residual_reward_contract,
        residual_reward_hash,
    )
    from rosclaw_soccer.training.native_team_courses import native_team_course_bank
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = native_team_course_bank()
    bank_hash = preflight_receiving_courses(courses)
    if bank_hash != base["course_bank_hash"] or [c.course_id for c in courses] != base["courses"]:
        raise ValueError("entire unchanged preflighted course bank required")
    motor_root = Path(plan["motor_source_root"])
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=motor_root, text=True).strip()
        != base["source_head"]
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=motor_root, text=True
        ).strip()
    ):
        raise ValueError("exact clean pinned native source required")
    a.output.mkdir(parents=True)
    with (a.output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(
            a.output / "declaration.json",
            dict(
                schema=plan["schema"],
                source_head=a.expected_head,
                seed=a.seed,
                plan_hash=a.plan_hash,
                input_file_hashes=pins,
                identity_hash=identity["hash"],
                initialization_hash=init["hash"],
                course_bank_hash=bank_hash,
                optimizer_updates=0,
                workers=4,
                maximum_world_executions=80,
                maximum_native_steps=480000,
                reward_contract=residual_reward_contract(),
                reward_contract_hash=residual_reward_hash(),
                action_semantics="NORMALIZED_RESIDUAL_BEFORE_ENVIRONMENT_GUARD",
                actor_context_dim=121,
                actor_feature_dim=131,
                critic_context_dim=152,
                all8_roles_explore=True,
                duplicates_are_not_training_examples=True,
                duration_is_sampler_truncation=True,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )

        def one_course(course_index, course):
            import numpy as np

            from rosclaw_soccer.rsi.native_role_residual_data import build_role_residual_episode

            case_root = a.output / course.course_id
            case_root.mkdir()
            bank, results, source_pinned = {}, [], {}
            sample_ids = []
            for trial in range(4):
                output = case_root / f"trial-{trial}"
                exploration_seed = a.seed * 100 + course_index * 10 + trial
                training = base["training_root_template"].format(seed=a.seed)
                command = [
                    sys.executable,
                    "-m",
                    "scripts.rsi_probe_native_team_intent",
                    "--intercept-course",
                    "--foot-track-course",
                    "--clearance-course",
                    "--ball-course",
                    course.course_id,
                    "--command-training",
                    training,
                    "--command-training-hash",
                    identity_report["training_hash"],
                    "--command-residual",
                    str(init_root / initialized["policy"]["path"]),
                    "--command-residual-hash",
                    initialized["policy"]["hash"],
                    "--residual-exploration-seed",
                    str(exploration_seed),
                    "--expected-head",
                    base["source_head"],
                    "--output",
                    str(output),
                ]
                for option, key in (
                    ("core-source-root", "core_root"),
                    ("core-expected-head", "core_head"),
                    ("parent", "parent_root"),
                    ("parent-hash", "parent_hash"),
                    ("own", "own_root"),
                    ("own-hash", "own_hash"),
                    ("previous", "previous_root"),
                    ("previous-hash", "previous_hash"),
                ):
                    command += [
                        "--" + option,
                        plan[key] if key in ("core_root", "core_head") else base[key],
                    ]
                foot = base["foot_courses"][course.course_id]
                command += ["--foot-reference", foot["root"], "--foot-reference-hash", foot["hash"]]
                subprocess.run(command, cwd=motor_root, check=True)
                path = output / "complete.json"
                if not path.is_file() or (output / "failed.json").exists():
                    raise ValueError("incomplete/failed physical source retained; no retry")
                report = json.loads(path.read_text())
                declaration = json.loads((output / "declaration.json").read_text())
                if (
                    len(report["records"]) != 4
                    or not report["fresh_reexecution_exact"]
                    or declaration["residual_exploration_seed"] != exploration_seed
                    or declaration["separate_residual_hash"] != initialized["policy"]["hash"]
                ):
                    raise ValueError("physical source differs from exploration commitment")
                for k, v in report["input_file_hashes"].items():
                    if k in source_pinned and source_pinned[k] != v:
                        raise ValueError("conflicting physical source dependencies")
                    source_pinned[k] = v
                source_pinned[str(path)] = digest(path)
                for r in report["records"]:
                    artifact = output / r["artifact"]["path"]
                    if (
                        artifact.is_symlink()
                        or artifact.parent.resolve() != output.resolve()
                        or digest(artifact) != r["artifact"]["hash"]
                    ):
                        raise ValueError("physical raw arrays changed")
                    source_pinned[str(artifact)] = r["artifact"]["hash"]
                row = next(
                    r
                    for r in report["records"]
                    if r["arm"] == "role-command-residual" and r["repeat"] == 0
                )
                data, metrics = build_role_residual_episode(
                    output, row, report["input_file_hashes"]
                )
                for key, value in data.items():
                    bank.setdefault(key, []).append(value.reshape((-1, *value.shape[2:])))
                length = data["rewards"].shape[0]
                sample_ids.extend(
                    (course_index, trial, frame + 50, player)
                    for frame in range(length)
                    for player in range(8)
                )
                result = dict(
                    trial=trial,
                    exploration_seed=exploration_seed,
                    source_hash=digest(path),
                    native_steps=report["native_steps"],
                    metrics=metrics,
                    records=report["records"],
                )
                write_once(case_root / f"trial-{trial}-metrics.json", result)
                results.append(result)
            arrays = {key: np.concatenate(values) for key, values in bank.items()}
            arrays["sample_ids"] = np.array(sample_ids, np.int32)
            arrays["training"] = np.full(
                len(sample_ids),
                course.course_id in ("outbound-north", "incoming-north", "incoming-east"),
                np.bool_,
            )
            path = case_root / "transitions.npz"
            np.savez_compressed(path, **arrays)
            check_pins(source_pinned)
            complete = dict(
                course=course.course_id,
                records=results,
                input_file_hashes=source_pinned,
                rows=len(sample_ids),
                dataset=dict(path=path.name, hash=digest(path)),
                native_steps=sum(r["native_steps"] for r in results),
                optimizer_updates=0,
                hardware_authorized=False,
                goal_complete=False,
            )
            write_once(case_root / "complete.json", complete)
            return dict(
                course=course.course_id,
                path=str(case_root / "complete.json"),
                hash=digest(case_root / "complete.json"),
                rows=len(sample_ids),
                native_steps=complete["native_steps"],
            )

        try:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
                futures = [workers.submit(one_course, i, c) for i, c in enumerate(courses)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    write_once(a.output / f"stage-{result['course']}.json", result)
                    results.append(result)
            check_pins(pins)
            write_once(
                a.output / "complete.json",
                dict(
                    status="NATIVE_ROLE_RESIDUAL_EXPLORATION_MEASURED_NOT_TRAINED",
                    source_head=a.expected_head,
                    input_file_hashes=pins,
                    plan_hash=a.plan_hash,
                    seed=a.seed,
                    initialization_hash=init["hash"],
                    identity_hash=identity["hash"],
                    training_hash=identity_report["training_hash"],
                    binding_hash=initialized["binding_hash"],
                    parent_hash=initialized["parent_hash"],
                    encoder_hash=initialized["encoder"]["hash"],
                    maximum_ratio=initialized["maximum_ratio"],
                    reward_contract_hash=residual_reward_hash(),
                    results=sorted(results, key=lambda r: base["courses"].index(r["course"])),
                    rows=sum(r["rows"] for r in results),
                    native_steps=sum(r["native_steps"] for r in results),
                    unique_initial_geometries=5,
                    optimizer_updates=0,
                    native_source_reexecutions_exact=True,
                    hardware_authorized=False,
                    promotion_authorized=False,
                    goal_complete=False,
                ),
            )
        except BaseException as error:
            write_once(
                a.output / "failed.json",
                dict(
                    error=repr(error),
                    known_completed_courses=results,
                    other_subprocess_outputs_retained=True,
                    blind_retry=False,
                    hardware_authorized=False,
                    goal_complete=False,
                ),
            )
            raise


if __name__ == "__main__":
    main()
