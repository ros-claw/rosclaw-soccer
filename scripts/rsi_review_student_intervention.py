"""Read-only raw recovery24 audit; no new NN/native physics or optimizer work."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("launch", "plan", "worker-source", "warm", "warm-repeat", "teacher-review", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "expected-head",
        "expected-launch-hash",
        "expected-plan-hash",
        "expected-teacher-review-hash",
    ):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or digest(a.launch) != a.expected_launch_hash
        or digest(a.plan / "complete.json") != a.expected_plan_hash
        or digest(a.teacher_review / "complete.json") != a.expected_teacher_review_hash
    ):
        raise ValueError(
            "clean frozen review/source and exact launch/plan/common20 evidence required"
        )
    launch = json.loads(a.launch.read_text())
    plan = json.loads((a.plan / "complete.json").read_text())
    prior = json.loads((a.teacher_review / "complete.json").read_text())
    if (
        plan["status"] != "ACTUAL_STUDENT30_TEACHER_RECOVERY24_DECLARED"
        or launch["source_head"] != plan["source_head"]
        or launch["plan_hash"] != a.expected_plan_hash
        or Path(launch["source"]) != a.worker_source
        or len(launch["jobs"]) != 16
        or len({j["unit"] for j in launch["jobs"]}) != 16
        or prior["status"] != "PRIVILEGED_TEACHER_FIXED24_REPRODUCTION_AUTHENTICATED"
        or subprocess.check_output(
            ["git", "-C", str(a.worker_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != launch["source_head"]
        or subprocess.check_output(["git", "-C", str(a.worker_source), "status", "--porcelain"])
    ):
        raise ValueError("exact frozen worker and all16 declared jobs required")
    for directory in ("src", "scripts"):
        for f in (a.worker_source / directory).rglob("*.py"):
            if digest(f) != digest(Path.cwd() / f.relative_to(a.worker_source)):
                raise ValueError("review must not change worker/helper numerical math")
    pins = dict(plan["input_file_hashes"])
    pins.update(
        {
            str(a.launch): a.expected_launch_hash,
            str(a.plan / "complete.json"): a.expected_plan_hash,
            str(a.teacher_review / "complete.json"): a.expected_teacher_review_hash,
        }
    )
    pins.update(
        {
            str(f): digest(f)
            for root in (Path.cwd() / "src", Path.cwd() / "scripts")
            for f in root.rglob("*.py")
        }
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            worker_source_head=launch["source_head"],
            launch_hash=a.expected_launch_hash,
            plan_hash=a.expected_plan_hash,
            new_native_steps=0,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    collected = {}
    try:
        for j in launch["jobs"]:
            match = re.fullmatch(
                r"rosclaw-student30-recovery-p([0-7])-20261020(-repeat)?\.service", j["unit"]
            )
            if not match:
                raise ValueError("only declared allowlisted recovery units may be observed")
            part, repeated = int(match[1]), bool(match[2])
            if (part, repeated) in collected:
                raise ValueError("duplicate part/role")
            wait_for_terminal_collection(j["unit"], wait=True)
            root = Path(j["root"])
            d = json.loads((root / "declaration.json").read_text())
            c = json.loads((root / "complete.json").read_text())
            expected = plan["indices"][part::8]
            if (
                d["source_head"] != launch["source_head"]
                or c["source_head"] != launch["source_head"]
                or d["indices"] != expected
                or c["indices"] != expected
                or d["part"] != part
                or d["mode"] != ("reproduce" if repeated else "probe")
                or c["same_implementation_neural_native_repeat"] is not repeated
                or c["status"]
                != (
                    "ACTUAL_STUDENT30_RECOVERY_SHARD_REPRODUCED"
                    if repeated
                    else "ACTUAL_STUDENT30_RECOVERY_SHARD_PRODUCED"
                )
                or c["optimizer_steps"] != 0
                or c["hardware_authorized"] is not False
                or c["promotion_authorized"] is not False
            ):
                raise ValueError("real completed same-source fixed-part producer/repeat required")
            for k, h in d["input_file_hashes"].items():
                if k in pins and pins[k] != h:
                    raise ValueError("conflicting declared raw/input/source lineage")
                pins[k] = h
            pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
            collected[part, repeated] = (root, d, c)
        if set(collected) != {(i, r) for i in range(8) for r in (False, True)}:
            raise ValueError("complete sixteen-part coverage required")
        check_pins(pins)
        summary = review_raw(a, plan, prior, collected, pins)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="ACTUAL_STUDENT30_RECOVERY24_RAW_REPRODUCTION_AUTHENTICATED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                summary=summary,
                reviewed_native_steps=summary["actual_new_native_steps_across_two_executions"],
                new_native_steps=0,
                optimizer_steps=0,
                reviewer_reexecutes_neural_or_physics=False,
                repeated_same_implementation_not_independent_algorithm=True,
                privileged_recovery_not_learned_actor=True,
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
                new_native_steps=0,
                optimizer_steps=0,
                automatic_physics_retry_authorized=False,
            ),
        )
        raise


def review_raw(a, plan, prior, collected, pins):
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.sonic_runup import (
        ISAACLAB_TO_MUJOCO,
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.rsi.bilateral_entry import choose_bilateral_candidate, observed_contact_foot
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.student_intervention_score import (
        fuse_physical_intervention30,
        score_student_intervention30,
    )
    from scripts.rsi_motor_student_rollout import exact_array
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_probe_student_intervention import compare_case, validate_warm_pair

    validate_warm_pair(a, pins, plan["indices"])
    declarations = collected[0, False][1]
    world_paths = [Path(k) for k in declarations["runtime_file_hashes"] if k.endswith(".mjb")]
    if len(world_paths) != 1:
        raise ValueError("one original compiled world required")
    world = mujoco.MjModel.from_binary_path(str(world_paths[0]))
    _, _, scale = _sonic_control_parameters(1.0, (1.0,) * 29)

    def load(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    old = {r["course_index"]: r for r in prior["summary"]["rows"]}
    rows = []
    for part in range(8):
        (first, fd, fc), (repeat, rd, rc) = collected[part, False], collected[part, True]
        if (
            fd["teacher_evidence_hash"] != rd["teacher_evidence_hash"]
            or len(fc["results"]) != 3
            or len(rc["results"]) != 3
        ):
            raise ValueError("same teacher evidence binding/full three-case shards required")
        for i, row, rerow in zip(
            plan["indices"][part::8], fc["results"], rc["results"], strict=True
        ):
            folder = first / f"course{i:03d}"
            if json.loads((folder / "complete.json").read_text()) != row:
                raise ValueError("case receipt differs from shard")
            compare_case(repeat / folder.name, folder, rerow)
            prefix = load(folder / "actual-student30.npz")
            for name in ("actual-student30.npz", "shadow-labels30.npz"):
                actual, warm = load(folder / name), load(a.warm / folder.name / name)
                if set(actual) != set(warm):
                    raise ValueError("qualified actual30 field identities required")
                for k, v in actual.items():
                    exact_array(v, warm[k])
            diagnostic = diagnose_entry_prefix(world, prefix, frames=30)
            if diagnostic != row["prefix_diagnostics"]:
                raise ValueError("raw preintervention safety/causality differs")
            eligible = (
                diagnostic["prefix_physical_safe"]
                and diagnostic["post_boundary_control_can_cause_first_touch"]
            )
            if not eligible:
                if (
                    row["status"] != "PREINTERVENTION_HISTORY_VETO"
                    or row["new_native_steps"] != 300
                    or row["forecast_native_steps"] != 0
                    or row["executed_native_steps"] != 300
                    or row["candidates"] != []
                    or row["selected"] is not None
                    or row["causal_qualified"] is not False
                    or row["denominator_retained"] is not True
                ):
                    raise ValueError("history veto must remain in24 and charge only warm native")
                chosen = None
                success = False
            else:
                if (
                    row["status"] != "STUDENT30_PRIVILEGED_RECOVERY_EXECUTED"
                    or len(row["candidates"]) != 9
                ):
                    raise ValueError("all nine futures and actual endpoint required")
                for index, c in enumerate(row["candidates"]):
                    data = load(folder / f"candidate{index}.npz")
                    if (
                        c["artifact_hash"] != digest(folder / f"candidate{index}.npz")
                        or json.loads((folder / f"candidate{index}.json").read_text()) != c
                    ):
                        raise ValueError("candidate raw artifact/receipt commitment differs")
                    joined = fuse_physical_intervention30(world, prefix, data)
                    metadata = dict(
                        neural_action_isaac=((270, 29), np.float32),
                        encoder_reference_poses=((270, 10, 36), np.float64),
                        entry_stage=((270,), np.int64),
                        entry_command=((270, 3), np.float64),
                        entry_target_root_xy_m=((270, 2), np.float64),
                        entry_error_m=((270,), np.float64),
                    )
                    if set(data) != set(joined) | set(metadata):
                        raise ValueError("complete original18 actual teacher270 fields required")
                    for k, (shape, dtype) in metadata.items():
                        if (
                            data[k].shape != shape
                            or data[k].dtype != dtype
                            or not np.isfinite(data[k]).all()
                        ):
                            raise ValueError("finite original typed teacher270 field required")
                    targets = (
                        G1SonicRunupController.default_angles
                        + data["neural_action_isaac"][:, ISAACLAB_TO_MUJOCO] * scale
                    )
                    exact_array(targets, data["joint_target_rad"])
                    if (
                        c["index"] != index
                        or c["score"] != score_student_intervention30(world, prefix, data)
                        or c["observed_contact_foot"]
                        != observed_contact_foot(world, joined, global_start_frame=0)
                        or c["report"]["actual_entry_start_frame"] != 30
                        or c["report"]["original_task_horizon"] != 300
                        or c["report"]["body"]["attempted_steps"] != 3000
                        or c["report"]["body"]["integrated_steps"] != 3000
                        or c["new_integrated_native_steps"] != 2700
                        or c["new_attempted_native_steps"] != 2700
                    ):
                        raise ValueError("raw future score/contact/report/native boundaries differ")
                chosen = choose_bilateral_candidate(row["candidates"])
                actual = load(folder / "selected-actual270.npz")
                forecast = load(folder / f"candidate{chosen}.npz")
                for k, v in actual.items():
                    exact_array(v, forecast[k])
                joined = fuse_physical_intervention30(world, prefix, actual)
                physical = load(folder / "selected-physical300.npz")
                if set(physical) != set(joined) or set(actual) != set(forecast):
                    raise ValueError("all actual fields required")
                for k, v in joined.items():
                    exact_array(v, physical[k])
                score = score_student_intervention30(world, prefix, actual)
                success = score["qualified"]
                if (
                    row["selected"] != chosen
                    or row["selected_score"] != score
                    or row["causal_qualified"] is not success
                    or row["forecast_native_steps"] != 24300
                    or row["executed_native_steps"] != 3000
                    or row["new_native_steps"] != 27300
                    or row["actual_all18_and_report_byte_exact_to_own_future"] is not True
                ):
                    raise ValueError("actual score/choice/allocation differs")
            if i not in old or old[i]["actor_outcomes"]["seed43"] is not False:
                raise ValueError("original fixed24 seed43 baseline identity differs")
            rows.append(
                dict(
                    course_index=i,
                    stratum=old[i]["stratum"],
                    eligible_at_actual30=bool(eligible),
                    qualified=bool(success),
                    selected_option=chosen,
                    common20_teacher_qualified=old[i]["teacher_qualified"],
                    first_foot_before30=not diagnostic[
                        "post_boundary_control_can_cause_first_touch"
                    ],
                    preintervention30_safe=diagnostic["prefix_physical_safe"],
                    native_steps_per_execution=row["new_native_steps"],
                )
            )
        for c in (fc, rc):
            if c["new_native_steps"] != sum(r["new_native_steps"] for r in c["results"]):
                raise ValueError("whole shard physical accounting differs")
    if sorted(r["course_index"] for r in rows) != plan["indices"]:
        raise ValueError("whole24 denominator required")
    charged = 2 * sum(r["native_steps_per_execution"] for r in rows)
    if charged > 1310400:
        raise ValueError("preregistered physical bound exceeded")
    rows.sort(key=lambda r: r["course_index"])
    return dict(
        contexts=24,
        eligible_at_actual30=sum(r["eligible_at_actual30"] for r in rows),
        teacher_recoveries_from_student30=sum(r["qualified"] for r in rows),
        old_seed43_successes=0,
        common20_teacher_successes=sum(r["common20_teacher_qualified"] for r in rows),
        actual_new_native_steps_across_two_executions=charged,
        rows=rows,
        attribution_includes_10_student_frames_and_10_less_remaining_frames=True,
        learned_policy_growth_proven=False,
        canonical_promotion_authorized=False,
    )


if __name__ == "__main__":
    main()
