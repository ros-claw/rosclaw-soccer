"""Read-only raw review of three intent-from0 arms and full NN/native repeats."""

import argparse
import json
import subprocess
from pathlib import Path


def review_episode(world, raw, row):
    import numpy as np

    from rosclaw_soccer.providers.g1.sonic_runup import (
        ISAACLAB_TO_MUJOCO,
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.rsi.bilateral_entry import observed_contact_foot
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry
    from rosclaw_soccer.rsi.recorded_entry_review import validate_recording
    from scripts.rsi_motor_student_rollout import exact_array

    validate_recording(raw, frames=300, nbody=world.nbody)
    for before, after in (
        ("canonical_qpos", "target_body_canonical_qpos_after_substep"),
        ("canonical_qvel", "target_body_canonical_qvel_after_substep"),
    ):
        exact_array(raw[before][1:], raw[after][:-1, -1])
    _, _, scale = _sonic_control_parameters(1.0, (1.0,) * 29)
    exact_array(
        G1SonicRunupController.default_angles
        + raw["neural_action_isaac"][:, ISAACLAB_TO_MUJOCO] * scale,
        raw["joint_target_rad"],
    )
    diagnostic = diagnose_entry_prefix(world, {k: v[:20] for k, v in raw.items()})
    score = score_full_entry(world, {k: v[20:] for k, v in raw.items()})
    score["original_suffix_qualified"] = score["qualified"]
    score["common_prefix_physical_safe"] = diagnostic["prefix_physical_safe"]
    if not diagnostic["prefix_physical_safe"]:
        score["qualified"], score["cost"], score["physical_prefix_and_suffix_safe"] = (
            False,
            None,
            False,
        )
    if (
        row["prefix_diagnostics"] != diagnostic
        or row["score"] != score
        or row["causal_qualified"]
        is not bool(
            score["qualified"] and diagnostic["post_boundary_control_can_cause_first_touch"]
        )
        or row["observed_contact_foot"] != observed_contact_foot(world, raw, global_start_frame=0)
    ):
        raise ValueError("raw full episode score/contact/prefix attribution differs")
    report = row["report"]
    handoff = report["handoff_frame"]
    if (
        report["body"]["frames"] != 300
        or report["body"]["attempted_steps"] != 3000
        or report["body"]["integrated_steps"] != 3000
        or row["new_native_steps"] != 3000
        or row["new_optimizer_steps"] != 0
        or row["initial_state_adopted_from_other_episode"] is not False
        or row["known_intent_before_frame0"] is not True
    ):
        raise ValueError("whole actual from0 physical episode and accounting required")
    stage = raw["entry_stage"]
    if not np.isin(stage, (0, 1, 2)).all() or np.any(stage[:20] != 0):
        raise ValueError("only original prefix/entry/contact/timeout stages permitted")
    if handoff is None:
        if np.any(stage == 1):
            raise ValueError("no handoff cannot invent source tracking")
    elif (
        type(handoff) is not int
        or not 30 <= handoff <= 90
        or np.any(stage[:handoff] != 0)
        or np.any(stage[handoff:] != 1)
    ):
        raise ValueError("original global handoff clock/stages required")
    if row["source_window"] == [466, 767] and (
        report["source_window"] != [466, 767]
        or report["source_playback_speed_changed"] is not False
    ):
        raise ValueError("compact source identity/timing differs")
    touch = score["first_foot_contact_global_frame"]
    return dict(
        reference_side=row["reference_side"],
        law_index=row["law_index"],
        qualified=row["causal_qualified"],
        handoff_frame=handoff,
        reference_handoff_occurred=handoff is not None,
        first_touch_during_source_tracking=bool(
            handoff is not None and touch is not None and touch >= handoff
        ),
        first_touch_without_source_tracking=bool(
            touch is not None and (handoff is None or touch < handoff)
        ),
        first_foot_contact_frame=score["first_foot_contact_global_frame"],
        handoff_to_contact_frames=score["first_foot_contact_global_frame"] - handoff
        if handoff is not None and score["first_foot_contact_global_frame"] is not None
        else None,
        forward_m=score["forward_to_task_end_m"],
        lateral_m=score["lateral_to_task_end_m"],
        peak_vx_mps=score["maximum_ball_vx_mps"],
        maximum_nonfoot_force_n=score["maximum_nonfoot_force_n"],
        minimum_pelvis_z_m=score["minimum_pelvis_z_m"],
        minimum_upright=score["minimum_upright"],
        root_drift_after_contact_m=score["post_touch_maximum_root_drift_m"],
        actual_contact_foot_sides=row["observed_contact_foot"]["first_contact_foot_sides"],
    )


def review_raw(a, plan, imports, pins):
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.motion_entry_profile import build_motion_entry
    from scripts.rsi_motion_entry_episode import ARMS
    from scripts.rsi_motor_student_rollout import exact_array
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_pilot_motion_entry import compare_episode, compare_original_right
    from scripts.rsi_probe_compact_source import load_arrays

    runtime = imports["original", False][1]["runtime_file_hashes"]
    worlds = [Path(k) for k in runtime if k.endswith(".mjb")]
    csvs = [Path(k) for k in runtime if k.endswith(".csv")]
    priors = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(worlds) != 1 or len(csvs) != 1 or len(priors) != 1:
        raise ValueError("one unchanged world/source/prior required")
    world = mujoco.MjModel.from_binary_path(str(worlds[0]))
    poses = np.loadtxt(csvs[0], delimiter=",", skiprows=1, dtype=np.float64)
    stored = json.loads(priors[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("unchanged original entry prior required")
    old_effective = json.loads((a.old_case.parent / "declaration.json").read_text())[
        "teacher_evidence_hash"
    ]
    initial = None
    summaries = {}
    total = 0
    expected = [(side, law) for side in ("right", "left") for law in (0, 3, 4)]
    for arm, (window, transport) in ARMS.items():
        (root, d, c), (repeat, rd, rc) = imports[arm, False], imports[arm, True]
        if (
            d["effective_motion_reference_evidence_hash"]
            != rd["effective_motion_reference_evidence_hash"]
            or len(c["results"]) != 6
            or len(rc["results"]) != 6
        ):
            raise ValueError("matching effective reference/full six episodes required")
        if arm == "original" and d["effective_motion_reference_evidence_hash"] != old_effective:
            raise ValueError("original RHS reference input must exactly match old teacher")
        rows = []
        for (side, law), row, rerow in zip(expected, c["results"], rc["results"], strict=True):
            folder = root / f"{side}-law{law}"
            if (
                row["arm"] != arm
                or row["reference_side"] != side
                or row["law_index"] != law
                or row["source_window"] != list(window)
            ):
                raise ValueError("predeclared arm/side/law/source identity differs")
            if json.loads((folder / "complete.json").read_text()) != row or row[
                "native_hash"
            ] != digest(folder / "native300.npz"):
                raise ValueError("raw native artifact/case commitment differs")
            compare_episode(repeat / folder.name, folder, rerow)
            raw = load_arrays(folder / "native300.npz")
            prefix = load_arrays(folder / "own-prefix20.npz")
            if set(raw) != set(prefix):
                raise ValueError("all prefix20 field identities required")
            for k, v in prefix.items():
                exact_array(v, raw[k][:20])
            if initial is None:
                initial = (raw["canonical_qpos"][0].copy(), raw["canonical_qvel"][0].copy())
            exact_array(raw["canonical_qpos"][0], initial[0])
            exact_array(raw["canonical_qvel"][0], initial[1])
            binding, entry, source = build_motion_entry(
                poses,
                prior,
                world_hash=digest(worlds[0]),
                motion_artifact_hash=digest(csvs[0]),
                reference_side=side,
                source_window=window,
                transport=transport,
            )
            binding.validate(
                source,
                entry,
                world_hash=digest(worlds[0]),
                source_window=window,
                reference_side=side,
            )
            declaration = json.loads((folder / "declaration.json").read_text())
            if (
                declaration["motion_entry_binding"] != json.loads(json.dumps(binding.report()))
                or declaration["motion_entry_binding_hash"] != binding.contract_hash
                or row["motion_entry_binding_hash"] != binding.contract_hash
                or row["entry_affordance_hash"] != entry.contract_hash
                or row["entry_ball_offset_xy_m"] != list(entry.ball_offset_xy_m)
                or row["actual_teacher_reference_evidence_hash"]
                != d["effective_motion_reference_evidence_hash"]
                or declaration["actual_teacher_reference_evidence_hash"]
                != row["actual_teacher_reference_evidence_hash"]
            ):
                raise ValueError(
                    "motion configuration/entry estimate/effective source binding differs"
                )
            if arm == "original" and side == "right":
                compare_original_right(folder, a.old_case, row)
            if arm == "compact_unadapted":
                old_folder = imports["original", False][0] / folder.name
                old_raw = load_arrays(old_folder / "native300.npz")
                old_row = json.loads((old_folder / "complete.json").read_text())
                handoff = row["report"]["handoff_frame"]
                if (
                    handoff != old_row["report"]["handoff_frame"]
                    or row["report"]["handoff_native_state_unchanged_hash"]
                    != old_row["report"]["handoff_native_state_unchanged_hash"]
                ):
                    raise ValueError("source-only intervention changed pre-handoff timing/state")
                n = 300 if handoff is None else handoff
                for k, v in raw.items():
                    exact_array(v[:n], old_raw[k][:n])
                if handoff is not None:
                    for k in ("canonical_qpos", "canonical_qvel"):
                        exact_array(raw[k][n], old_raw[k][n])
            rows.append(review_episode(world, raw, row))
        for complete in (c, rc):
            if (
                complete["new_native_steps"] != 18000
                or sum(r["new_native_steps"] for r in complete["results"]) != 18000
            ):
                raise ValueError("all6 whole actual episodes per process must be charged")
            total += 18000
        summaries[arm] = dict(
            episodes=6,
            qualified_episodes=sum(r["qualified"] for r in rows),
            offline_best_of_six_could_succeed=any(r["qualified"] for r in rows),
            qualified_during_source_tracking=sum(
                r["qualified"] and r["first_touch_during_source_tracking"] for r in rows
            ),
            qualified_without_source_tracking=sum(
                r["qualified"] and r["first_touch_without_source_tracking"] for r in rows
            ),
            rows=rows,
        )
    if total != 108000:
        raise ValueError("exact paired pilot budget required")
    return dict(
        independent_world_contexts=1,
        case_index=21,
        reviewed_native_steps=total,
        arms=summaries,
        all_initial_physical_states_byte_equal=True,
        all_episodes_continuous_native_from0=True,
        learned_policy_improvement_proven=False,
        offline_best_of_six_not_deployed_actor=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("launch", "plan", "worker-source", "old-case", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "expected-launch-hash", "expected-plan-hash"):
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
    ):
        raise ValueError("clean frozen review and exact launch/plan required")
    plan = json.loads((a.plan / "complete.json").read_text())
    launch = json.loads(a.launch.read_text())
    if (
        plan["status"] != "INTENT_FROM0_CASE21_PILOT_DECLARED"
        or plan["course"]["course_index"] != 21
        or len(launch["jobs"]) != 6
        or len({j["unit"] for j in launch["jobs"]}) != 6
        or launch["source_head"] != plan["source_head"]
        or launch["plan_hash"] != a.expected_plan_hash
        or Path(launch["source"]) != a.worker_source
        or subprocess.check_output(
            ["git", "-C", str(a.worker_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != launch["source_head"]
        or subprocess.check_output(["git", "-C", str(a.worker_source), "status", "--porcelain"])
    ):
        raise ValueError("exact three producers/three actual NN/native repeats required")
    for dirname in ("src", "scripts"):
        for f in (a.worker_source / dirname).rglob("*.py"):
            if digest(f) != digest(Path.cwd() / f.relative_to(a.worker_source)):
                raise ValueError("read-only reviewer cannot change pilot/control math")
    pins = dict(plan["input_file_hashes"])
    if pins.get(str(a.old_case / "complete.json")) != digest(a.old_case / "complete.json"):
        raise ValueError("original RHS comparator identity differs")
    pins.update(
        {str(a.launch): a.expected_launch_hash, str(a.plan / "complete.json"): a.expected_plan_hash}
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
            new_native_steps=0,
            new_neural_calls=0,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    imports = {}
    try:
        for j in launch["jobs"]:
            arm, repeated = j["arm"], j["mode"] == "reproduce"
            if (
                arm not in plan["arms"]
                or j["mode"] not in ("produce", "reproduce")
                or (arm, repeated) in imports
            ):
                raise ValueError("unique declared arm/mode required")
            tag = arm.replace("_", "-")
            expected_unit = (
                f"rosclaw-motion-entry-pilot-{tag}-20261020"
                + ("-repeat" if repeated else "")
                + ".service"
            )
            if j["unit"] != expected_unit:
                raise ValueError("only allowlisted declared pilot unit may be observed")
            wait_for_terminal_collection(j["unit"], wait=True)
            root = Path(j["root"])
            d = json.loads((root / "declaration.json").read_text())
            c = json.loads((root / "complete.json").read_text())
            if (
                d["source_head"] != launch["source_head"]
                or c["source_head"] != launch["source_head"]
                or d["arm"] != arm
                or c["arm"] != arm
                or d["mode"] != j["mode"]
                or c["status"]
                != ("INTENT_FROM0_ARM_REPRODUCED" if repeated else "INTENT_FROM0_ARM_PRODUCED")
                or c["optimizer_steps"] != 0
                or c["hardware_authorized"] is not False
                or c["promotion_authorized"] is not False
            ):
                raise ValueError("actual terminal0 bound arm and repeat evidence required")
            for k, h in d["input_file_hashes"].items():
                if k in pins and pins[k] != h:
                    raise ValueError("conflicting raw/source lineage")
                pins[k] = h
            pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
            imports[arm, repeated] = (root, d, c)
        if set(imports) != {(arm, r) for arm in plan["arms"] for r in (False, True)}:
            raise ValueError("all six scientific jobs required")
        check_pins(pins)
        summary = review_raw(a, plan, imports, pins)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="INTENT_FROM0_CASE21_RAW_REPRODUCTION_AUTHENTICATED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                summary=summary,
                reviewed_native_steps=108000,
                new_native_steps=0,
                new_neural_calls=0,
                optimizer_steps=0,
                same_implementation_repeat_not_independent_algorithm=True,
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


if __name__ == "__main__":
    main()
