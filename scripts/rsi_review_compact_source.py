"""Raw six-option source-lead audit; read-only, zero NN or integrated steps."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def audit_candidate(world, prefix, data, c, old_data, old_c):
    import numpy as np

    from rosclaw_soccer.providers.g1.sonic_runup import (
        ISAACLAB_TO_MUJOCO,
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.rsi.bilateral_entry import observed_contact_foot
    from rosclaw_soccer.rsi.student_intervention_score import (
        fuse_physical_intervention30,
        score_student_intervention30,
    )
    from scripts.rsi_motor_student_rollout import exact_array

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
        raise ValueError("complete original18 teacher270 fields required")
    for k, (shape, dtype) in metadata.items():
        if data[k].shape != shape or data[k].dtype != dtype or not np.isfinite(data[k]).all():
            raise ValueError("finite typed teacher270 metadata required")
    _, _, scale = _sonic_control_parameters(1.0, (1.0,) * 29)
    exact_array(
        G1SonicRunupController.default_angles
        + data["neural_action_isaac"][:, ISAACLAB_TO_MUJOCO] * scale,
        data["joint_target_rad"],
    )
    report = c["report"]
    handoff = report["handoff_frame"]
    if not np.isin(data["entry_stage"], (0, 1, 2)).all():
        raise ValueError("only original entry/contact/timeout stages permitted")
    if (
        report["source_window"] != [466, 767]
        or report["source_lead_protocol"] != "compact60_before_declared526_event_original120Hz"
        or report["source_playback_speed_changed"] is not False
        or report["actual_entry_start_frame"] != 30
        or report["original_task_horizon"] != 300
        or report["body"]["attempted_steps"] != 3000
        or report["body"]["integrated_steps"] != 3000
        or c["new_integrated_native_steps"] != 2700
        or c["new_attempted_native_steps"] != 2700
        or c["score"] != score_student_intervention30(world, prefix, data)
        or c["observed_contact_foot"] != observed_contact_foot(world, joined, global_start_frame=0)
    ):
        raise ValueError("raw compact future score/contact/time/count differs")
    if (
        handoff != old_c["report"]["handoff_frame"]
        or report["handoff_native_state_unchanged_hash"]
        != old_c["report"]["handoff_native_state_unchanged_hash"]
    ):
        raise ValueError("source lead must not alter pre-handoff timing or native state")
    if handoff is not None and (type(handoff) is not int or not 30 <= handoff <= 90):
        raise ValueError("original global30..90 entry deadline required")
    n = 270 if handoff is None else handoff - 30
    if handoff is None:
        if np.any(data["entry_stage"] == 1):
            raise ValueError("no source handoff must not invent contact stage")
    elif np.any(data["entry_stage"][:n] != 0) or np.any(data["entry_stage"][n:] != 1):
        raise ValueError("raw stages must preserve actual reported handoff boundary")
    if set(data) != set(old_data):
        raise ValueError("same old/new field identities required")
    for k, v in data.items():
        exact_array(v[:n], old_data[k][:n])
    if handoff is not None:
        for k in ("canonical_qpos", "canonical_qvel"):
            exact_array(data[k][n], old_data[k][n])
    return joined


def review_raw(a, plan, old, imports, pins):
    import mujoco

    from rosclaw_soccer.rsi.bilateral_entry import observed_contact_foot
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.source_lead_protocol import (
        ORIGINAL_NINE_ZERO_C_INDICES,
        choose_zero_c_six,
        matched_old_six_selection,
    )
    from rosclaw_soccer.rsi.student_intervention_score import (
        fuse_physical_intervention30,
        score_student_intervention30,
    )
    from scripts.rsi_motor_student_rollout import exact_array
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_probe_compact_source import (
        compare_old_six,
        compare_six_case,
        exact_artifact,
        load_arrays,
    )

    oldlaunch = json.loads(a.old_launch.read_text())
    oldcases = {}
    for job in oldlaunch["jobs"]:
        if not job["dependencies"]:
            root = Path(job["root"])
            for row in json.loads((root / "complete.json").read_text())["results"]:
                i = row["course_index"]
                if i in oldcases:
                    raise ValueError("unique old actual case required")
                oldcases[i] = (root / f"course{i:03d}", row)
    if set(oldcases) != set(plan["indices"]):
        raise ValueError("all matched old24 actual cases required")
    baseline = [
        dict(course_index=i, **matched_old_six_selection(oldcases[i][1])) for i in plan["indices"]
    ]
    if baseline != plan["matched_old_baseline"]:
        raise ValueError("declared six-option matching differs from old raw costs")
    zr, zd, zc = imports["zero"]
    zfirst = json.loads((zr / "first/course021/complete.json").read_text())
    zsecond = json.loads((zr / "second/course021/complete.json").read_text())
    compare_old_six(zr / "first/course021", oldcases[21][0], zfirst)
    compare_old_six(zr / "second/course021", oldcases[21][0], zsecond)
    compare_six_case(zr / "second/course021", zr / "first/course021", zsecond)
    if (
        zc["new_native_steps"] != 38400
        or zfirst["new_native_steps"] + zsecond["new_native_steps"] != 38400
    ):
        raise ValueError("both actual19200 old-six qualifications must be charged")
    # The original zero runner writes a new declaration hash but deliberately
    # reuses old effective reference evidence inside its parity executions.
    # Preserve both; never present the header hash as the effective NN input.
    effective = json.loads((oldcases[21][0].parent / "declaration.json").read_text())[
        "teacher_evidence_hash"
    ]
    worldpaths = [
        Path(k) for k in imports[0, False][1]["runtime_file_hashes"] if k.endswith(".mjb")
    ]
    if len(worldpaths) != 1:
        raise ValueError("one bound original compiled world required")
    world = mujoco.MjModel.from_binary_path(str(worldpaths[0]))
    old_strata = {r["course_index"]: r["stratum"] for r in old["summary"]["rows"]}
    rows = []
    charged = 0
    for part in range(8):
        (first, fd, fc), (repeat, rd, rc) = imports[part, False], imports[part, True]
        if (
            fd["teacher_evidence_hash"] != rd["teacher_evidence_hash"]
            or len(fc["results"]) != 3
            or len(rc["results"]) != 3
        ):
            raise ValueError("same full three-case source/evidence producer and repeat required")
        for i, row, rerow in zip(
            plan["indices"][part::8], fc["results"], rc["results"], strict=True
        ):
            folder = first / f"course{i:03d}"
            if json.loads((folder / "complete.json").read_text()) != row:
                raise ValueError("case report differs from shard")
            compare_six_case(repeat / folder.name, folder, rerow)
            oldfolder, oldrow = oldcases[i]
            for name in ("actual-student30.npz", "shadow-labels30.npz"):
                exact_artifact(folder / name, oldfolder / name)
            prefix = load_arrays(folder / "actual-student30.npz")
            diagnostic = diagnose_entry_prefix(world, prefix, frames=30)
            if diagnostic != row["prefix_diagnostics"]:
                raise ValueError("raw actual prefix30 safety/first-contact differs")
            eligible = (
                diagnostic["prefix_physical_safe"]
                and diagnostic["post_boundary_control_can_cause_first_touch"]
            )
            if not eligible:
                if (
                    row["status"] != "PREINTERVENTION_HISTORY_VETO"
                    or row["selected"] is not None
                    or row["forecast_native_steps"] != 0
                    or row["executed_native_steps"] != 300
                    or row["causal_qualified"] is not False
                    or row["new_native_steps"] != 300
                    or row["candidates"]
                ):
                    raise ValueError("history veto must stay in denominator and charge only300")
                score = None
                success = False
                latency = None
            else:
                if (
                    row["status"] != "STUDENT30_PRIVILEGED_RECOVERY_EXECUTED"
                    or row["source_window"] != [466, 767]
                    or row["option_protocol"] != "zero_c_six"
                    or len(row["candidates"]) != 6
                ):
                    raise ValueError("single compact window and all fixed six options required")
                for index, c in enumerate(row["candidates"]):
                    path = folder / f"candidate{index}.npz"
                    if (
                        c["artifact_hash"] != digest(path)
                        or json.loads((folder / f"candidate{index}.json").read_text()) != c
                    ):
                        raise ValueError("raw candidate artifact/receipt differs")
                    oldindex = ORIGINAL_NINE_ZERO_C_INDICES[index]
                    audit_candidate(
                        world,
                        prefix,
                        load_arrays(path),
                        c,
                        load_arrays(oldfolder / f"candidate{oldindex}.npz"),
                        oldrow["candidates"][oldindex],
                    )
                selected = choose_zero_c_six(row["candidates"])
                c = row["candidates"][selected]
                exact_artifact(
                    folder / "selected-actual270.npz", folder / f"candidate{selected}.npz"
                )
                data = load_arrays(folder / "selected-actual270.npz")
                joined = fuse_physical_intervention30(world, prefix, data)
                actual = load_arrays(folder / "selected-physical300.npz")
                if set(joined) != set(actual):
                    raise ValueError("complete physical300 fields required")
                for k, v in joined.items():
                    exact_array(v, actual[k])
                score = score_student_intervention30(world, prefix, data)
                success = score["qualified"]
                if (
                    row["selected"] != selected
                    or row["selected_score"] != score
                    or row["selected_report"] != c["report"]
                    or row["causal_qualified"] is not success
                    or row["new_native_steps"] != 19200
                    or row["forecast_native_steps"] != 16200
                    or row["executed_native_steps"] != 3000
                    or row["observed_contact_foot"]
                    != observed_contact_foot(world, joined, global_start_frame=0)
                ):
                    raise ValueError("actual choice/report/score/count/contact differs")
                handoff = c["report"]["handoff_frame"]
                touch = score["first_foot_contact_global_frame"]
                latency = touch - handoff if touch is not None and handoff is not None else None
            oldscore = oldrow["selected_score"]
            oldhandoff = oldrow["candidates"][oldrow["selected"]]["report"]["handoff_frame"]
            oldtouch = oldscore["first_foot_contact_global_frame"]
            rows.append(
                dict(
                    course_index=i,
                    stratum=old_strata[i],
                    old_qualified=oldrow["causal_qualified"],
                    new_qualified=bool(success),
                    old_physical_safe=oldscore["physical_prefix_and_suffix_safe"],
                    new_physical_safe=False
                    if score is None
                    else score["physical_prefix_and_suffix_safe"],
                    old_handoff_to_touch_frames=oldtouch - oldhandoff
                    if oldtouch is not None and oldhandoff is not None
                    else None,
                    new_handoff_to_touch_frames=latency,
                    new_score=score,
                    old_score=oldscore,
                )
            )
        for c in (fc, rc):
            if c["new_native_steps"] != sum(r["new_native_steps"] for r in c["results"]):
                raise ValueError("full shard charge differs")
            charged += c["new_native_steps"]
    if sorted(r["course_index"] for r in rows) != plan["indices"] or charged > 921600:
        raise ValueError("full24 denominator and candidate physics bound required")
    rows.sort(key=lambda r: r["course_index"])
    return dict(
        contexts=24,
        old_success=sum(r["old_qualified"] for r in rows),
        new_success=sum(r["new_qualified"] for r in rows),
        rescues=[r["course_index"] for r in rows if r["new_qualified"] and not r["old_qualified"]],
        regressions=[
            r["course_index"] for r in rows if r["old_qualified"] and not r["new_qualified"]
        ],
        added_unsafe=[
            r["course_index"] for r in rows if r["old_physical_safe"] and not r["new_physical_safe"]
        ],
        candidate_native_steps=charged,
        qualification_native_steps=38400,
        rows=rows,
        zero_declared_teacher_evidence_hash=zd["teacher_evidence_hash"],
        zero_effective_teacher_reference_evidence_hash=effective,
        zero_header_hash_is_not_effective_reference_binding=zd["teacher_evidence_hash"]
        != effective,
        prehandoff_numeric_equality_verified_for_all_options=True,
        learned_actor_growth_proven=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("launch", "plan", "worker-source", "old-review", "old-launch", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "expected-head",
        "expected-launch-hash",
        "expected-plan-hash",
        "expected-old-review-hash",
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
        or digest(a.old_review / "complete.json") != a.expected_old_review_hash
    ):
        raise ValueError(
            "frozen read-only audit source and exact launch/plan/old evidence required"
        )
    launch = json.loads(a.launch.read_text())
    plan = json.loads((a.plan / "complete.json").read_text())
    old = json.loads((a.old_review / "complete.json").read_text())
    if (
        len(launch["jobs"]) != 17
        or len({j["unit"] for j in launch["jobs"]}) != 17
        or plan["status"] != "COMPACT_SOURCE_LEAD24_DECLARED"
        or plan["options"] != 6
        or plan["source_head"] != launch["source_head"]
        or Path(launch["source"]) != a.worker_source
        or launch["plan_hash"] != a.expected_plan_hash
        or plan["candidate_source_window"] != [466, 767]
        or plan["original_source_window"] != [286, 767]
        or old["status"] != "ACTUAL_STUDENT30_RECOVERY24_RAW_REPRODUCTION_AUTHENTICATED"
        or subprocess.check_output(
            ["git", "-C", str(a.worker_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != launch["source_head"]
        or subprocess.check_output(["git", "-C", str(a.worker_source), "status", "--porcelain"])
    ):
        raise ValueError("complete fixed zero plus16 source-bound scientific jobs required")
    for dirname in ("src", "scripts"):
        for f in (a.worker_source / dirname).rglob("*.py"):
            if digest(f) != digest(Path.cwd() / f.relative_to(a.worker_source)):
                raise ValueError("audit must preserve exact worker/control math")
    pins = dict(plan["input_file_hashes"])
    if pins.get(str(a.old_launch)) != digest(a.old_launch):
        raise ValueError("original baseline launch identity differs")
    pins.update(
        {
            str(a.launch): a.expected_launch_hash,
            str(a.plan / "complete.json"): a.expected_plan_hash,
            str(a.old_review / "complete.json"): a.expected_old_review_hash,
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
            new_native_steps=0,
            new_neural_calls=0,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    imports = {}
    try:
        for j in launch["jobs"]:
            if j["unit"] == "rosclaw-compact-source-old6-parity-20261020.service":
                key = "zero"
                expected_mode = "zero"
                expected = [21]
                status = "ORIGINAL_SOURCE_SIX_ACTUAL30_NUMERIC_PARITY"
            else:
                m = re.fullmatch(
                    r"rosclaw-compact-source-p([0-7])-20261020(-repeat)?\.service", j["unit"]
                )
                if not m:
                    raise ValueError("only allowlisted declared compact units may be observed")
                key = (int(m[1]), bool(m[2]))
                expected_mode = "reproduce" if key[1] else "probe"
                expected = plan["indices"][key[0] :: 8]
                status = (
                    "COMPACT_SOURCE_LEAD_SHARD_REPRODUCED"
                    if key[1]
                    else "COMPACT_SOURCE_LEAD_SHARD_PRODUCED"
                )
            if key in imports:
                raise ValueError("duplicate zero/part/role")
            wait_for_terminal_collection(j["unit"], wait=True)
            root = Path(j["root"])
            d = json.loads((root / "declaration.json").read_text())
            c = json.loads((root / "complete.json").read_text())
            if (
                d["source_head"] != launch["source_head"]
                or c["source_head"] != launch["source_head"]
                or c["status"] != status
                or d["mode"] != expected_mode
                or d["part"] != (None if key == "zero" else key[0])
                or d["indices"] != expected
                or c["indices"] != expected
                or c["optimizer_steps"] != 0
                or c["hardware_authorized"] is not False
                or c["promotion_authorized"] is not False
            ):
                raise ValueError("actual terminal0 bound complete source/part/mode required")
            for k, h in d["input_file_hashes"].items():
                if k in pins and pins[k] != h:
                    raise ValueError("conflicting raw/source lineage")
                pins[k] = h
            pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
            imports[key] = (root, d, c)
        if set(imports) != {"zero"} | {(i, r) for i in range(8) for r in (False, True)}:
            raise ValueError("all17 complete scientific stages required")
        check_pins(pins)
        result = review_raw(a, plan, old, imports, pins)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="COMPACT_SOURCE_LEAD24_RAW_REPRODUCTION_AUTHENTICATED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                summary=result,
                new_native_steps=0,
                new_neural_calls=0,
                optimizer_steps=0,
                reviewed_native_steps=result["candidate_native_steps"]
                + result["qualification_native_steps"],
                same_implementation_repetition_not_independent_algorithm=True,
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
