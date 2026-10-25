"""Independent full native readback of old/new task-response teacher; no stepping."""

import argparse
import json
from pathlib import Path


def verify_target_edits(edits, targets, *, start, plan, codec, limits):
    """Read raw codec commitments; this is not a second implementation of the NN."""
    import numpy as np

    default, scale, mapping, action_lo, action_hi = codec
    offsets = plan.offsets()
    if len(edits) != len(targets):
        raise ValueError("one target/history commitment for every controlled NN tick required")
    for j, (edit, target) in enumerate(zip(edits, targets, strict=True)):
        control = start + j
        local = control - plan.start_control
        requested = offsets[local] if 0 <= local < len(offsets) else np.zeros(29)
        original = np.array(edit["original_nn_target_hg_rad"], np.float64)
        raw = np.array(edit["original_raw_action_il"], np.float32)
        executed = np.array(edit.get("executed_raw_action_il", raw), np.float32)
        if (
            edit["control_step"] != control
            or type(edit["applied"]) is not bool
            or type(edit["fallback"]) is not bool
            or (edit["applied"] and edit["fallback"])
            or any(
                edit[k]
                for k in (
                    "physical_state_modified",
                    "reference_modified",
                    "neural_weights_modified",
                    "hardware_authorized",
                )
            )
            or not np.array_equal(np.array(edit["requested_offset_hg_rad"]), requested)
            or any(
                a.shape != (29,) or not np.isfinite(a).all()
                for a in (original, raw, executed, target)
            )
            or not np.array_equal((default + scale * raw[mapping]).astype(np.float64), original)
            or not np.array_equal((default + scale * executed[mapping]).astype(np.float64), target)
            or (executed < action_lo).any()
            or (executed > action_hi).any()
        ):
            raise ValueError("actual target/action history codec or absolute plan clock differs")
        difference = target - original
        changed = difference != 0
        if (
            (abs(difference) > 0.060001).any()
            or (target[changed] < limits[changed, 0]).any()
            or (target[changed] > limits[changed, 1]).any()
            or (changed & (requested == 0)).any()
            or bool(changed.any()) != edit["applied"]
        ):
            raise ValueError("encoded actual target exceeds original declared authority")
        if edit["applied"] and (
            not edit["next_action_history_matches_executed_codec"]
            or not np.array_equal(np.array(edit["applied_offset_hg_rad"]), difference)
            or edit["maximum_target_offset_rad"] != float(abs(difference).max())
        ):
            raise ValueError("recorded applied target/history commitment differs")


def verify_sequential_response(windows, forecasts, recorded):
    """Replay the finite optimizer from raw-derived observations, without physics."""
    import numpy as np

    from rosclaw_soccer.rsi.sequential_joint_response import search_sequential_joint_response
    from rosclaw_soccer.sim.contracts import hash_json

    cursor = 37

    def evaluate(name, coefficients):
        nonlocal cursor
        if (
            cursor >= len(windows)
            or windows[cursor]["name"] != name
            or not np.array_equal(coefficients, np.array(windows[cursor]["coefficients"]))
        ):
            raise ValueError(
                "sequential signed probes/center coefficients differ from physical search"
            )
        value = (forecasts[cursor], windows[cursor]["outcome"])
        cursor += 1
        return value

    reconstructed = search_sequential_joint_response(forecasts[0], windows[0]["outcome"], evaluate)
    if cursor != len(windows) or hash_json(reconstructed) != hash_json(recorded):
        raise ValueError("sequential NN feedback/acceptance/fixed normalization report differs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("pilot", "champion-root", "backbone-model", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--pilot-hash", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    cp = a.pilot / "complete.json"
    if a.output.exists() or digest(cp) != a.pilot_hash:
        raise ValueError("unused audit output and exact completed physical pilot required")
    report = json.loads(cp.read_text())
    dp = a.pilot / "declaration.json"
    dec = json.loads(dp.read_text())
    sequential = dec.get("sequential_restoration", False)
    aim = dec.get("aim_command_search", False)
    executed = dec.get("executed_target_search", False)
    if (
        any(type(v) is not bool for v in (sequential, aim, executed))
        or sum((sequential, aim, executed)) > 1
    ):
        raise ValueError("explicit boolean response protocol required")
    candidate_cap = 73 if executed else 57 if aim else 273 if sequential else 96
    forecast_cap = 6 * (37 + candidate_cap) * 1280
    if (
        report["status"]
        != (
            "FULL_NN_EXECUTED_TARGET_PILOT_COMPLETE"
            if executed
            else "FULL_NN_AIM_COMMAND_PILOT_COMPLETE"
            if aim
            else "FULL_NN_JOINT_TASK_RESPONSE_PILOT_COMPLETE"
        )
        or digest(dp) != report["declaration_hash"]
        or dec["course_indices"] != [0, 2, 8, 22, 27, 30]
        or dec["requested_forecast_native_step_ceiling"] != forecast_cap
        or dec["actual_world_ceiling"] != 12
        or dec["actual_native_step_ceiling"] != 43200
        or [r["course_index"] for r in report["results"]] != dec["course_indices"]
    ):
        raise ValueError("fixed whole six-state paired physics pilot required")
    if sequential and any(
        dec.get(k) != v
        for k, v in dict(
            maximum_response_rounds=4,
            initial_force_normalization_fixed=True,
            require_observed_ghost_progress_for_new_center=True,
            new_candidate_ceiling=273,
            proposal_l1_trust_rad=0.06,
            proposal_axis_trust_rad=0.03,
        ).items()
    ):
        raise ValueError("frozen sequential restoration declaration differs")
    pins = dict(report["input_file_hashes"])
    pins[str(cp)], pins[str(dp)] = a.pilot_hash, report["declaration_hash"]
    source_pins(pins)
    check_pins(pins)
    from scripts.rsi_review_motion_timing_teacher import verify_synergy_search
    from scripts.rsi_teacher_physics_context import load_teacher_context

    context = load_teacher_context(pins, a.backbone_model)
    import numpy as np

    from rosclaw_soccer.rsi.aim_command_kick_player import declared_aim_commands
    from rosclaw_soccer.rsi.executed_target_synergy import ExecutedTargetSynergy
    from rosclaw_soccer.rsi.joint_task_response import (
        propose_joint_task_response,
        response_features,
    )
    from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    lo = context["layout"]
    codec = None
    if executed:
        from importlib import import_module

        import yaml

        cfgs = [Path(k) for k in pins if k.endswith("/policy/robonaldo/config/freekick.yaml")]
        if len(cfgs) != 1:
            raise ValueError("one pinned original float32 action encoder configuration required")
        cfg = yaml.safe_load(cfgs[0].read_text())
        module = import_module(context["types"][2].__module__)
        action_lo, action_hi = module._compile_action_clip_bounds(cfg, cfg.get("clip_actions", 3.0))
        codec = (
            np.array(cfg["default_joint_pos"], np.float32),
            np.array(cfg["action_scale"], np.float32),
            np.array(context["types"][3]),
            action_lo,
            action_hi,
        )
        if any(
            dec.get(k) != v
            for k, v in dict(
                new_candidate_ceiling=73,
                new_probe_axes=6,
                temporal_lobe="two24interval_lobes",
                second_lobe_coefficients_zero=False,
                proposal_axis_trust_rad=0.06,
            ).items()
        ):
            raise ValueError("frozen bounded actual-target protocol differs")
    aim_grid = declared_aim_commands()
    if aim and (
        dec["aim_command_world_points"] != [list(c.world_point) for c in aim_grid]
        or dec["task_scoring_goal_world_point"] != [8.0, 0.0, 0.11]
    ):
        raise ValueError("fixed20 desired command grid without task-goal mutation required")
    original_cp = a.champion_root / "complete.json"
    if pins.get(str(original_cp)) != digest(original_cp):
        raise ValueError("bound original full teacher bank required")
    champion = json.loads(original_cp.read_text())

    def record(path, expected, root):
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError("confined exact control record required")
        pins[str(path)] = expected
        row = json.loads(path.read_text())
        return row, raw(path.parent, row["artifact"], root)

    def raw(folder, ref, root):
        path = folder / ref["path"]
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != ref["hash"]:
            raise ValueError("confined exact raw native array required")
        pins[str(path)] = ref["hash"]
        with np.load(path, allow_pickle=False) as saved:
            return {k: saved[k].copy() for k in saved.files}

    def same(before, after):
        return set(before) == set(after) and all(
            np.array_equal(before[k], after[k]) for k in before
        )

    def future_exact(prediction, actual, frame, n):
        for k in ("qpos", "qvel", "torque", "ball_forces"):
            if not np.array_equal(prediction[k], actual[k][frame * 10 : (frame + n) * 10]):
                raise ValueError("predicted native future differs from actual")
        if not np.array_equal(
            prediction["joint_targets"], actual["joint_targets"][frame : frame + n]
        ):
            raise ValueError("predicted future NN targets differ from actual")

    results = []
    native = charged = integrated = worlds = 0
    rescues, regressions = [], []
    for item in report["results"]:
        i = item["course_index"]
        actual = {}
        outcomes = {}
        health = {}
        parent_forecasts = {}
        old = {}
        for arm in ("B", "C"):
            ref = champion["results"][i]["arms"][arm]
            old[arm] = record(a.champion_root / ref["record"], ref["hash"], a.champion_root)
        for arm in ("B", "C"):
            ref = item["arms"][arm]
            path = a.pilot / ref["record"]
            row, arrays = record(path, ref["hash"], a.pilot)
            if any(
                row[k] != old["C"][0][k]
                for k in ("course", "parameters", "experimental_fast_approach", "entry_control")
            ):
                raise ValueError("actual same-state comparison changed course or approach")
            outcome = relabel_arrays(
                arrays, controls=row["controls"], ball_address=lo.bq, root_address=lo.rq
            )
            pelvis = arrays["qpos"][:, lo.rq : lo.rq + 7]
            healthy = bool(
                row["controls"] == 360
                and (pelvis[:, 2] >= 0.65).all()
                and (1 - 2 * (pelvis[:, 4] ** 2 + pelvis[:, 5] ** 2) >= 0.9).all()
            )
            if (
                hash_json(outcome) != hash_json(row["outcome"])
                or healthy != row["healthy_full_horizon"]
            ):
                raise ValueError("full actual body/task label differs")
            if any(
                ref[k] != value
                for k, value in (
                    ("task_pass", outcome["task_pass"]),
                    ("safe", outcome["safe"]),
                    ("healthy", healthy),
                )
            ):
                raise ValueError("producer summary differs from raw physical outcomes")
            if (abs(arrays["torque"][:, lo.ai]) > 0.85 * context["hard_limits"] + 1e-12).any():
                raise ValueError("unchanged hard native torque cap violated")
            if arm == "B" and not same(arrays, old["C"][1]):
                raise ValueError("old whole actual champion did not repeat exactly")
            if arm == "C":
                entry = row["entry_control"]
                for k in ("qpos", "qvel", "torque", "ball_forces"):
                    if not np.array_equal(arrays[k][: entry * 10], actual["B"][k][: entry * 10]):
                        raise ValueError("paired pre-proposal measured prefix differs")
            events = row["controller_events"]
            if len(events) != 1:
                raise ValueError("one actual measured entry decision required")
            event = events[0]
            forecasts = []
            windows = []
            peaks = []
            charged_arm = integrated_arm = 0
            for j, f in enumerate(row["candidate_forecasts"]):
                pred = raw(path.parent, f["artifact"], a.pilot)
                frame, n = f["donor_control_frame"], f["controls"]
                requested = min(128, 360 - frame)
                if (
                    frame != row["entry_control"]
                    or f["requested_controls"] != requested
                    or not 1 <= n <= requested
                ):
                    raise ValueError("exact actual entry and bounded forecast horizon required")
                pred.update(controls=n, donor_control_frame=frame)
                window = predicted_touch_window(
                    pred,
                    ball_address=lo.bq,
                    root_address=lo.rq,
                    start_control=frame,
                    diagnostic_tail=True,
                )
                if n < requested and window["predicted_body_healthy"]:
                    raise ValueError("unexplained forecast censoring")
                windows.append(
                    dict(
                        name=f["name"],
                        outcome=window,
                        coefficients=event["candidate_windows"][j]["coefficients"],
                        **(
                            dict(aim_world_point=event["candidate_windows"][j]["aim_world_point"])
                            if aim and arm == "C"
                            else {}
                        ),
                    )
                )
                peaks.append(float(pred["ball_forces"][:, 0, 2].max()))
                forecasts.append(pred)
                if executed:
                    if arm == "C" and j >= 37:
                        plan = ExecutedTargetSynergy(
                            frame,
                            tuple(tuple(float(x) for x in r) for r in context["basis"]),
                            tuple(tuple(float(x) for x in r) for r in windows[-1]["coefficients"]),
                        )
                        verify_target_edits(
                            f["target_edits"],
                            pred["joint_targets"],
                            start=frame,
                            plan=plan,
                            codec=codec,
                            limits=context["model"].jnt_range[lo.joints],
                        )
                    elif f["target_edits"]:
                        raise ValueError(
                            "old reference branch must not silently use target authority"
                        )
                charged_arm += requested * 10
                integrated_arm += n * 10
            if hash_json(windows) != hash_json(event["candidate_windows"]):
                raise ValueError("raw predicted outcome labels differ")
            old_choice = verify_synergy_search(windows[:37], peaks[:37], event["generations"])
            old_pass = bool(old_choice or windows[0]["outcome"]["window_pass"])
            if arm == "B" or old_pass:
                expected = 1 if windows[0]["outcome"]["window_pass"] else 37
                if len(windows) != expected or event["joint_task_response"] is not None:
                    raise ValueError("old qualified branch must prohibit new search")
            elif executed:
                if len(windows) != 73 or event["joint_task_response"] is not None:
                    raise ValueError(
                        "36 bounded executed-target plans required after old abstention"
                    )
                additional = event["additional_candidate_search"]
                synthetic = [windows[0]]
                for w in windows[37:]:
                    if not w["name"].startswith("target-"):
                        raise ValueError("explicit target candidate namespace required")
                    synthetic.append(dict(w, name=w["name"][7:]))
                verify_synergy_search(synthetic, [peaks[0], *peaks[37:]], additional["generations"])
                expected_report = dict(
                    schema="soccer.executed_target_synergy_search.v1",
                    candidate_count=36,
                    generations=additional["generations"],
                    seed=7001,
                    reference_modified_by_new_plans=False,
                    actual_target_authority="original_float32_codec_plus_bounded_HG29_synergy",
                    subsequent_nn_history="actual_composite_action",
                    task_exam_modified=False,
                    neural_optimizer_updates=0,
                    hardware_authorized=False,
                )
                if hash_json(additional) != hash_json(expected_report):
                    raise ValueError("target authority or next-history search declaration differs")
            elif aim:
                if len(windows) != 57 or event["joint_task_response"] is not None:
                    raise ValueError("exact20 additional desired command forecasts required")
                expected_report = dict(
                    schema="soccer.bounded_neural_aim_command_search.v1",
                    candidate_count=20,
                    task_scoring_goal_world_point=[8.0, 0.0, 0.11],
                    command_world_points=[list(c.world_point) for c in aim_grid],
                    measured_ball_modified=False,
                    past_target_history_modified=False,
                    task_exam_modified=False,
                    reference_arrays_modified_by_new_commands=False,
                    neural_optimizer_updates=0,
                    hardware_authorized=False,
                )
                if hash_json(event["additional_candidate_search"]) != hash_json(expected_report):
                    raise ValueError("aim commands changed task, measurements, or declared search")
                for j, command in enumerate(aim_grid, 37):
                    if (
                        windows[j]["name"] != f"aim-goal-{j - 37:02d}"
                        or windows[j]["coefficients"] is not None
                        or windows[j]["aim_world_point"] != list(command.world_point)
                    ):
                        raise ValueError("named bounded aim command differs")
                if any(w["aim_world_point"] is not None for w in windows[:37]):
                    raise ValueError(
                        "original frozen candidates cannot contain hidden aim commands"
                    )
            elif sequential:
                verify_sequential_response(windows, forecasts, event["joint_task_response"])
            else:
                if len(windows) not in (95, 96):
                    raise ValueError(
                        "complete58 full-joint probes plus at most one proposal required"
                    )
                probes = []
                for axis in range(29):
                    for k, sign in enumerate((-1, 1)):
                        j = 37 + axis * 2 + k
                        coeff = np.zeros((2, 29))
                        coeff[0, axis] = sign * 0.03
                        if (
                            windows[j]["name"]
                            != f"joint-fd-{axis:02d}-{'minus' if sign < 0 else 'plus'}"
                            or not np.array_equal(coeff, np.array(windows[j]["coefficients"]))
                        ):
                            raise ValueError("named signed joint-reference probe differs")
                        probes.append(
                            (axis, sign, response_features(forecasts[j], windows[j]["outcome"]))
                        )
                proposal, fit = propose_joint_task_response(
                    response_features(forecasts[0], windows[0]["outcome"]), probes
                )
                if hash_json(fit) != hash_json(event["joint_task_response"]):
                    raise ValueError("actual full NN response fit/constraints differ")
                if proposal is None:
                    if len(windows) != 95:
                        raise ValueError(
                            "abstained response must not execute a fabricated proposal"
                        )
                elif (
                    len(windows) != 96
                    or windows[95]["name"] != "joint-task-response"
                    or not np.array_equal(proposal, np.array(windows[95]["coefficients"]))
                ):
                    raise ValueError("one exact bounded full NN response proposal required")

            def rank(j, candidates=windows, force_peaks=peaks):
                w = candidates[j]["outcome"]
                return (
                    int(w["window_pass"]),
                    int(w["predicted_body_healthy"]),
                    int(w["predicted_nonfoot_clean"]),
                    int(w["quality_observed"]),
                    w["reward"],
                    -force_peaks[j],
                )

            eligible = [j for j, w in enumerate(windows) if j and w["outcome"]["window_pass"]]
            chosen = max(eligible, key=rank) if eligible else 0
            expected_basis = (
                "parent"
                if not chosen
                else "executed-target"
                if executed and chosen >= 37
                else "aim-command"
                if aim and chosen >= 37
                else "joint-reference"
                if chosen >= 37
                else "backbone"
            )
            if (
                event["selected"] != windows[chosen]["name"]
                or event["selected_basis"] != expected_basis
                or row["changed_controls"] != int(chosen != 0)
            ):
                raise ValueError("physically eligible selection differs")
            if (
                chosen
                and not (aim and chosen >= 37)
                and event["installation"]["coefficients"] != windows[chosen]["coefficients"]
            ):
                raise ValueError("installed reference coefficients differ")
            if aim:
                expected_command = (
                    aim_grid[chosen - 37].target()
                    if chosen >= 37
                    else np.array([8.0, 0.0, 0.11], np.float32)
                )
                if (
                    row["task_scoring_goal_world_point"] != [8.0, 0.0, 0.11]
                    or row["desired_command_world_point"] != expected_command.tolist()
                ):
                    raise ValueError(
                        "task goal and actually selected desired command were conflated"
                    )
                if chosen >= 37 and (
                    event["installation"]["command_world_point"] != expected_command.tolist()
                    or any(
                        event["installation"][k]
                        for k in (
                            "past_history_modified",
                            "measured_ball_modified",
                            "physical_state_modified",
                            "task_scoring_goal_modified",
                        )
                    )
                ):
                    raise ValueError("installed aim command modified measurements/history/task")
            if executed:
                if event["original_actual_target_authority_unchanged"] != (chosen < 37):
                    raise ValueError("actual target authority was mislabeled as unchanged NN")
                if chosen >= 37:
                    installation = event["installation"]
                    if installation["start_control"] != row["entry_control"] or any(
                        installation[k]
                        for k in (
                            "reference_modified",
                            "neural_weights_modified",
                            "hardware_authorized",
                        )
                    ):
                        raise ValueError("selected target plan changed reference or clock")
                    coeff = windows[chosen]["coefficients"]
                    plan = ExecutedTargetSynergy(
                        row["entry_control"],
                        tuple(tuple(float(x) for x in r) for r in context["basis"]),
                        (tuple(coeff[0]), tuple(coeff[1])),
                    )
                    verify_target_edits(
                        row["actual_target_edits"],
                        arrays["joint_targets"][row["entry_control"] :],
                        start=row["entry_control"],
                        plan=plan,
                        codec=codec,
                        limits=context["model"].jnt_range[lo.joints],
                    )
                elif row["actual_target_edits"]:
                    raise ValueError("passing old branch must preserve complete old control path")
            if chosen >= 37 and not aim and not executed:
                installation = event["installation"]
                if installation["hg_to_isaac"] != [int(j) for j in context["types"][3]] or any(
                    b["start_frame"] != row["admitted_reference_phase"]
                    for b in installation["blocks"]
                ):
                    raise ValueError(
                        "installed full-joint permutation or measured reference phase differs"
                    )
            frame = row["entry_control"]
            future_exact(forecasts[0], old["B"][1], frame, forecasts[0]["controls"])
            future_exact(forecasts[chosen], arrays, frame, forecasts[chosen]["controls"])
            if row["forecasts"] != dict(
                forecast_calls=len(windows),
                attempted_native_steps=charged_arm,
                integrated_native_steps=integrated_arm,
                future_nn_calls=integrated_arm // 10,
                maximum_native_steps=(37 if arm == "B" else candidate_cap) * 1280,
            ):
                raise ValueError("all forecast NN/native budget counters differ")
            actual[arm], outcomes[arm], health[arm] = arrays, outcome, healthy
            parent_forecasts[arm] = forecasts[0]
            native += len(arrays["qpos"])
            worlds += 1
            charged += charged_arm
            integrated += integrated_arm
        if not same(
            {k: v for k, v in parent_forecasts["B"].items() if isinstance(v, np.ndarray)},
            {k: v for k, v in parent_forecasts["C"].items() if isinstance(v, np.ndarray)},
        ):
            raise ValueError("same-state parent forecasts differ")
        if i in (0, 30) and not same(actual["B"], actual["C"]):
            raise ValueError("old complete skill retention failed")
        if not outcomes["B"]["task_pass"] and outcomes["C"]["task_pass"] and health["C"]:
            rescues.append(i)
        if any(outcomes["B"][k] and not outcomes["C"][k] for k in ("task_pass", "safe")) or (
            health["B"] and not health["C"]
        ):
            regressions.append(i)
        results.append(
            dict(course_index=i, B=outcomes["B"], C=outcomes["C"], C_healthy=health["C"])
        )
    if (
        native > 43200
        or native != report["actual_native_steps"]
        or worlds != 12
        or charged != report["forecast_requested_native_steps"]
        or integrated != report["forecast_integrated_native_steps"]
        or charged > forecast_cap
        or rescues != report["rescues"]
        or regressions != report["regressions"]
    ):
        raise ValueError("whole pilot physical accounting differs")
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="FULL_NN_EXECUTED_TARGET_RAW_AND_PHYSICAL_EXAM_VERIFIED"
            if executed
            else "FULL_NN_AIM_COMMAND_RAW_AND_PHYSICAL_EXAM_VERIFIED"
            if aim
            else "FULL_NN_JOINT_TASK_RESPONSE_RAW_AND_PHYSICAL_EXAM_VERIFIED",
            aim_command_search=aim,
            executed_target_search=executed,
            sequential_restoration=sequential,
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            rescues=rescues,
            regressions=regressions,
            actual_worlds=worlds,
            actual_native_steps=native,
            forecast_requested_native_steps=charged,
            forecast_integrated_native_steps=integrated,
            qualified_for_expansion=len(rescues) >= 2
            and not regressions
            and all(r["C_healthy"] for r in results),
            new_physical_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
