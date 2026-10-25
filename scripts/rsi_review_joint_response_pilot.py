"""Independent full native readback of old/new task-response teacher; no stepping."""

import argparse
import json
from pathlib import Path


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
    if (
        report["status"] != "FULL_NN_JOINT_TASK_RESPONSE_PILOT_COMPLETE"
        or digest(dp) != report["declaration_hash"]
        or dec["course_indices"] != [0, 2, 8, 22, 27, 30]
        or dec["requested_forecast_native_step_ceiling"] != 1021440
        or dec["actual_world_ceiling"] != 12
        or dec["actual_native_step_ceiling"] != 43200
        or [r["course_index"] for r in report["results"]] != dec["course_indices"]
    ):
        raise ValueError("fixed whole six-state paired physics pilot required")
    pins = dict(report["input_file_hashes"])
    pins[str(cp)], pins[str(dp)] = a.pilot_hash, report["declaration_hash"]
    source_pins(pins)
    check_pins(pins)
    from scripts.rsi_review_motion_timing_teacher import verify_synergy_search
    from scripts.rsi_teacher_physics_context import load_teacher_context

    context = load_teacher_context(pins, a.backbone_model)
    import numpy as np

    from rosclaw_soccer.rsi.joint_task_response import (
        propose_joint_task_response,
        response_features,
    )
    from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    lo = context["layout"]
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
                    )
                )
                peaks.append(float(pred["ball_forces"][:, 0, 2].max()))
                forecasts.append(pred)
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
                "parent" if not chosen else "joint-reference" if chosen >= 37 else "backbone"
            )
            if (
                event["selected"] != windows[chosen]["name"]
                or event["selected_basis"] != expected_basis
                or row["changed_controls"] != int(chosen != 0)
            ):
                raise ValueError("physically eligible selection differs")
            if chosen and event["installation"]["coefficients"] != windows[chosen]["coefficients"]:
                raise ValueError("installed reference coefficients differ")
            if chosen >= 37:
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
                maximum_native_steps=(37 if arm == "B" else 96) * 1280,
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
        or charged > 1021440
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
            status="FULL_NN_JOINT_TASK_RESPONSE_RAW_AND_PHYSICAL_EXAM_VERIFIED",
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
