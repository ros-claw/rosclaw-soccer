"""Native readback of outcome-selected interventions and every candidate window."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--pilot-hash", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    cp = a.pilot / "complete.json"
    if a.output.exists() or digest(cp) != a.pilot_hash:
        raise ValueError("unused readback output and exact closed pilot required")
    report = json.loads(cp.read_text())
    pins = dict(report["input_file_hashes"])
    pins[str(cp)] = a.pilot_hash
    pairs = []
    for result in report["results"]:
        pair = {}
        for arm in ("B", "C"):
            ref = result["arms"][arm]
            path = a.pilot / ref["record"]
            if digest(path) != ref["hash"]:
                raise ValueError("actual row identity differs")
            row = json.loads(path.read_text())
            pins[str(path)] = ref["hash"]
            for artifact in [
                row["artifact"],
                *(r["artifact"] for r in row.get("candidate_forecasts", [])),
            ]:
                target = path.parent / artifact["path"]
                if not target.resolve().is_relative_to(a.pilot.resolve()):
                    raise ValueError("unconfined scientific artifact")
                pins[str(target)] = artifact["hash"]
            pair[arm] = row, path.parent
        pairs.append((result["course_index"], pair))
    source_pins(pins)
    check_pins(pins)

    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.outcome_forecast_kick_player import choose_window_rescue
    from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique qualified body scene required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    results, total_attempted, total_integrated = [], 0, 0
    for index, pair in pairs:
        arrays = {}
        for arm, (row, folder) in pair.items():
            with np.load(folder / row["artifact"]["path"], allow_pickle=False) as raw:
                arrays[arm] = {k: raw[k].copy() for k in raw.files}
            if hash_json(
                relabel_arrays(
                    arrays[arm],
                    controls=row["controls"],
                    ball_address=layout.bq,
                    root_address=layout.rq,
                )
            ) != hash_json(row["outcome"]):
                raise ValueError("actual native label differs")
        c, folder = pair["C"]
        b = pair["B"][0]
        events = c.get("controller_events", [])
        if len(events) > 1 or c.get("changed_controls", 0) != sum(e["changed"] for e in events):
            raise ValueError("one-decision/one-intervention contract differs")
        readback, candidates = [], []
        attempted = integrated = calls = future_nn = 0
        for item in c.get("candidate_forecasts", []):
            with np.load(folder / item["artifact"]["path"], allow_pickle=False) as raw:
                predicted = {k: raw[k].copy() for k in raw.files}
            frame, n = item["donor_control_frame"], item["controls"]
            predicted["controls"] = n
            expected = min(64, 300 - frame)
            if item["requested_controls"] != expected or not 1 <= n <= expected:
                raise ValueError("forecast time allocation differs")
            outcome = predicted_touch_window(
                predicted, ball_address=layout.bq, root_address=layout.rq, start_control=frame
            )
            if n < expected and outcome["predicted_body_healthy"]:
                raise ValueError("short forecast lacks physical termination")
            calls += 1
            attempted += expected * 10
            integrated += n * 10
            future_nn += n - 1
            readback.append(dict(name=item["name"], outcome=outcome))
            candidates.append(
                (item["name"], predicted["joint_targets"][0], None, predicted, outcome)
            )
            if item["name"] == "parent":
                for key in ("qpos", "qvel", "torque", "ball_forces"):
                    if not np.array_equal(
                        predicted[key], arrays["B"][key][frame * 10 : (frame + n) * 10]
                    ):
                        raise ValueError(
                            "parent counterfactual differs from actual original B future"
                        )
                if not np.array_equal(
                    predicted["joint_targets"], arrays["B"]["joint_targets"][frame : frame + n]
                ):
                    raise ValueError("parent future NN targets differ from actual B")
        if events:
            event = events[0]
            if hash_json(readback) != hash_json(event["candidate_windows"]):
                raise ValueError("candidate window math differs")
            chosen = choose_window_rescue(candidates)
            if candidates[chosen][0] != event["selected"] or (chosen != 0) != event["changed"]:
                raise ValueError("selected candidate not justified by observed-window rescue")
            frame = event["frame"]
            selected = candidates[chosen][3]
            n = len(selected["joint_targets"])
            for key in ("qpos", "qvel", "torque", "ball_forces"):
                if not np.array_equal(
                    arrays["B"][key][: frame * 10], arrays["C"][key][: frame * 10]
                ):
                    raise ValueError("decision lacks identical actual starting history")
                if not np.array_equal(
                    selected[key], arrays["C"][key][frame * 10 : (frame + n) * 10]
                ):
                    raise ValueError("selected future differs from actual C branch")
        if not c.get("changed_controls", 0) and any(
            not np.array_equal(arrays["B"][k], v) for k, v in arrays["C"].items()
        ):
            raise ValueError("unmodified C does not preserve original physical trajectory")
        if "forecasts" in c and c["forecasts"] != dict(
            forecast_calls=calls,
            attempted_native_steps=attempted,
            integrated_native_steps=integrated,
            future_nn_calls=future_nn,
            maximum_native_steps=8320,
        ):
            raise ValueError("forecast budget/NN call readback differs")
        if "forecasts" not in c and calls:
            raise ValueError("unadmitted controller nevertheless allocated forecasts")
        total_attempted += attempted
        total_integrated += integrated
        nonfoot = np.flatnonzero(arrays["B"]["ball_forces"][:, 0, 2] > 1.0)
        first = b["outcome"]["first_foot_contact_frame"]
        entry = b["entry_control"]
        foot_trigger = shin_trigger = None
        if first is not None and entry is not None:
            for frame in range(max(1, entry), min(first + 6, 240)):
                scratch = mujoco.MjData(model)
                scratch.qpos[:] = arrays["B"]["qpos"][frame * 10 - 1]
                scratch.qvel[:] = arrays["B"]["qvel"][frame * 10 - 1]
                surfaces = measure_ball_surfaces(model, scratch, layout)["surfaces"]
                if foot_trigger is None and any(
                    s["available"] and s["gap_m"] <= 0.06
                    for name, s in surfaces.items()
                    if name.endswith("foot")
                ):
                    foot_trigger = frame
                if shin_trigger is None and any(
                    s["available"] and s["gap_m"] <= 0.04
                    for name, s in surfaces.items()
                    if name.endswith("shin")
                ):
                    shin_trigger = frame
        results.append(
            dict(
                course_index=index,
                B=b["outcome"],
                C=c["outcome"],
                B_healthy=b["healthy_full_horizon"],
                C_healthy=c["healthy_full_horizon"],
                changed_controls=c.get("changed_controls", 0),
                candidate_windows=readback,
                first_original_nonfoot=None if not len(nonfoot) else int(nonfoot[0] // 10),
                original_foot_trigger=foot_trigger,
                original_shin_hazard_trigger=shin_trigger,
            )
        )
    if (
        total_attempted != report["forecast_attempted_native_steps"]
        or total_integrated != report["forecast_native_steps"]
    ):
        raise ValueError("whole pilot prediction accounting differs")
    rescues = [
        r["course_index"]
        for r in results
        if not r["B"]["task_pass"] and r["C"]["task_pass"] and r["C_healthy"]
    ]
    regressions = [
        r["course_index"]
        for r in results
        if (r["B_healthy"] and not r["C_healthy"])
        or any(r["B"][k] and not r["C"][k] for k in ("task_pass", "safe"))
    ]
    if rescues != report["rescues"] or regressions != report["regressions"]:
        raise ValueError("full physical rescue/retention labels differ")
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="NN_OUTCOME_TEACHER_NATIVE_AND_COUNTERFACTUAL_WINDOWS_VERIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            rescues=rescues,
            regressions=regressions,
            forecast_attempted_native_steps=total_attempted,
            forecast_integrated_native_steps=total_integrated,
            new_physical_steps=0,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
