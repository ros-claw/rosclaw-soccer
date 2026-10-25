"""Independent raw readback of finite reference timing; no new physics steps."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--pilot-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--shard-root", type=Path, nargs=4, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    cp = a.pilot / "complete.json"
    if a.output.exists() or digest(cp) != a.pilot_hash:
        raise ValueError("unused output and exact closed pilot required")
    report = json.loads(cp.read_text())
    if report["interface"] != "motion-timing":
        raise ValueError("explicit motion timing protocol required")
    pins = dict(report["input_file_hashes"])
    pins[str(cp)] = a.pilot_hash
    pins[str(a.pilot / "declaration.json")] = report["declaration_hash"]
    declaration = json.loads((a.pilot / "declaration.json").read_text())
    expected_indices = [b * 10 + j for b in range(4) for j in (0, 1, 2, 3)]
    if (
        declaration["course_indices"] != expected_indices
        or declaration["timing_intervals"] != 24
        or declaration["timing_peak_shift_frames"] != [-3.0, 3.0]
        or declaration["forecast_native_step_ceiling"] != 46080
        or [r["course_index"] for r in report["results"]] != expected_indices
    ):
        raise ValueError("frozen timing experiment differs")
    source_pins(pins)
    check_pins(pins)

    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    scene = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
    if len(scene) != 1:
        raise ValueError("one qualified body required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scene[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")

    def load_raw(folder, ref):
        path = folder / ref["path"]
        if not path.resolve().is_relative_to(a.pilot.resolve()) or digest(path) != ref["hash"]:
            raise ValueError("unconfined or changed raw artifact")
        pins[str(path)] = ref["hash"]
        with np.load(path, allow_pickle=False) as raw:
            return {k: raw[k].copy() for k in raw.files}

    results, rescues, regressions = [], [], []
    attempted = integrated = native = worlds = 0
    for result in report["results"]:
        pair, arrays = {}, {}
        for arm in ("B", "C"):
            ref = result["arms"][arm]
            path = a.pilot / ref["record"]
            if not path.resolve().is_relative_to(a.pilot.resolve()) or digest(path) != ref["hash"]:
                raise ValueError("changed actual record")
            pins[str(path)] = ref["hash"]
            row = json.loads(path.read_text())
            raw = load_raw(path.parent, row["artifact"])
            if hash_json(
                relabel_arrays(
                    raw, controls=row["controls"], ball_address=layout.bq, root_address=layout.rq
                )
            ) != hash_json(row["outcome"]):
                raise ValueError("actual football label differs")
            pelvis = raw["qpos"][:, layout.rq : layout.rq + 7]
            healthy = bool(
                (pelvis[:, 2] >= 0.65).all()
                and (1 - 2 * (pelvis[:, 4] ** 2 + pelvis[:, 5] ** 2) >= 0.9).all()
            )
            if healthy != row["healthy_full_horizon"]:
                raise ValueError("actual whole-horizon body label differs")
            native += len(raw["qpos"])
            worlds += 1
            pair[arm], arrays[arm] = row, raw
        b, c = pair["B"], pair["C"]
        old_record_path = (
            a.shard_root[result["course_index"] % 4]
            / f"course-{result['course_index']:02d}"
            / "onnx.json"
        )
        if pins.get(str(old_record_path)) != digest(old_record_path):
            raise ValueError("explicit pinned historical original record required")
        old_record = json.loads(old_record_path.read_text())
        old_path = old_record_path.parent / old_record["artifact"]["path"]
        if digest(old_path) != old_record["artifact"]["hash"]:
            raise ValueError("historical original trajectory changed")
        pins[str(old_path)] = old_record["artifact"]["hash"]
        with np.load(old_path, allow_pickle=False) as old:
            if set(old.files) != set(arrays["B"]) or any(
                not np.array_equal(old[k], value) for k, value in arrays["B"].items()
            ):
                raise ValueError("actual B differs from historical whole original trajectory")
        events = c.get("controller_events", [])
        items = c.get("candidate_forecasts", [])
        if len(events) > 1 or c.get("changed_controls", 0) not in (0, 1):
            raise ValueError("one reference installation contract differs")
        forecasts = []
        windows = []
        charged = advanced = 0
        for item in items:
            folder = a.pilot / result["arms"]["C"]["record"]
            raw = load_raw(folder.parent, item["artifact"])
            frame, n = item["donor_control_frame"], item["controls"]
            requested = min(96, 360 - frame)
            if (
                frame != c["entry_control"]
                or item["requested_controls"] != requested
                or not 1 <= n <= requested
            ):
                raise ValueError("forecast entry/horizon differs")
            raw["controls"] = n
            score = predicted_touch_window(
                raw,
                ball_address=layout.bq,
                root_address=layout.rq,
                start_control=frame,
                diagnostic_tail=True,
            )
            if n < requested and score["predicted_body_healthy"]:
                raise ValueError("unexplained early termination")
            charged += requested * 10
            advanced += n * 10
            forecasts.append((item["name"], raw, frame, n))
            windows.append(dict(name=item["name"], outcome=score))
        if items:
            if not events or hash_json(windows) != hash_json(events[0]["candidate_windows"]):
                raise ValueError("candidate scores differ")
            expected_names = (
                ["parent"]
                if windows[0]["outcome"]["window_pass"]
                else ["parent", "timing--3", "timing-+3"]
            )
            if [f[0] for f in forecasts] != expected_names:
                raise ValueError("finite candidate sequence differs")
            eligible = [i for i, w in enumerate(windows) if i and w["outcome"]["window_pass"]]
            chosen = max(eligible, key=lambda i: windows[i]["outcome"]["reward"]) if eligible else 0
            if events[0]["selected"] != forecasts[chosen][0] or c["changed_controls"] != int(
                chosen != 0
            ):
                raise ValueError("selected reference plan differs")
            for index, actual in ((0, arrays["B"]), (chosen, arrays["C"])):
                _, prediction, frame, n = forecasts[index]
                for k in ("qpos", "qvel", "torque", "ball_forces"):
                    if not np.array_equal(prediction[k], actual[k][frame * 10 : (frame + n) * 10]):
                        raise ValueError("predicted native future differs from actual")
                if not np.array_equal(
                    prediction["joint_targets"], actual["joint_targets"][frame : frame + n]
                ):
                    raise ValueError("predicted NN targets differ from actual")
            counts = c["forecasts"]
            if counts != dict(
                forecast_calls=len(items),
                attempted_native_steps=charged,
                integrated_native_steps=advanced,
                future_nn_calls=advanced // 10,
                maximum_native_steps=2880,
            ):
                raise ValueError("forecast NN/budget accounting differs")
        elif c.get("changed_controls", 0):
            raise ValueError("changed reference without forecast")
        if c.get("changed_controls", 0) == 0 and any(
            not np.array_equal(arrays["B"][k], arrays["C"][k]) for k in arrays["B"]
        ):
            raise ValueError("OFF branch differs from original")
        attempted += charged
        integrated += advanced
        index = result["course_index"]
        if (
            not b["outcome"]["task_pass"]
            and c["outcome"]["task_pass"]
            and c["healthy_full_horizon"]
        ):
            rescues.append(index)
        if (b["healthy_full_horizon"] and not c["healthy_full_horizon"]) or any(
            b["outcome"][k] and not c["outcome"][k] for k in ("task_pass", "safe")
        ):
            regressions.append(index)
        results.append(
            dict(
                course_index=index,
                candidate_windows=windows,
                B=b["outcome"],
                C=c["outcome"],
                C_healthy=c["healthy_full_horizon"],
            )
        )
    if (
        rescues != report["rescues"]
        or regressions != report["regressions"]
        or attempted != report["forecast_attempted_native_steps"]
        or integrated != report["forecast_native_steps"]
        or native != report["actual_native_steps"]
        or worlds != report["actual_worlds"]
    ):
        raise ValueError("whole physical experiment accounting differs")
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="MOTION_TIMING_TEACHER_RAW_FUTURES_AND_FULL_EXAM_VERIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            rescues=rescues,
            regressions=regressions,
            actual_worlds=worlds,
            forecast_attempted_native_steps=attempted,
            forecast_integrated_native_steps=integrated,
            qualified_for_expansion=len(rescues) >= 2 and not regressions,
            new_physical_steps=0,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
