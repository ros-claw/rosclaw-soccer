"""Independent raw readback of finite reference timing; no new physics steps."""

import argparse
import json
from pathlib import Path


def verify_synergy_search(windows, peaks, generations):
    """Regenerate finite CEM from raw-derived labels, not the controller helper."""
    import numpy as np

    def rank(i):
        w = windows[i]["outcome"]
        return (
            int(w["window_pass"]),
            int(w["predicted_body_healthy"]),
            int(w["predicted_nonfoot_clean"]),
            int(w["quality_observed"]),
            float(w["reward"]),
            -peaks[i],
        )

    if windows[0]["coefficients"] is not None:
        raise ValueError("parent must have no reference plan")
    if windows[0]["outcome"]["window_pass"]:
        if len(windows) != 1 or generations:
            raise ValueError("passing parent must not trigger CEM")
        return 0
    if len(windows) != 37 or len(generations) != 3:
        raise ValueError("exact finite CEM allocation required")
    rng = np.random.default_rng(7001)
    mean, std = np.zeros((2, 6)), np.full((2, 6), 0.025)
    for generation in range(3):
        noise = rng.normal(size=(6, 2, 6))
        population = np.clip(mean + np.concatenate((noise, -noise)) * std, -0.06, 0.06)
        offset = 1 + generation * 12
        for i in range(12):
            if windows[offset + i]["name"] != f"g{generation}-p{i:02d}" or not np.array_equal(
                np.array(windows[offset + i]["coefficients"]), population[i]
            ):
                raise ValueError("CEM candidate differs from seeded data-space search")
        elite_ids = sorted(range(12), key=lambda i: rank(offset + i), reverse=True)[:3]
        g = dict(
            generation=generation, elite_indices=elite_ids, mean=mean.tolist(), std=std.tolist()
        )
        if g != generations[generation]:
            raise ValueError("CEM elite/distribution differs from raw native outcomes")
        elite = population[elite_ids]
        mean = 0.5 * mean + 0.5 * elite.mean(axis=0)
        std = np.maximum(0.006, 0.5 * std + 0.5 * elite.std(axis=0))
    eligible = [i for i, w in enumerate(windows) if i and w["outcome"]["window_pass"]]
    return max(eligible, key=rank) if eligible else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--pilot-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--shard-root", type=Path, nargs=4, required=True)
    p.add_argument("--synergy-model", type=Path)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    cp = a.pilot / "complete.json"
    if a.output.exists() or digest(cp) != a.pilot_hash:
        raise ValueError("unused output and exact closed pilot required")
    report = json.loads(cp.read_text())
    if report["interface"] not in ("motion-timing", "motion-synergy"):
        raise ValueError("explicit motion reference protocol required")
    synergy = report["interface"] == "motion-synergy"
    pins = dict(report["input_file_hashes"])
    fitted = None
    if synergy:
        if a.synergy_model is None or pins.get(str(a.synergy_model)) != digest(a.synergy_model):
            raise ValueError("explicit pinned synergy model required")
        fitted = json.loads(a.synergy_model.read_text())
    pins[str(cp)] = a.pilot_hash
    pins[str(a.pilot / "declaration.json")] = report["declaration_hash"]
    declaration = json.loads((a.pilot / "declaration.json").read_text())
    expected_indices = [b * 10 + j for b in range(4) for j in (0, 1, 2, 3)]
    if (
        declaration["course_indices"] != expected_indices
        or (
            not synergy
            and (
                declaration["timing_intervals"] != 24
                or declaration["timing_peak_shift_frames"] != [-3.0, 3.0]
            )
        )
        or declaration["forecast_native_step_ceiling"] != (568320 if synergy else 46080)
        or [r["course_index"] for r in report["results"]] != expected_indices
    ):
        raise ValueError("frozen timing experiment differs")
    if synergy and any(
        declaration[k] != value
        for k, value in dict(
            synergy_cem_population=12,
            synergy_cem_generations=3,
            synergy_cem_elites=3,
            synergy_cem_seed=7001,
            synergy_cem_initial_std_rad=0.025,
            synergy_cem_minimum_std_rad=0.006,
            synergy_cem_update_smoothing=0.5,
            synergy_reference_trust_rad=0.06,
            synergy_reference_intervals=48,
            synergy_temporal_lobes=2,
        ).items()
    ):
        raise ValueError("frozen data-synergy search protocol differs")
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
    innovations = []
    training_indices = []
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
        if synergy and b["outcome"]["task_pass"] and b["healthy_full_horizon"]:
            innovations.append(
                np.diff(arrays["B"]["joint_targets"][b["entry_control"] : 300], axis=0)
            )
            training_indices.append(result["course_index"])
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
        peaks = []
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
            peaks.append(float(raw["ball_forces"][:, 0, 2].max()))
            if synergy:
                windows[-1]["coefficients"] = events[0]["candidate_windows"][len(windows) - 1][
                    "coefficients"
                ]
        if items:
            if not events or hash_json(windows) != hash_json(events[0]["candidate_windows"]):
                raise ValueError("candidate scores differ")
            expected_names = (
                ["parent"]
                if windows[0]["outcome"]["window_pass"]
                else ["parent"] + [f"g{g}-p{i:02d}" for g in range(3) for i in range(12)]
                if synergy
                else ["parent", "timing--3", "timing-+3"]
            )
            if [f[0] for f in forecasts] != expected_names:
                raise ValueError("finite candidate sequence differs")
            eligible = [i for i, w in enumerate(windows) if i and w["outcome"]["window_pass"]]
            chosen = max(eligible, key=lambda i: windows[i]["outcome"]["reward"]) if eligible else 0
            if synergy:
                chosen = verify_synergy_search(windows, peaks, events[0]["generations"])
                if (
                    chosen
                    and events[0]["installation"]["coefficients"] != windows[chosen]["coefficients"]
                ):
                    raise ValueError("installed coefficients differ from selected physical plan")
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
                maximum_native_steps=35520 if synergy else 2880,
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
    if synergy:
        x = np.concatenate(innovations).astype(np.float64)
        _, singular, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
        reconstructed = vt[:6].copy()
        for row in reconstructed:
            row *= 1 if row[np.argmax(abs(row))] > 0 else -1
            row /= abs(row).max()
        artifact = a.synergy_model.parent / fitted["artifact"]["path"]
        if pins.get(str(artifact)) != digest(artifact):
            raise ValueError("bound fitted model artifact differs")
        with np.load(artifact, allow_pickle=False) as raw:
            if not np.array_equal(reconstructed, raw["basis_hg"]):
                raise ValueError("learned subspace differs from actual successful B data")
        fit = fitted["fit"]
        if (
            len(x) != fit["samples"]
            or training_indices != [r["course_index"] for r in fitted["records"]]
            or not np.array_equal(singular, np.array(fit["singular_values"]))
            or float((singular[:6] ** 2).sum() / (singular**2).sum())
            != fit["explained_variance_fraction"]
        ):
            raise ValueError("data model provenance/variance differs")
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED"
                if synergy
                else "MOTION_TIMING_TEACHER_RAW_FUTURES_AND_FULL_EXAM_VERIFIED"
            ),
            source_head=a.expected_head,
            data_synergy_fit_independently_recomputed=synergy,
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
