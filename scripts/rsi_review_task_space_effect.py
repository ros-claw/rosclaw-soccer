"""Read actual B/C contact effects, including first same-state causal divergence."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--pilot-hash", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    cp = a.pilot / "complete.json"
    if a.output.exists() or digest(cp) != a.pilot_hash:
        raise ValueError("unused readout and completed pinned pilot required")
    report = json.loads(cp.read_text())
    if report["status"] != "BOUNDED_SURFACE_CONTACT_PILOT_COMPLETE":
        raise ValueError("closed complete B/C physical study required")
    pins = dict(report["input_file_hashes"])
    pins[str(cp)] = a.pilot_hash
    records = []
    for pair in report["results"]:
        rows = {}
        for arm in ("B", "C"):
            reference = pair["arms"][arm]
            path = a.pilot / reference["record"]
            if (
                not path.resolve().is_relative_to(a.pilot.resolve())
                or digest(path) != reference["hash"]
            ):
                raise ValueError("physical record identity differs")
            row = json.loads(path.read_text())
            artifact = path.parent / row["artifact"]["path"]
            pins[str(path)] = reference["hash"]
            pins[str(artifact)] = row["artifact"]["hash"]
            rows[arm] = row, artifact
        records.append((pair["course_index"], rows))
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique original physical model required")

    import collections

    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    results, status_counts = [], collections.Counter()
    for index, rows in records:
        data, metadata = {}, {}
        for arm, (row, artifact) in rows.items():
            with np.load(artifact, allow_pickle=False) as raw:
                data[arm] = {k: raw[k].copy() for k in raw.files}
            if hash_json(
                relabel_arrays(
                    data[arm],
                    controls=row["controls"],
                    ball_address=layout.bq,
                    root_address=layout.rq,
                )
            ) != hash_json(row["outcome"]):
                raise ValueError("native physical label readback differs")
            metadata[arm] = dict(
                task_pass=row["outcome"]["task_pass"],
                safe=row["outcome"]["safe"],
                healthy=row["healthy_full_horizon"],
                first_foot=row["outcome"]["first_foot_contact_frame"],
                nonfoot_peak_n=float(data[arm]["ball_forces"][:, 0, 2].max()),
            )
        c = rows["C"][0]
        authority_states = []
        first = metadata["B"]["first_foot"]
        entry = rows["B"][0]["entry_control"]
        if first is not None and entry is not None:
            for frame in range(max(1, entry), first + 1):
                state = mujoco.MjData(model)
                state.qpos[:] = data["B"]["qpos"][frame * 10 - 1]
                state.qvel[:] = data["B"]["qvel"][frame * 10 - 1]
                surfaces = measure_ball_surfaces(model, state, layout)["surfaces"]
                foot_gaps = [
                    s["gap_m"] for n, s in surfaces.items() if n.endswith("foot") and s["available"]
                ]
                if not foot_gaps or min(foot_gaps) > 0.18:
                    continue
                nominal = data["B"]["joint_targets"][frame, :12]
                limits = model.jnt_range[layout.joints[:12]]
                lower = np.maximum(limits[:, 0] - nominal, -0.03)
                upper = np.minimum(limits[:, 1] - nominal, 0.03)
                authority_states.append(
                    dict(
                        frame=frame,
                        no_prior_increment_admissible_joint_count=int((lower <= upper).sum()),
                        inadmissible_joint_names=[
                            G1_DDS_JOINT_NAMES[j] for j in np.flatnonzero(lower > upper)
                        ],
                    )
                )
        events = c.get("controller_events", [])
        changes = [e for e in events if e["changed"]]
        if len(changes) != c.get("changed_controls", 0):
            raise ValueError("actual controller event count differs")
        status_counts.update(s for e in events for s in e.get("solver_statuses", []))
        causal = None
        if not changes:
            if any(not np.array_equal(data["B"][k], v) for k, v in data["C"].items()):
                raise ValueError("no-change C differs from original native execution")
        else:
            event = changes[0]
            frame = event["frame"]
            for key in ("qpos", "qvel", "torque", "ball_forces"):
                if not np.array_equal(data["B"][key][: frame * 10], data["C"][key][: frame * 10]):
                    raise ValueError(
                        "first target intervention lacks identical actual starting state"
                    )
            native = (frame + 1) * 10 - 1
            gaps = {}
            for arm in ("B", "C"):
                scratch = mujoco.MjData(model)
                scratch.qpos[:] = data[arm]["qpos"][native]
                scratch.qvel[:] = data[arm]["qvel"][native]
                surfaces = measure_ball_surfaces(model, scratch, layout)["surfaces"]
                gaps[arm] = {n: s.get("gap_m") for n, s in surfaces.items()}
            delta = data["C"]["joint_targets"][frame] - data["B"]["joint_targets"][frame]
            if not np.allclose(delta, event["requested_delta_rad"], atol=1e-12, rtol=0):
                raise ValueError("first intervention requested/actual target delta differs")
            causal = dict(
                first_changed_control=frame,
                actual_starting_state_identical=True,
                maximum_requested_target_delta_rad=float(abs(delta).max()),
                maximum_actual_joint_response_delta_rad=float(
                    abs(
                        data["C"]["qpos"][native, layout.qi] - data["B"]["qpos"][native, layout.qi]
                    ).max()
                ),
                maximum_actual_torque_delta_nm=float(
                    abs(
                        data["C"]["torque"][frame * 10 : (frame + 1) * 10]
                        - data["B"]["torque"][frame * 10 : (frame + 1) * 10]
                    ).max()
                ),
                measured_surface_gaps_after_first_actual_control=gaps,
                whole_episode_outcome_not_implied_by_one_step=True,
            )
        results.append(
            dict(
                course_index=index,
                arms=metadata,
                changed_controls=len(changes),
                first_same_state_causal_effect=causal,
                nominal_tracking_error_max_rad=max(
                    (e.get("nominal_tracking_error_rad", 0.0) for e in events), default=0.0
                ),
                original_precontact_zero_increment_authority=authority_states,
            )
        )
    if (
        sum(rows[arm][0]["native_steps"] for _, rows in records for arm in ("B", "C"))
        != report["actual_native_steps"]
    ):
        raise ValueError("actual native accounting differs")
    rescues = [
        r["course_index"]
        for r in results
        if not r["arms"]["B"]["task_pass"]
        and r["arms"]["C"]["task_pass"]
        and r["arms"]["C"]["healthy"]
    ]
    regressions = [
        r["course_index"]
        for r in results
        if any(
            r["arms"]["B"][k] and not r["arms"]["C"][k] for k in ("task_pass", "safe", "healthy")
        )
    ]
    if rescues != report["rescues"] or regressions != report["regressions"]:
        raise ValueError("paired retention/rescue readback differs")
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="TASK_SPACE_NATIVE_EFFECT_READBACK_COMPLETE",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            solver_status_counts=dict(status_counts),
            rescues=rescues,
            regressions=regressions,
            actual_changed_controls=sum(r["changed_controls"] for r in results),
            new_physical_steps=0,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
