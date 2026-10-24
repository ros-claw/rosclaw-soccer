"""Reconstruct archived pre-solver foot/ball geometry after whole-own audit.

No new stepping, policy inference, model fitting, force estimate or teacher
qualification. The current two-G1 capsule world is not the old sphere-foot world.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source-root", "own-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("source-hash", "own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    sp, op = a.source_root / "complete.json", a.own_root / "complete.json"
    if (
        a.output.exists()
        or any(r.is_symlink() or (r / "failed.json").exists() for r in (a.source_root, a.own_root))
        or any(p.is_symlink() for p in (sp, op))
        or [digest(sp), digest(op)] != [a.source_hash, a.own_hash]
    ):
        raise ValueError("exact completed source/whole-own evidence and unused output required")
    s, o = [json.loads(p.read_text()) for p in (sp, op)]
    if (
        s["status"] != "JOINT_DELIVERY_TEACHER38_COLLECTED_PENDING_ALL_OWN"
        or o["status"] != "JOINT_DELIVERY_TEACHER38_ALL_WHOLE_OWN_NATIVE_MATCH"
        or o["source_hash"] != a.source_hash
        or o["config"] != s["config"]
        or o["independent_whole_world_replay"] is not True
        or o["records"] != [{k: v for k, v in r.items() if k != "artifacts"} for r in s["records"]]
        or len(s["records"]) != 38
        or o["counts"] != s["counts"]
        or any(
            r[k] is not False
            for r in (s, o)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("all38 independently reproduced same-model physical histories required")
    pins = dict(o["input_file_hashes"])
    pins.update({str(sp): a.source_hash, str(op): a.own_hash})
    from scripts.rsi_joint_delivery_pilot_common import admit

    old_pins, _, _, _, inherited_root, baseline = admit(
        Path(s["expert_own_root"]), s["expert_own_hash"], s["config"]
    )
    for key, value in old_pins.items():
        if key in pins and pins[key] != value:
            raise ValueError("conflicting kinematic witness lineage")
        pins[key] = value
    histories = [("inherited", inherited_root, baseline)] + [
        (f"selected_or_validation_{i}", a.source_root, s["records"][i]) for i in (32, 34, 36)
    ]
    for _, root, row in histories:
        for name in ("initial", "physical"):
            art = row["artifacts"][name]
            path = root / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (root / f"trial{row['index']}").resolve()
                or digest(path) != art["hash"]
            ):
                raise ValueError("exact confined archived native states required")
            pins[str(path)] = art["hash"]
    xml = inherited_root / "scene.xml"
    if xml.is_symlink() or pins.get(str(xml)) != digest(xml):
        raise ValueError("exact qualified two-G1 standard scene required")
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    records = []
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.physics.contact_kinematic_reconstruction import (
            contact_kinematics,
            native_pre_step_state,
        )
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

        model = mujoco.MjModel.from_xml_string(xml.read_text())
        if (
            mujoco.__version__ != "3.5.0"
            or (model.nq, model.nv, model.nu) != (79, 76, 58)
            or model.opt.timestep != 0.002
            or model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        ):
            raise ValueError("original callback-free non-RK4 shared2ms clock required")
        layout = NativeVelocityLayout(
            model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
        )
        feet = tuple(
            int(model.body("a/" + side + "_ankle_roll_link").id) for side in ("left", "right")
        )
        for label, root, row in histories:
            with np.load(root / row["artifacts"]["initial"]["path"], allow_pickle=False) as z:
                qi, vi = z["qpos"].copy(), z["qvel"].copy()
            with np.load(root / row["artifacts"]["physical"]["path"], allow_pickle=False) as z:
                q, v = (
                    z["qpos"].reshape(-1, model.nq).copy(),
                    z["qvel"].reshape(-1, model.nv).copy(),
                )
                f = z["ball_forces"].reshape(-1, 2, 3).copy()
                target = z["target_a"].copy()
            strength = f[:3000, 0, :2].max(axis=1)
            touches = np.flatnonzero(strength > 10)
            if not len(touches):
                raise ValueError(
                    "selected diagnostic history must have actual recorded A-foot strike"
                )
            first = int(touches[0])
            if first != row["outcome"]["incoming"]["A_first_foot_strike_substep"]:
                raise ValueError("archived actual force clock and qualified outcome disagree")
            steps = [int(k) for k in touches if k < first + 100]
            events = []
            goal = np.array(row["course"]["target"], dtype=np.float64)
            for k in steps:
                qb, vb = native_pre_step_state(qi, vi, q, v, k)
                contacts = contact_kinematics(
                    model, qb, vb, ball_body=layout.ball, contact_bodies=feet, goal=goal
                )
                if not contacts:
                    raise ValueError(
                        "recorded foot force has no pre-step reconstructed foot geometry"
                    )
                events.append(
                    dict(
                        native_substep=k,
                        solver_time_s=k * 0.002,
                        recorded_max_A_foot_force_N=float(strength[k]),
                        geometric_contacts=contacts,
                    )
                )
            result = dict(
                label=label,
                original_index=row["index"],
                course=row["course"],
                outcome=row["outcome"],
                first_A_foot_substep=first,
                diagnostic_window_end_exclusive=min(first + 100, 3000),
                thresholded_force_steps=len(steps),
                events=events,
                ball_velocity_after_first_step_world=v[first, layout.bv : layout.bv + 3].tolist(),
                ball_velocity_after_contact_window_world=v[
                    min(first + 99, len(v) - 1), layout.bv : layout.bv + 3
                ].tolist(),
                A_joint_target_at_control80=target[80].tolist(),
                full_recorded_foot_impulse_claim=False,
                reconstructed_force_used=False,
            )
            records.append(result)
            write_once(a.output / (label + ".json"), result)
            print(
                json.dumps(
                    dict(
                        label=label,
                        first=first,
                        force_steps=len(steps),
                        reconstructed_pairs=sum(len(e["geometric_contacts"]) for e in events),
                    )
                ),
                flush=True,
            )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="WHOLE_OWN_ARCHIVED_CONTACT_KINEMATICS_RECONSTRUCTED_NO_NEW_PHYSICS",
                source_head=a.expected_head,
                source_hash=a.source_hash,
                own_hash=a.own_hash,
                input_file_hashes=pins,
                scene_hash=digest(xml),
                records=records,
                kinematic_clock="PRE_STEP_INITIAL_OR_PREVIOUS_POST_STEP_STATE",
                geometric_contact_not_individual_measured_force=True,
                physical_steps_executed=0,
                neural_inferences=0,
                optimizer_updates=0,
                teacher_qualified=False,
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
                completed_histories=len(records),
                physical_steps_executed=0,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
