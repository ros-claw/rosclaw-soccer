"""Independent research teacher audit and explicit collision counterfactuals.

No inference or learning. All cases (not selected highlights) get a strict native
torque replay plus no-robot-ball and no-foot-ball fixed-torque counterfactuals.
Ghosts continue recorded torques even if unstable; they are NOT policy outcomes.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def collision_counterfactual(model, *, mode, ball, feet, robot):
    if mode not in ("no-robot-ball", "no-foot-ball") or model.npair:
        raise ValueError("explicit contact-bit-only probe and no explicit pairs required")
    for g in range(model.ngeom):
        t, a = int(model.geom_contype[g]), int(model.geom_conaffinity[g])
        if (t, a) == (0, 0):
            continue
        if (t, a) != (1, 1):
            raise ValueError("unexpected source collision mask; do not approximate")
        body = int(model.geom_bodyid[g])
        if body == 0:
            t, a = 1, 14
        elif body == ball:
            t, a = 2, 1 if mode == "no-robot-ball" else 9
        elif body in feet:
            t, a = 4, 13
        elif body in robot:
            t, a = 8, 13 if mode == "no-robot-ball" else 15
        else:
            raise ValueError("undeclared colliding body in counterfactual")
        model.geom_contype[g], model.geom_conaffinity[g] = t, a


def main():
    import mujoco
    import onnx

    from rosclaw_soccer.rsi.native_contact_window import native_contact_window
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_teacher_policy import strict_parameters
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import subprocess

    receipt = a.root / "complete.json"
    if (
        a.output.exists()
        or digest(receipt) != a.hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output, exact completed probe and frozen audit source required")
    r = json.loads(receipt.read_text())
    if mujoco.get_mjcb_control() is not None:
        raise ValueError("no external control callbacks allowed")
    if (
        r["status"] != "PAID_RESEARCH_NATIVE_TEACHER_PROBE_NOT_PROMOTED"
        or r["strict_replays_exact"] is not True
        or r["whole_repeats_exact"] is not True
    ):
        raise ValueError("actual completed research-only native teacher probe required")
    dp = a.root / "declaration.json"
    d = json.loads(dp.read_text())
    pins = {**r["input_file_hashes"], str(receipt): a.hash, str(dp): digest(dp)}
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            counterfactuals=["no-robot-ball", "no-foot-ball"],
            strict_original_replay=True,
            evaluate_all_rep0_cases=True,
            ghosts_fixed_recorded_torque_no_policy_no_body_stop=True,
            ghost_horizon="factual foot10N+600 substeps inclusive if observed; else source end",
            maximum_native_steps=r["live_native_steps"] // 2 * 3,
            new_neural_rows=0,
            optimizer_updates=0,
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )

    def load(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    model_files = [Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx")]
    if len(model_files) != 1:
        raise ValueError("one actual source teacher required")
    graph = onnx.load(model_files[0], load_external_data=False)
    names, params = strict_parameters({x.key: x.value for x in graph.metadata_props})
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

    order = np.array([names.index(name) for name in G1_DDS_JOINT_NAMES])
    offset, scale, kp, kd = [
        params[k][order]
        for k in ("default_joint_pos", "action_scale", "joint_stiffness", "joint_damping")
    ]
    # Named published effort limits, not guessed from shape/zero ctrlrange.
    effort = np.array(
        [
            139
            if ("hip_roll" in name or "knee" in name)
            else 88
            if ("hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint")
            else 50
            if ("ankle" in name or name in ("waist_roll_joint", "waist_pitch_joint"))
            else 5
            if ("wrist_pitch" in name or "wrist_yaw" in name)
            else 25
            for name in G1_DDS_JOINT_NAMES
        ]
    )
    expected = {
        (profile[0], case, repeat)
        for profile in d["profiles"]
        for case in range(len(d["cases"]))
        for repeat in (0, 1)
    }
    if {(x["profile"], x["case"], x["repeat"]) for x in r["results"]} != expected or len(
        r["results"]
    ) != len(expected):
        raise ValueError("complete declared balanced case/repeat matrix required")
    native = audited_nn = original_steps = 0
    results, first = [], {}
    try:
        for row in r["results"]:
            profile, case, repeat = row["profile"], row["case"], row["repeat"]
            relative = f"{profile}/case{case}-repeat{repeat}.npz"
            path = a.root / relative
            if digest(path) != r["raw_hashes"][relative]:
                raise ValueError("raw original teacher trajectory changed")
            pins[str(path)] = r["raw_hashes"][relative]
            z = load(path)
            key = (profile, case)
            if repeat == 0:
                first[key] = z
            else:
                old = first.pop(key)
                if set(old) != set(z) or any(not np.array_equal(old[k], z[k]) for k in z):
                    raise ValueError("whole independent LSTM/physical repeat differs")
            controls, nn = len(z["target"]), len(z["NN_action"])
            if controls != row["controls"] or nn != row["neural_rows"]:
                raise ValueError("actual control/inference counts differ")
            original_steps += controls * 10
            audited_nn += nn
            if (
                not np.array_equal(z["neural_control"], np.arange(controls) < nn)
                or z["NN_h_before"][0].any()
                or z["NN_c_before"][0].any()
                or not np.array_equal(z["NN_h_before"][1:], z["NN_h_after"][:-1])
                or not np.array_equal(z["NN_c_before"][1:], z["NN_c_after"][:-1])
                or z["NN_observation"][0, 125:154].any()
                or not np.array_equal(z["NN_observation"][1:, 125:154], z["NN_action"][:-1])
            ):
                raise ValueError(
                    "temporal policy reset, action history or inference-clock mismatch"
                )
            target = offset + z["NN_action"][:, order] * scale
            if not np.array_equal(target, z["target"][:nn]):
                raise ValueError("named action scale/offset/HG permutation differs")
            if controls > nn and not np.array_equal(
                z["target"][nn:], np.broadcast_to(target[-1], z["target"][nn:].shape)
            ):
                raise ValueError("held-target observation tail added new neural commands")
            xml = a.root / profile / "declared-scene.xml"
            if str(xml) not in pins:
                raise ValueError("exact compiled scene recipe missing")
            m = mujoco.MjModel.from_xml_path(str(xml))
            layout = NativeVelocityLayout(m, robot_prefix="", ball_body_name="soccer_ball")
            feet = [m.body(s + "_ankle_roll_link").id for s in ("left", "right")]
            robot = set(range(1, m.nbody)) - {layout.ball, m.body("soccer_goal_marker").id}
            before_q = z["native_qpos_before"][:, layout.qi]
            before_v = z["native_qvel_before"][:, layout.vi]
            q = np.concatenate((before_q[:, None], z["qpos"][:, :-1, 7:36]), axis=1)
            v = np.concatenate((before_v[:, None], z["qvel"][:, :-1, 6:35]), axis=1)
            torque = np.clip(kp * (z["target"][:, None] - q) - kd * v, -effort, effort)
            if not np.array_equal(torque, z["torque"]):
                raise ValueError("all2ms actual bounded PD torques differ")
            q, v, f = (
                z["qpos"].reshape(-1, 43),
                z["qvel"].reshape(-1, 41),
                z["forces"].reshape(-1, 3),
            )
            post = native_contact_window(
                force=f[:, :2],
                canonical_qpos=q,
                canonical_qvel=v,
                goal=np.array(row["goal_world"]),
                intended_foot=0 if "left" in row["source"] else 1,
            )
            bad = (
                (z["qpos"][:, :, 2] < 0.65)
                | (1 - 2 * (z["qpos"][:, :, 4] ** 2 + z["qpos"][:, :, 5] ** 2) < 0.9)
            ).any(axis=1)
            if (
                post != row["postcontact"]
                or bool((f[:, 2] > 1).any()) != row["forbidden"]
                or not np.array_equal(bad, z["body_failed"])
                or (bad.any() and not np.array_equal(np.flatnonzero(bad), [controls - 1]))
            ):
                raise ValueError("raw ball/body outcome or terminal-stop differs")
            if repeat:
                continue
            attribution, ghosts = {}, {}
            for mode in ("original", "no-robot-ball", "no-foot-ball"):
                model = mujoco.MjModel.from_xml_path(str(xml))
                if mode != "original":
                    collision_counterfactual(
                        model, mode=mode, ball=layout.ball, feet=feet, robot=robot
                    )
                data = mujoco.MjData(model)
                for field in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
                    getattr(data, field)[:] = z["initial_" + field]
                data.time = float(z["initial_time"])
                mujoco.mj_forward(model, data)
                data.qacc_warmstart[:] = z["initial_qacc_warmstart"]
                qq, vv = [], []
                warning = tuple(int(x.number) for x in data.warning)
                maximum_steps = controls * 10
                if mode != "original" and post["post_contact_quality_observed"]:
                    maximum_steps = int(post["first_foot_contact_substep"]) + 601
                for control in range(controls):
                    if control * 10 >= maximum_steps:
                        break
                    mujoco.mj_forward(model, data)
                    for sub in range(10):
                        if control * 10 + sub >= maximum_steps:
                            break
                        data.ctrl[layout.ai] = z["torque"][control, sub]
                        mujoco.mj_step(model, data)
                        native += 1
                        if tuple(int(x.number) for x in data.warning) != warning:
                            raise ValueError("native numerical warning/reset in counterfactual")
                        cq, cv = layout.canonical(data)
                        qq.append(cq)
                        vv.append(cv)
                        ff = np.zeros(3)
                        for ci in range(data.ncon):
                            contact = data.contact[ci]
                            x, y = (int(model.geom_bodyid[g]) for g in contact.geom)
                            other = y if x == layout.ball else x if y == layout.ball else -1
                            if other not in robot:
                                continue
                            wrench = np.zeros(6)
                            mujoco.mj_contactForce(model, data, ci, wrench)
                            force = float(np.linalg.norm(wrench[:3]))
                            role = feet.index(other) if other in feet else 2
                            ff[role] = max(ff[role], force)
                            if mode == "original" and force > 1:
                                name = model.body(other).name
                                record = attribution.setdefault(
                                    name,
                                    dict(
                                        first_substep=control * 10 + sub,
                                        peak_force_N=0.0,
                                        contact_impulse_magnitude_Ns=0.0,
                                        samples=0,
                                    ),
                                )
                                record["peak_force_N"] = max(record["peak_force_N"], force)
                                record["contact_impulse_magnitude_Ns"] += force * 0.002
                                record["samples"] += 1
                        if mode == "original" and (
                            not np.array_equal(cq, z["qpos"][control, sub])
                            or not np.array_equal(cv, z["qvel"][control, sub])
                            or not np.array_equal(ff, z["forces"][control, sub])
                        ):
                            raise ValueError("independent original strict replay differs")
                        if (mode == "no-robot-ball" and ff.any()) or (
                            mode == "no-foot-ball" and ff[:2].any()
                        ):
                            raise ValueError("counterfactual collision masking failed")
                qq, vv = np.array(qq), np.array(vv)
                first_hit = post["first_foot_contact_substep"]
                progress = None
                if first_hit is not None and first_hit + 600 < len(qq):
                    direction = np.array(row["goal_world"])[:2] - q[first_hit, 36:38]
                    direction /= np.linalg.norm(direction)
                    progress = float(
                        (qq[first_hit + 600, 36:38] - qq[first_hit, 36:38]) @ direction
                    )
                if mode != "original":
                    file = a.output / f"{profile}-case{case}-{mode}.npz"
                    np.savez_compressed(file, qpos=qq, qvel=vv)
                    ghosts[mode] = dict(
                        raw_hash=digest(file),
                        progress_in_factual_window_m=progress,
                        minimum_pelvis_height_m=float(qq[:, 2].min()),
                        no_NN_no_state_overwrite=True,
                    )
            results.append(
                dict(
                    profile=profile,
                    case=case,
                    attribution=attribution,
                    ghosts=ghosts,
                    factual_postcontact=post,
                )
            )
            print(json.dumps(results[-1]), flush=True)
        if first or original_steps != r["live_native_steps"] or audited_nn != r["neural_rows"]:
            raise ValueError("complete raw original work accounting differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_TEACHER_CAUSAL_RESEARCH_AUDIT_NOT_PROMOTED",
                input_file_hashes=pins,
                results=results,
                native_steps=native,
                new_neural_rows=0,
                optimizer_updates=0,
                original_all_substeps_replayed_exact=True,
                raw_RNN_PD_masks_outcomes_exact=True,
                counterfactual_ghosts_not_policy_outcomes=True,
                commercial_publicity_authorized=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                native_steps=native,
                new_neural_rows=0,
                optimizer_updates=0,
                commercial_publicity_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
