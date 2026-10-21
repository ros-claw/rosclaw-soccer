"""Research-only full soccer LSTM native probe; NOT a publicity certificate.

External CC-BY-NC weights/source stay external. Fixed TRAIN8 ball cases and
mass/reference arms declared before integration. No optimization, result search,
reference padding, hidden resets or hardware. Native recorded-torque replays.
"""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--reference-head", required=True)
    p.add_argument("--policy-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--case-set", choices=("smoke8", "full16"), default="smoke8")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or subprocess.check_output(["git", "-C", str(a.reference_root), "status", "--porcelain"])
        or subprocess.check_output(
            ["git", "-C", str(a.reference_root), "rev-parse", "HEAD"], text=True
        ).strip()
        != a.reference_head
    ):
        raise ValueError("unused external output and exact clean source checkouts required")
    policy_path = a.reference_root / "ckp/policy_30000.onnx"
    if digest(policy_path) != a.policy_hash:
        raise ValueError("exact actual soccer LSTM required")
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES

    pins = {str(policy_path): a.policy_hash}
    for f in (a.reference_root / "README.md", a.reference_root / "LICENSE.md"):
        pins[str(f)] = digest(f)
    for root in (
        Path.cwd() / "src",
        Path.cwd() / "scripts",
        a.reference_root / "exp",
        a.reference_root / "source/whole_body_tracking/soccer/assets/unitree_description",
    ):
        for f in root.rglob("*"):
                if f.is_file() and (f.suffix.lower() in (".py", ".xml", ".stl", ".obj", ".png")):
                pins[str(f)] = digest(f)
    for name in TRAIN_MOTION_NAMES:
        file = a.reference_root / "motions/soccer-standard" / name
        pins[str(file)] = digest(file)
    check_pins(pins)
    profiles = [
        ("cached-light", 0.1, False),
        ("cached-standard", 0.43, False),
        ("aligned-standard", 0.43, True),
    ]
    cases = [
        dict(
            source=n,
            variant=v,
            ball_local_xy=[
                0.65 if v == 0 else 1.0,
                (0.15 if "left" in n else -0.15) if v == 0 else 0.0,
            ],
            ball_velocity_local=[0.0, 0.0] if v == 0 else [-0.2, 0.1 if "left" in n else -0.1],
        )
        for n in TRAIN_MOTION_NAMES
        for v in range(1 if a.case_set == "smoke8" else 2)
    ]
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            reference_head=a.reference_head,
            input_file_hashes=pins,
            profiles=profiles,
            cases=cases,
            repetitions=2,
            maximum_reference_controls=300,
            maximum_held_target_tail_controls=60,
            maximum_native_steps=len(cases) * 3 * 2 * 360 * 10,
            maximum_replay_steps=len(cases) * 3 * 2 * 360 * 10,
            maximum_neural_rows=len(cases) * 3 * 2 * 300,
            exported260_reference_outputs_unused=True,
            action_and_hc_graph_independent_of_exported_time_index=True,
            current_state_forward_before_observation=True,
            original_initial_world_angular_velocity_reset_semantics=True,
            no_reference_padding=True,
            tail="hold last actual target, no NN/ref input",
            body_guard_height_m=0.65,
            body_guard_upright=0.9,
            guard_at_control_boundary=True,
            license="CC-BY-NC4.0 external reference: research only",
            commercial_publicity_authorized=False,
            formal52_unchanged=True,
            optimizer_updates=0,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    live = replay = neural = attempted_neural = 0
    rows, nnrows, subrows = [], [], []
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy

        sys.path.insert(0, str(a.reference_root))
        from exp.mujoco_soccer.constants import (
            BODY_NAMES,
            DEFAULT_MJCF,
            ISAACLAB_JOINT_NAMES,
            JOINT_EFFORT_LIMIT_ARRAY,
        )
        from exp.mujoco_soccer.math_utils import quat_apply
        from exp.mujoco_soccer.motion import load_motion_clips
        from exp.mujoco_soccer.observations import build_observation
        from exp.mujoco_soccer.scene import MujocoSoccer, make_experiment_xml

        if mujoco.get_mjcb_control() is not None:
            raise ValueError("no external control callback allowed")
        # Qualify weights and160 ABI BEFORE any MjData allocation. No NN call.
        contract = PaidResearchPolicy(policy_path, expected_hash=a.policy_hash, horizon=260)
        if contract.joint_names != ISAACLAB_JOINT_NAMES:
            raise ValueError("published ISAACLAB actor order must match named metadata exactly")
        params = {k: v[contract.hg_from_policy].copy() for k, v in contract.parameters.items()}
        offsets, scales, kp, kd = [
            params[k]
            for k in ("default_joint_pos", "action_scale", "joint_stiffness", "joint_damping")
        ]
        effort = JOINT_EFFORT_LIMIT_ARRAY.copy()
        if effort.shape != (29,) or not np.isfinite(effort).all() or (effort <= 0).any():
            raise ValueError("positive original named29 torque clipping limits required")
        clips = {}
        for name in TRAIN_MOTION_NAMES:
            file = a.reference_root / "motions/soccer-standard" / name
            with np.load(file, allow_pickle=False) as z:
                length = len(z["joint_pos"])
                for key, shape in dict(
                    joint_pos=(length, 29),
                    joint_vel=(length, 29),
                    body_pos_w=(length, 30, 3),
                    body_quat_w=(length, 30, 4),
                    body_lin_vel_w=(length, 30, 3),
                    body_ang_vel_w=(length, 30, 3),
                ).items():
                    if z[key].shape != shape or not np.isfinite(z[key]).all():
                        raise ValueError("complete exact30body actual motion required")
                if (
                    not 1 <= length <= 300
                    or float(z["fps"].reshape(-1)[0]) != 50
                    or abs(np.linalg.norm(z["body_quat_w"], axis=2) - 1).max() > 1e-5
                ):
                    raise ValueError("bounded original50Hz source and unit quaternions required")
            clips[name] = load_motion_clips(file)[0]
        models, model_paths, layouts = {}, {}, {}
        for profile, mass, _ in profiles:
            folder = a.output / profile
            folder.mkdir()
            source_xml = make_experiment_xml(DEFAULT_MJCF, folder, goal_width=2.0)
            tree = ET.parse(source_xml)
            tree.getroot().find(".//geom[@name='soccer_ball_geom']").set("mass", str(mass))
            # Generated experiment asset only, never overwrite external reference.
            xml = folder / "declared-scene.xml"
            tree.write(xml, encoding="utf-8", xml_declaration=True)
            pins[str(xml)] = digest(xml)
            env = MujocoSoccer(xml)
            m = env.model
            layout = NativeVelocityLayout(m, robot_prefix="", ball_body_name="soccer_ball")
            if (
                (m.nq, m.nv, m.nu) != (43, 41, 29)
                or m.opt.timestep != 0.002
                or abs(m.body_subtreemass[layout.pelvis] - 33.341142) > 1e-9
                or abs(m.body_mass[layout.ball] - mass) > 1e-12
                or not np.allclose(
                    m.body_inertia[layout.ball], 0.4 * mass * 0.11**2, atol=1e-12, rtol=0
                )
                or m.geom_size[env.ball_geom_id, 0] != 0.11
                or not np.array_equal(env.joint_qpos_addr, layout.qi)
                or not np.array_equal(env.joint_dof_addr, layout.vi)
                or not np.array_equal(env.actuator_ids, layout.ai)
                or m.actuator_ctrllimited.any()
                or m.actuator_forcelimited.any()
                or m.actuator_biastype.any()
                or m.actuator_dyntype.any()
                or not np.all(m.actuator_gainprm[:, 0] == 1)
                or not np.all(m.actuator_gear[:, 0] == 1)
                or m.actuator_gear[:, 1:].any()
            ):
                raise ValueError(
                    "original torque-motor body/PD map and declared actual sphere required"
                )
            models[profile], layouts[profile] = env, layout
            model_paths[profile] = xml
            # Every declared ball case preflighted; no resampling or dropping.
            for case in cases:
                clip = clips[case["source"]]
                quat = clip.body_quat_w[0, 0].astype(np.float64)
                ball = quat_apply(quat, np.r_[case["ball_local_xy"], 0.0])
                ball[2] = 0.11
                velocity = quat_apply(quat, np.r_[case["ball_velocity_local"], 0.0])[:2]
                env.reset(clip, np.zeros(2), ball, velocity)
                if env.has_ball_robot_contact():
                    raise ValueError("declared initial robot/ball overlap; entire probe rejected")
        check_pins(pins)
        write_once(
            a.output / "preflight.json",
            dict(
                input_file_hashes=pins,
                exact_160_ABI_and_named29_PD=True,
                all_initial_ball_cases_no_overlap=True,
                ball_radius_m=0.11,
                ball_masses_kg=[0.1, 0.43],
                robot_mass_kg=33.341142,
                source_fps=50,
                native_dt_s=0.002,
                neural_rows=0,
                physics_steps=0,
                commercial_publicity_authorized=False,
                hardware_authorized=False,
            ),
        )
        results, raw_hashes = [], {}
        for profile, mass, aligned in profiles:
            env, layout = models[profile], layouts[profile]
            m, d = env.model, env.data
            foot_bodies = [m.body(s + "_ankle_roll_link").id for s in ("left", "right")]
            robot = set(env.body_ids.values())

            def forces(data, m=m, ball=layout.ball, robot=robot, foot_bodies=foot_bodies):
                value = np.zeros(3)
                for ci in range(data.ncon):
                    c = data.contact[ci]
                    x, y = (int(m.geom_bodyid[g]) for g in c.geom)
                    other = y if x == ball else x if y == ball else -1
                    if other in robot:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(m, data, ci, wrench)
                        role = foot_bodies.index(other) if other in foot_bodies else 2
                        value[role] = max(value[role], np.linalg.norm(wrench[:3]))
                return value

            for case_id, case in enumerate(cases):
                first_raw = None
                clip = clips[case["source"]]
                for repeat in range(2):
                    quat = clip.body_quat_w[0, 0].astype(np.float64)
                    ball = quat_apply(quat, np.r_[case["ball_local_xy"], 0.0])
                    ball[2] = 0.11
                    velocity = quat_apply(quat, np.r_[case["ball_velocity_local"], 0.0])[:2]
                    goal = quat_apply(quat, np.array([6.0, 0.0, 0.0]))
                    goal[2] = 0.11
                    env.reset(clip, np.zeros(2), ball, velocity)
                    normal = goal[:2] / np.linalg.norm(goal[:2])
                    env.set_goal_marker(goal, normal)
                    policy = PaidResearchPolicy(
                        policy_path, expected_hash=a.policy_hash, horizon=clip.length
                    )
                    initial = {
                        k: getattr(d, k).copy()
                        for k in (
                            "qpos",
                            "qvel",
                            "qacc_warmstart",
                            "ctrl",
                            "mocap_pos",
                            "mocap_quat",
                        )
                    }
                    initial["time"] = np.array(d.time)
                    initial_h, initial_c = policy.h.copy(), policy.c.copy()
                    last_action = np.zeros(29, dtype=np.float32)
                    ref_pos, ref_vel, ref_ang = (
                        np.zeros(29, dtype=np.float32),
                        np.zeros(29, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                    )
                    rows, nnrows, subrows = [], [], []
                    warning = tuple(int(x.number) for x in d.warning)
                    for frame in range(clip.length + 60):
                        mujoco.mj_forward(m, d)
                        is_neural = frame < clip.length
                        if is_neural:
                            if aligned:
                                ref_pos, ref_vel = clip.joint_pos[frame], clip.joint_vel[frame]
                                ref_ang = clip.body_ang_vel_w[frame, BODY_NAMES.index("torso_link")]
                            obs = build_observation(
                                env,
                                clip,
                                frame,
                                last_action,
                                ref_pos,
                                ref_vel,
                                ref_ang,
                                env.ball_pos,
                                goal,
                                offsets,
                                obs_dim=None,
                            )
                            step = policy.step(obs, frame=frame)
                            neural += 1
                            attempted_neural += 1
                            nnrows.append(step)
                            last_action = step["action"].copy()
                            target = offsets + last_action[contract.hg_from_policy] * scales
                            if not aligned:
                                ref_pos, ref_vel = clip.joint_pos[frame], clip.joint_vel[frame]
                                ref_ang = clip.body_ang_vel_w[frame, BODY_NAMES.index("torso_link")]
                        row = dict(
                            native_qpos_before=d.qpos.copy(),
                            native_qvel_before=d.qvel.copy(),
                            time_before=float(d.time),
                            target=target.copy(),
                            neural_control=is_neural,
                        )
                        subrows = []
                        for _ in range(10):
                            torque = np.clip(
                                kp * (target - env.joint_pos) - kd * env.joint_vel, -effort, effort
                            )
                            if not np.isfinite(torque).all():
                                raise ValueError("nonfinite teacher target/PD; no repair")
                            d.ctrl[layout.ai] = torque
                            mujoco.mj_step(m, d)
                            live += 1
                            if tuple(
                                int(x.number) for x in d.warning
                            ) != warning or not np.array_equal(d.actuator_force[layout.ai], torque):
                                raise ValueError(
                                    "native warning or original clipped torque law differs"
                                )
                            q, v = layout.canonical(d)
                            subrows.append(
                                dict(qpos=q, qvel=v, forces=forces(d), torque=torque.copy())
                            )
                        row.update({k: np.stack([s[k] for s in subrows]) for k in subrows[0]})
                        q = row["qpos"]
                        row["body_failed"] = bool(
                            q[:, 2].min() < 0.65
                            or (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).min() < 0.9
                        )
                        rows.append(row)
                        if row["body_failed"]:
                            break
                    raw = {k: np.array([row[k] for row in rows]) for k in rows[0]}
                    raw.update({"NN_" + k: np.stack([row[k] for row in nnrows]) for k in nnrows[0]})
                    raw.update({"initial_" + k: v for k, v in initial.items()})
                    if not np.array_equal(raw["NN_h_before"][0], initial_h) or not np.array_equal(
                        raw["NN_c_before"][0], initial_c
                    ):
                        raise ValueError("episode hidden state not zero-initialized")
                    if len(nnrows) > 1 and (
                        not np.array_equal(raw["NN_h_before"][1:], raw["NN_h_after"][:-1])
                        or not np.array_equal(raw["NN_c_before"][1:], raw["NN_c_after"][:-1])
                    ):
                        raise ValueError("RNN history reset or cross-episode contamination")
                    if first_raw is not None and (
                        set(raw) != set(first_raw)
                        or any(not np.array_equal(raw[k], first_raw[k]) for k in raw)
                    ):
                        raise ValueError("whole independent LSTM/physics repeat differs")
                    first_raw = raw
                    path = a.output / profile / f"case{case_id}-repeat{repeat}.npz"
                    np.savez_compressed(path, **raw)
                    raw_hashes[str(path.relative_to(a.output))] = digest(path)
                    rd = mujoco.MjData(m)
                    for key in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
                        getattr(rd, key)[:] = initial[key]
                    rd.time = float(initial["time"])
                    mujoco.mj_forward(m, rd)
                    rd.qacc_warmstart[:] = initial["qacc_warmstart"]
                    for control in range(len(rows)):
                        mujoco.mj_forward(m, rd)
                        if not np.array_equal(
                            rd.qpos, raw["native_qpos_before"][control]
                        ) or not np.array_equal(rd.qvel, raw["native_qvel_before"][control]):
                            raise ValueError("recorded-torque replay entry differs")
                        for sub in range(10):
                            rd.ctrl[layout.ai] = raw["torque"][control, sub]
                            mujoco.mj_step(m, rd)
                            replay += 1
                            rq, rv = layout.canonical(rd)
                            if (
                                not np.array_equal(rq, raw["qpos"][control, sub])
                                or not np.array_equal(rv, raw["qvel"][control, sub])
                                or not np.array_equal(forces(rd), raw["forces"][control, sub])
                            ):
                                raise ValueError("strict native torque/q/v/force replay differs")
                    q, v, f = (
                        raw[k].reshape(-1, width)
                        for k, width in (("qpos", 43), ("qvel", 41), ("forces", 3))
                    )
                    post = native_contact_window(
                        force=f[:, :2],
                        canonical_qpos=q,
                        canonical_qvel=v,
                        goal=goal,
                        intended_foot=0 if "left" in case["source"] else 1,
                    )
                    result = dict(
                        profile=profile,
                        case=case_id,
                        repeat=repeat,
                        source=case["source"],
                        ball_mass_kg=mass,
                        controls=len(rows),
                        neural_rows=policy.neural_calls,
                        body_failed=bool(raw["body_failed"].any()),
                        forbidden=bool((f[:, 2] > 1).any()),
                        postcontact=post,
                        ball_world_initial=ball.tolist(),
                        goal_world=goal.tolist(),
                    )
                    results.append(result)
                    print(json.dumps(result), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_RESEARCH_NATIVE_TEACHER_PROBE_NOT_PROMOTED",
                input_file_hashes=pins,
                raw_hashes=raw_hashes,
                results=results,
                live_native_steps=live,
                replay_native_steps=replay,
                neural_rows=neural,
                strict_replays_exact=True,
                whole_repeats_exact=True,
                optimizer_updates=0,
                commercial_publicity_authorized=False,
                formal52_unchanged=True,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        import numpy as np

        for name, trace in (
            ("failed-controls", rows),
            ("failed-neural", nnrows),
            ("failed-substeps", subrows),
        ):
            if trace:
                np.savez_compressed(
                    a.output / (name + ".npz"),
                    **{k: np.array([r[k] for r in trace]) for k in trace[0]},
                )
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                live_native_steps=live,
                replay_native_steps=replay,
                neural_rows=neural,
                attempted_neural_rows=attempted_neural,
                optimizer_updates=0,
                commercial_publicity_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
