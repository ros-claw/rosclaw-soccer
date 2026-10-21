"""Fixed16 TRAIN cases x3 privileged-reference arms, native PD and exact replay.

No gradients, no checkpoint selection, no DEV or Fresh execution. Tests whether
larger reference-guided contact targets can improve the measured ball outcome.
Not a deployable controller and not the original formal52 benchmark.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline-root", "starts-root", "learner-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("baseline-hash", "starts-hash", "learner-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != args.expected_head
        )
    ):
        raise ValueError("clean frozen source and unused output required")
    reports, pins = {}, {}
    for key in ("baseline", "starts", "learner"):
        path = getattr(args, key + "_root") / "complete.json"
        if digest(path) != getattr(args, key + "_hash"):
            raise ValueError("immutable completed prerequisite required")
        reports[key] = json.loads(path.read_text())
        pins.update(reports[key]["input_file_hashes"])
        pins[str(path)] = digest(path)
    if (
        reports["baseline"]["all_old_baseline_fields_bitexact"] is not True
        or reports["starts"]["ball_placement_protocol"] != "body-bound-v1"
        or reports["learner"]["actor_kind"] != "bounded-residual"
    ):
        raise ValueError("qualified original frame-zero native foundation required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        for path in folder.rglob("*.py"):
            pins[str(path)] = digest(path)
    cp = args.learner_root / "native-contact-episode20.pt"
    checkpoint = [r for r in reports["learner"]["checkpoints"] if r["episode"] == 20]
    if len(checkpoint) != 1 or checkpoint[0]["file"] != cp.name:
        raise ValueError("fixed final20 checkpoint identity required")
    pins[str(cp)] = checkpoint[0]["hash"]
    pins[str(args.starts_root / "train-starts.npz")] = reports["starts"]["starts_hash"]
    for name, h in reports["starts"]["plan_hashes"].items():
        pins[str(args.starts_root / (name.removesuffix(".npz") + "-plan.npz"))] = h
    for world in range(16):
        path = args.baseline_root / f"world{world}-initial-repeat0/native.npz"
        pins[str(path)] = digest(path)
        matched = [
            r
            for r in reports["baseline"]["results"]
            if (r["world"], r["arm"], r["repeat"]) == (world, "initial", 0)
        ]
        if len(matched) != 1 or matched[0]["raw_hash"] != pins[str(path)]:
            raise ValueError("exact original baseline trajectory required")
    check_pins(pins)
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.native_reference_authority.v1",
            source_head=args.expected_head,
            worlds=list(range(16)),
            gains=[0.0, 0.25, 0.5],
            max_target_correction_rad=0.25,
            gate_near_far_m=[0.35, 0.7],
            full_reference_horizon=True,
            original_PD_and_model_unchanged=True,
            privileged_reference=True,
            maximum_live_steps=144000,
            maximum_replay_steps=144000,
            optimizer_updates=0,
            input_file_hashes=pins,
            goal_complete=False,
            hardware_authorized=False,
        ),
    )
    live = replay = neural = 0
    rows, partial = [], []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.contact_replication_decision import ball_outcome
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_contact_checkpoint import NativeResidualContactSession
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.native_reference_authority import reference_contact_target
        from rosclaw_soccer.rsi.native_reference_observation import native_reference_observation177
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

        torch.set_num_threads(1)

        def pinned(suffix):
            matches = [Path(p) for p in pins if p.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError("exact unambiguous prerequisite required: " + suffix)
            return matches[0]

        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        parent = pinned("/ball-aware-update200.pt")
        config = pinned("/policy/velocity/v0/params/deploy.yaml")
        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        feet = [model.body("robot/" + s + "_ankle_roll_link").id for s in ("left", "right")]
        torso = model.body("robot/torso_link").id
        with np.load(args.starts_root / "train-starts.npz", allow_pickle=False) as z:
            starts = {k: z[k].copy() for k in z.files}
        if not np.array_equal(starts["motion_id"], np.arange(64) % 8):
            raise ValueError("unchanged balanced source identities required")
        plans = []
        for motion in TRAIN_MOTION_NAMES:
            with np.load(
                args.starts_root / (motion.removesuffix(".npz") + "-plan.npz"), allow_pickle=False
            ) as z:
                plans.append({k: z[k].copy() for k in z.files})
        if any(not 1 <= len(p["joint"]) <= 300 for p in plans):
            raise ValueError("bounded full reference horizon required")

        def contacts(data):
            ff, nf = np.zeros(2), 0.0
            for ci in range(data.ncon):
                x, y = [int(model.geom_bodyid[g]) for g in data.contact[ci].geom]
                other = y if x == layout.ball else x if y == layout.ball else -1
                if other >= 0 and (model.body(other).name or "").startswith("robot/"):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, wrench)
                    magnitude = float(np.linalg.norm(wrench[:3]))
                    if other in feet:
                        ff[feet.index(other)] = max(ff[feet.index(other)], magnitude)
                    else:
                        nf = max(nf, magnitude)
            return ff, nf

        for world in range(16):
            plan = plans[int(starts["motion_id"][world])]
            for gain in (0.0, 0.25, 0.5):
                session = NativeResidualContactSession(
                    onnx,
                    config,
                    expected_model_hash=pins[str(onnx)],
                    expected_config_hash=pins[str(config)],
                    checkpoint=cp,
                    expected_checkpoint_hash=pins[str(cp)],
                    parent_checkpoint=parent,
                    expected_parent_hash=pins[str(parent)],
                    expected_native_model_hash=pins[str(mp)],
                    arm="initial",
                )
                d = mujoco.MjData(model)
                d.qpos[:], d.qvel[:] = starts["qpos"][world], starts["qvel"][world]
                warning = tuple(int(w.number) for w in d.warning)
                partial, rejected = [], False
                for frame in range(len(plan["joint"])):
                    mujoco.mj_forward(model, d)
                    obs, next_phase = native_reference_observation177(
                        data=d,
                        layout=layout,
                        params=session.params,
                        plan=plan,
                        frame=frame,
                        origin=starts["origins"][world],
                        goal=starts["goal_world"][world],
                        last_action=session.last_action,
                        phase=session.phase,
                        torso=torso,
                    )
                    row = dict(
                        native_qpos_before=d.qpos.copy(),
                        native_qvel_before=d.qvel.copy(),
                        qacc_warmstart_before=d.qacc_warmstart.copy(),
                        previous_action=session.last_action.copy(),
                        phase_before=float(session.phase),
                        ball_to_feet=d.xpos[layout.ball] - d.xpos[feet],
                    )
                    q, v = layout.canonical(d)
                    base = session.propose_ball_aware(
                        q,
                        v,
                        layout.angular_velocity_body(d),
                        obs[6:9].copy(),
                        obs[98:168].copy(),
                        obs[168:177].copy(),
                        frame=frame,
                    )
                    neural += 1
                    if session.phase != next_phase or not np.array_equal(
                        obs, np.r_[session.last_observation, session.last_reference]
                    ):
                        raise ValueError("measured input/history differs")
                    target, gate = reference_contact_target(
                        base, plan["joint"][frame], obs[168:177], gain=gain
                    )
                    bounds = model.jnt_range[layout.joints]
                    if np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1]):
                        rejected = True
                        break
                    if not np.array_equal(target, base):
                        session.last_action = (
                            (target - np.asarray(session.params.action_offset, dtype=np.float32))
                            / np.asarray(session.params.action_scale, dtype=np.float32)
                        ).astype(np.float32)
                    row.update(
                        observation=obs,
                        action=session.last_action.copy(),
                        target_rad=target,
                        teacher_gate=gate,
                        foundation_target_rad=base,
                    )
                    d.ctrl[layout.ai] = target
                    qs, vs, fs, ns = [], [], [], []
                    for _ in range(10):
                        mujoco.mj_step(model, d)
                        live += 1
                        if tuple(int(w.number) for w in d.warning) != warning:
                            raise ValueError("native warning during authority probe")
                        ai = layout.ai
                        expected = np.clip(
                            model.actuator_gainprm[ai, 0] * target
                            + model.actuator_biasprm[ai, 0]
                            + model.actuator_biasprm[ai, 1] * d.actuator_length[ai]
                            + model.actuator_biasprm[ai, 2] * d.actuator_velocity[ai],
                            model.actuator_forcerange[ai, 0],
                            model.actuator_forcerange[ai, 1],
                        )
                        if not np.allclose(d.actuator_force[ai], expected, atol=1e-8, rtol=0):
                            raise ValueError("native PD law differs")
                        q, v = layout.canonical(d)
                        ff, nf = contacts(d)
                        qs.append(q)
                        vs.append(v)
                        fs.append(ff)
                        ns.append(nf)
                    row.update(
                        canonical_qpos_after=np.array(qs),
                        canonical_qvel_after=np.array(vs),
                        foot_force=np.array(fs),
                        nonfoot_force=np.array(ns),
                    )
                    partial.append(row)
                folder = args.output / f"world{world}-gain{gain}"
                folder.mkdir()
                if not partial:
                    raise ValueError("empty trajectory; preserve failed campaign")
                raw = {k: np.array([r[k] for r in partial]) for k in partial[0]}
                path = folder / "native.npz"
                np.savez_compressed(path, **raw)
                if gain == 0:
                    with np.load(
                        args.baseline_root / f"world{world}-initial-repeat0/native.npz",
                        allow_pickle=False,
                    ) as z:
                        if any(not np.array_equal(z[k], raw[k]) for k in z.files):
                            raise ValueError(
                                "gain0 must reproduce every original baseline field exactly"
                            )
                rd = mujoco.MjData(model)
                rd.qpos[:], rd.qvel[:] = starts["qpos"][world], starts["qvel"][world]
                for frame in range(len(partial)):
                    mujoco.mj_forward(model, rd)
                    if not np.array_equal(
                        rd.qpos, raw["native_qpos_before"][frame]
                    ) or not np.array_equal(rd.qvel, raw["native_qvel_before"][frame]):
                        raise ValueError("replay entry differs")
                    rd.ctrl[layout.ai] = raw["target_rad"][frame]
                    for sub in range(10):
                        mujoco.mj_step(model, rd)
                        replay += 1
                        q, v = layout.canonical(rd)
                        ff, nf = contacts(rd)
                        if any(
                            not np.array_equal(x, y)
                            for x, y in (
                                (q, raw["canonical_qpos_after"][frame, sub]),
                                (v, raw["canonical_qvel_after"][frame, sub]),
                                (ff, raw["foot_force"][frame, sub]),
                                (nf, raw["nonfoot_force"][frame, sub]),
                            )
                        ):
                            raise ValueError("strict recorded-control physical replay differs")
                cq = raw["canonical_qpos_after"].reshape(-1, 43)
                post = native_contact_window(
                    force=raw["foot_force"].reshape(-1, 2),
                    canonical_qpos=cq,
                    canonical_qvel=raw["canonical_qvel_after"].reshape(-1, 41),
                    goal=starts["goal_world"][world],
                    intended_foot=int(starts["expected_foot"][world]),
                )
                result = dict(
                    world=world,
                    gain=gain,
                    controls=len(partial),
                    guard_rejected=rejected,
                    raw_hash=digest(path),
                    postcontact=post,
                    correct_foot_first=post["correct_foot_first"],
                    forbidden=bool((raw["nonfoot_force"] > 1).any()),
                    stable=bool(
                        cq[:, 2].min() >= 0.65
                        and (1 - 2 * (cq[:, 4] ** 2 + cq[:, 5] ** 2)).min() >= 0.9
                    ),
                    strict_replay_exact=True,
                    maximum_target_correction_rad=float(
                        np.abs(raw["target_rad"] - raw["foundation_target_rad"]).max()
                    ),
                )
                result["clean"] = bool(
                    post["correct_foot_first"] and not result["forbidden"] and not rejected
                )
                result["ball_outcome"] = bool(not rejected and ball_outcome(result))
                rows.append(result)
                write_once(folder / "result.json", result)
                print(
                    json.dumps(
                        dict(
                            world=world,
                            gain=gain,
                            clean=result["clean"],
                            stable=result["stable"],
                            ball_outcome=result["ball_outcome"],
                            guard_rejected=rejected,
                            live_steps=live,
                        )
                    ),
                    flush=True,
                )
        check_pins(pins)
        summary = {
            str(g): {
                k: sum(bool(r[k]) for r in rows if r["gain"] == g)
                for k in (
                    "clean",
                    "correct_foot_first",
                    "forbidden",
                    "stable",
                    "ball_outcome",
                    "guard_rejected",
                )
            }
            for g in (0.0, 0.25, 0.5)
        }
        write_once(
            args.output / "complete.json",
            dict(
                status="NATIVE_REFERENCE_AUTHORITY_DIAGNOSTIC_NOT_SKILL",
                input_file_hashes=pins,
                results=rows,
                summary=summary,
                live_native_steps=live,
                replay_native_steps=replay,
                policy_rows=neural,
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        if partial:
            np.savez_compressed(
                args.output / "failed-partial.npz",
                **{k: np.array([r[k] for r in partial]) for k in partial[0]},
            )
        write_once(
            args.output / "failed.json",
            dict(
                error=repr(error),
                live_native_steps=live,
                replay_native_steps=replay,
                policy_rows=neural,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
