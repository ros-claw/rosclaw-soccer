"""All64 TRAIN frame-zero full-horizon native neural exam, not a football pass.

The pilot trains at measured precontact states; this exam MUST begin at the
original frame-zero states. Fixed final20 vs frozen initial, no checkpoint pick.
Every rollout has an independent plain native recorded-control physics replay.
Eight representative sources repeat complete neural execution independently.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("learner-root", "starts-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("learner-hash", "starts-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--baseline-episodes-root", type=Path)
    p.add_argument("--baseline-episodes-hash")
    p.add_argument("--task-set", choices=("train64", "dev64"), default="train64")
    p.add_argument("--baseline-reuse-root", type=Path)
    p.add_argument("--baseline-reuse-hash")
    p.add_argument("--checkpoint-intent", choices=("shared", "bounded-residual"), default="shared")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("new frozen exam and unused output required")
    dev = a.task_set == "dev64"
    if (not dev and not all((a.baseline_episodes_root, a.baseline_episodes_hash))) or (
        dev and any((a.baseline_episodes_root, a.baseline_episodes_hash))
    ):
        raise ValueError("TRAIN requires historical baseline; new DEV must not borrow it")
    repeated_worlds = [0, 1, 2, 3, 32, 33, 34, 35] if dev else list(range(8))
    pins, reports = {}, []
    for key in ("learner", "starts") if dev else ("learner", "starts", "baseline_episodes"):
        f = getattr(a, key + "_root") / "complete.json"
        h = getattr(a, key + "_hash")
        if digest(f) != h:
            raise ValueError("immutable completed prerequisite required")
        r = json.loads(f.read_text())
        reports.append(r)
        pins.update(r["input_file_hashes"])
        pins[str(f)] = h
    learner, starts = reports[:2]
    baseline = {} if dev else reports[2]
    residual = a.checkpoint_intent == "bounded-residual"
    reuse_rows = {}
    if a.baseline_reuse_root or a.baseline_reuse_hash:
        if not all((a.baseline_reuse_root, a.baseline_reuse_hash)) or not residual:
            raise ValueError(
                "explicit immutable baseline reuse is only defined for frozen residuals"
            )
        rf = a.baseline_reuse_root / "complete.json"
        if digest(rf) != a.baseline_reuse_hash:
            raise ValueError("exact complete baseline execution proof required")
        reused = json.loads(rf.read_text())
        if (
            reused.get("checkpoint_intent") != "bounded-residual"
            or reused.get("task_set", "train64") != a.task_set
            or reused.get("all_strict_replays_exact") is not True
            or reused.get("all_representative_repeats_exact") is not True
            or (not dev and reused.get("all_old_baseline_fields_bitexact") is not True)
            or reused["input_file_hashes"].get(str(a.starts_root / "complete.json"))
            != a.starts_hash
        ):
            raise ValueError("same task instances and actually replayed baseline required")
        declaration_path = a.baseline_reuse_root / "declaration.json"
        original_head = json.loads(declaration_path.read_text())["source_head"]
        roots = [
            Path(k).parent.parent
            for k in reused["input_file_hashes"]
            if k.endswith("/scripts/rsi_exam_native_contact_policy.py")
        ]
        active_roots = [
            root
            for root in roots
            if subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            == original_head
        ]
        if len(active_roots) != 1:
            raise ValueError("unambiguous original active baseline source required")
        for relative in (
            "src/rosclaw_soccer/rsi/velocity_prior_actor.py",
            "src/rosclaw_soccer/rsi/native_reference_observation.py",
            "src/rosclaw_soccer/rsi/native_velocity_transfer.py",
            "src/rosclaw_soccer/providers/g1/velocity_policy.py",
            "src/rosclaw_soccer/rsi/ball_aware_inputs.py",
            "src/rosclaw_soccer/rsi/native_motion_reference.py",
            "src/rosclaw_soccer/rsi/native_contact_adapter.py",
            "src/rosclaw_soccer/rsi/native_contact_checkpoint.py",
        ):
            if digest(Path.cwd() / relative) != reused["input_file_hashes"].get(
                str(active_roots[0] / relative)
            ):
                raise ValueError(
                    "baseline inference/observation/codec behavior changed; cannot reuse"
                )
        reuse_rows = {
            (r["world"], r["repeat"]): r for r in reused["results"] if r["arm"] == "initial"
        }
        if set(reuse_rows) != {
            (w, rep) for w in range(64) for rep in range(2 if w in repeated_worlds else 1)
        }:
            raise ValueError("complete balanced64 baseline with all declared repeats required")
        for suffix in (
            "/ball-aware-update200.pt",
            "/policy/velocity/v0/exported/policy.onnx",
            "/policy/velocity/v0/params/deploy.yaml",
            "/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb",
        ):
            current = {k: h for k, h in pins.items() if k.endswith(suffix)}
            original = {k: h for k, h in reused["input_file_hashes"].items() if k.endswith(suffix)}
            if len(current) != 1 or current != original:
                raise ValueError("baseline body, codec and immutable foundation must match")
        pins.update(reused["input_file_hashes"])
        pins[str(rf)] = a.baseline_reuse_hash
        pins[str(declaration_path)] = digest(declaration_path)
        for (w, rep), row in reuse_rows.items():
            pins[str(a.baseline_reuse_root / f"world{w}-initial-repeat{rep}/native.npz")] = row[
                "raw_hash"
            ]
    expected_status = (
        "NATIVE_CONTACT24_RESIDUAL_PPO_CANDIDATE_REQUIRES_FULL_EXAM"
        if residual
        else "NATIVE_CONTACT24_PPO_CANDIDATE_REQUIRES_FULL_EXAM"
    )
    if (
        learner.get("status") != expected_status
        or learner.get("actor_kind", "shared") != a.checkpoint_intent
        or learner.get("protocol") != "native-contact24-v1"
        or len(learner.get("results", [])) != 20
        or learner["counts"]["ppo_steps"] < 1
        or starts.get("ball_placement_protocol") != "body-bound-v1"
        or (
            not dev
            and (
                starts.get("worlds") != 64
                or baseline.get("ball_placement_protocol") != "body-bound-v1"
            )
        )
        or (
            dev
            and (
                starts.get("status") != "NATIVE_BODY_BOUND_DEV_TASKS_NOT_POLICY_EVIDENCE"
                or starts.get("source_split") != "DEV evaluation only"
                or [(r.get("source"), r.get("worlds")) for r in starts.get("results", [])]
                != [("soccer-standard-007_left.npz", 32), ("soccer-standard-010_right.npz", 32)]
            )
        )
    ):
        raise ValueError(
            "completed native learner and unchanged full body-bound TRAIN64 source required"
        )
    checkpoint = next(r for r in learner["checkpoints"] if r["episode"] == 20)
    cp = a.learner_root / checkpoint["file"]
    pins[str(cp)] = checkpoint["hash"]
    initial_path = a.starts_root / "train-starts.npz"
    if dev:
        for r in starts["results"]:
            folder = a.starts_root / r["source"].removesuffix(".npz")
            pins[str(folder / "starts.npz")] = r["starts_hash"]
            pins[str(folder / "plan.npz")] = r["plan_hash"]
    else:
        pins[str(initial_path)] = starts["starts_hash"]
        for name, h in starts["plan_hashes"].items():
            pins[str(a.starts_root / (name.removesuffix(".npz") + "-plan.npz"))] = h
        for r in baseline["results"]:
            pins[str(a.baseline_episodes_root / f"world{r['world']}-repeat{r['repeat']}.npz")] = r[
                "raw_hash"
            ]
    for root in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_split="DEV evaluation only, no gradients"
            if dev
            else "TRAIN evaluation only, DEV untouched",
            task_set=a.task_set,
            worlds=64,
            checkpoint_selection="fixed final20 vs immutable parent baseline",
            checkpoint_intent=a.checkpoint_intent,
            baseline_reused=bool(reuse_rows),
            baseline_reuse_receipt_hash=a.baseline_reuse_hash,
            reused_baseline_is_not_new_physics_or_new_independent_data=True,
            arms=["initial", "final"],
            full_independent_repeated_worlds=repeated_worlds,
            maximum_controls=300,
            maximum_live_native_steps=432000,
            maximum_replay_native_steps=432000,
            maximum_policy_rows=43200,
            optimizer_updates=0,
            start="original frame zero, NOT precontact snapshots",
            no_state_overwrites=True,
            body_ball_motor_colliders_and_original_goal_initial_states_unchanged=True,
            baseline_must_reproduce_all_old_native_fields_bitexact=not dev,
            strict_physical_replay="all substep canonical q/v/feet/nonfoot bitexact",
            correct_foot_threshold_n=10.0,
            forbidden_contact_threshold_n=1.0,
            stable_thresholds=dict(pelvis_height_m=0.65, upright=0.9),
            not_independent_holdout_or_goalkeeper_team_evidence=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    live_steps = replay_steps = nn_rows = 0
    raw_rows = []
    current_world = current_arm = None
    results = []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.mjlab_motion_curriculum import (
            DEVELOPMENT_MOTION_NAMES,
            TRAIN_MOTION_NAMES,
        )
        from rosclaw_soccer.rsi.native_contact_checkpoint import (
            NativeContactSession,
            NativeResidualContactSession,
        )
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.native_reference_observation import native_reference_observation177
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

        torch.set_num_threads(1)
        session_type = NativeResidualContactSession if residual else NativeContactSession

        def pinned(suffix):
            matches = [Path(k) for k in pins if k.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError("one exact pinned input required: " + suffix)
            return matches[0]

        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        parent = pinned("/ball-aware-update200.pt")
        cfg = pinned("/policy/velocity/v0/params/deploy.yaml")
        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        feet = [model.body("robot/" + s + "_ankle_roll_link").id for s in ("left", "right")]
        torso = model.body("robot/torso_link").id
        names = DEVELOPMENT_MOTION_NAMES if dev else TRAIN_MOTION_NAMES
        plans = []
        if dev:
            batches = []
            for name in names:
                folder = a.starts_root / name.removesuffix(".npz")
                with np.load(folder / "starts.npz", allow_pickle=False) as z:
                    batches.append({k: z[k].copy() for k in z.files})
                for key, shape in {
                    "qpos": (32, 43),
                    "qvel": (32, 41),
                    "observation": (32, 177),
                    "goal_world": (32, 3),
                    "origins": (32, 3),
                    "expected_foot": (32,),
                }.items():
                    if batches[-1][key].shape != shape or not np.isfinite(batches[-1][key]).all():
                        raise ValueError("complete finite declared DEV32 source required")
                if not np.all(batches[-1]["expected_foot"] == len(batches) - 1):
                    raise ValueError("DEV task foot identity changed")
            initial = {k: np.concatenate([batch[k] for batch in batches]) for k in batches[0]}
            initial["motion_id"] = np.repeat(np.arange(2), 32)
        else:
            with np.load(initial_path, allow_pickle=False) as z:
                initial = {k: z[k].copy() for k in z.files}
            if not np.array_equal(initial["motion_id"], np.arange(64) % 8):
                raise ValueError("unchanged balanced TRAIN64 source identities required")
        for name in names:
            plan_path = (
                (a.starts_root / name.removesuffix(".npz") / "plan.npz")
                if dev
                else (a.starts_root / (name.removesuffix(".npz") + "-plan.npz"))
            )
            with np.load(plan_path, allow_pickle=False) as z:
                plans.append({k: z[k].copy() for k in z.files})
            if not 1 <= len(plans[-1]["joint"]) <= 300 or (
                dev and len(plans[-1]["joint"]) != starts["results"][len(plans) - 1]["controls"]
            ):
                raise ValueError("full declared bounded reference horizon required")

        def force(data):
            ff, nonfoot = np.zeros(2), 0.0
            for ci in range(data.ncon):
                c = data.contact[ci]
                x, y = (int(model.geom_bodyid[g]) for g in c.geom)
                other = y if x == layout.ball else x if y == layout.ball else -1
                if other >= 0 and (model.body(other).name or "").startswith("robot/"):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, wrench)
                    mag = float(np.linalg.norm(wrench[:3]))
                    if other in feet:
                        ff[feet.index(other)] = max(ff[feet.index(other)], mag)
                    else:
                        nonfoot = max(nonfoot, mag)
            return ff, nonfoot

        for world in range(64):
            current_world = world
            sid = int(initial["motion_id"][world])
            plan, origin, goal = plans[sid], initial["origins"][world], initial["goal_world"][world]
            horizon = len(plan["joint"])
            for arm in ("initial", "final"):
                current_arm = arm
                first = None
                for repeat in range(2 if world in repeated_worlds else 1):
                    if arm == "initial" and reuse_rows:
                        import os

                        row = reuse_rows[world, repeat]
                        source = (
                            a.baseline_reuse_root
                            / f"world{world}-initial-repeat{repeat}/native.npz"
                        )
                        # Admit the new checkpoint's zero-head/foundation identity,
                        # even though it requires no new baseline integration.
                        session_type(
                            onnx,
                            cfg,
                            expected_model_hash=pins[str(onnx)],
                            expected_config_hash=pins[str(cfg)],
                            checkpoint=cp,
                            expected_checkpoint_hash=checkpoint["hash"],
                            parent_checkpoint=parent,
                            expected_parent_hash=pins[str(parent)],
                            expected_native_model_hash=pins[str(mp)],
                            arm="initial",
                        )
                        with np.load(source, allow_pickle=False) as z:
                            if (
                                not np.array_equal(
                                    z["native_qpos_before"][0], initial["qpos"][world]
                                )
                                or not np.array_equal(
                                    z["native_qvel_before"][0], initial["qvel"][world]
                                )
                                or len(z["target_rad"]) != horizon
                            ):
                                raise ValueError("reused baseline entry or full horizon differs")
                        folder = a.output / f"world{world}-initial-repeat{repeat}"
                        folder.mkdir()
                        os.symlink(source.resolve(), folder / "native.npz")
                        result = {
                            **row,
                            "baseline_reused_from": str(source),
                            "new_physics_steps": 0,
                            "new_policy_rows": 0,
                        }
                        write_once(folder / "result.json", result)
                        results.append(result)
                        continue
                    session = session_type(
                        onnx,
                        cfg,
                        expected_model_hash=pins[str(onnx)],
                        expected_config_hash=pins[str(cfg)],
                        checkpoint=cp,
                        expected_checkpoint_hash=checkpoint["hash"],
                        parent_checkpoint=parent,
                        expected_parent_hash=pins[str(parent)],
                        expected_native_model_hash=pins[str(mp)],
                        arm=arm,
                    )
                    d = mujoco.MjData(model)
                    d.qpos[:], d.qvel[:] = initial["qpos"][world], initial["qvel"][world]
                    warnings = tuple(int(w.number) for w in d.warning)
                    raw_rows = []
                    for frame in range(horizon):
                        mujoco.mj_forward(model, d)
                        obs, phase_after = native_reference_observation177(
                            data=d,
                            layout=layout,
                            params=session.params,
                            plan=plan,
                            frame=frame,
                            origin=origin,
                            goal=goal,
                            last_action=session.last_action,
                            phase=session.phase,
                            torso=torso,
                        )
                        if (
                            dev
                            and frame == 0
                            and not np.allclose(
                                obs, initial["observation"][world], atol=2e-5, rtol=0
                            )
                        ):
                            raise ValueError(
                                "native DEV first177 input differs from actual reset export"
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
                        target = session.propose_ball_aware(
                            q,
                            v,
                            layout.angular_velocity_body(d),
                            obs[6:9].copy(),
                            obs[98:168].copy(),
                            obs[168:177].copy(),
                            frame=frame,
                        )
                        nn_rows += 1
                        actual_obs = np.r_[session.last_observation, session.last_reference]
                        if not np.array_equal(obs, actual_obs) or session.phase != phase_after:
                            raise ValueError(
                                "new checkpoint input/history differs from actual inference"
                            )
                        row.update(
                            observation=actual_obs,
                            action=session.last_action.copy(),
                            target_rad=target,
                        )
                        d.ctrl[layout.ai] = target
                        qs, vs, fs, ns = [], [], [], []
                        for _ in range(10):
                            mujoco.mj_step(model, d)
                            live_steps += 1
                            if tuple(int(w.number) for w in d.warning) != warnings:
                                raise ValueError("native warning in full source exam")
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
                                raise ValueError("original native PD differs during exam")
                            q, v = layout.canonical(d)
                            ff, nf = force(d)
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
                        raw_rows.append(row)
                    raw = {k: np.array([r[k] for r in raw_rows]) for k in raw_rows[0]}
                    folder = a.output / f"world{world}-{arm}-repeat{repeat}"
                    folder.mkdir()
                    path = folder / "native.npz"
                    np.savez_compressed(path, **raw)
                    if repeat and any(not np.array_equal(raw[k], first[k]) for k in raw):
                        raise ValueError(
                            "complete independent NN/physics representative repeat differs"
                        )
                    first = raw
                    if arm == "initial" and not dev:
                        with np.load(
                            a.baseline_episodes_root / f"world{world}-repeat0.npz",
                            allow_pickle=False,
                        ) as z:
                            if set(z.files) != set(raw) or any(
                                not np.array_equal(z[k], raw[k]) for k in z.files
                            ):
                                raise ValueError(
                                    "new parser initial arm did not reproduce complete baseline"
                                )
                    rd = mujoco.MjData(model)
                    rd.qpos[:], rd.qvel[:] = initial["qpos"][world], initial["qvel"][world]
                    for frame in range(horizon):
                        mujoco.mj_forward(model, rd)
                        if not np.array_equal(
                            rd.qpos, raw["native_qpos_before"][frame]
                        ) or not np.array_equal(rd.qvel, raw["native_qvel_before"][frame]):
                            raise ValueError("strict physical replay entry differs")
                        rd.ctrl[layout.ai] = raw["target_rad"][frame]
                        for sub in range(10):
                            mujoco.mj_step(model, rd)
                            replay_steps += 1
                            rq, rv = layout.canonical(rd)
                            ff, nf = force(rd)
                            if (
                                not np.array_equal(rq, raw["canonical_qpos_after"][frame, sub])
                                or not np.array_equal(rv, raw["canonical_qvel_after"][frame, sub])
                                or not np.array_equal(ff, raw["foot_force"][frame, sub])
                                or nf != raw["nonfoot_force"][frame, sub]
                            ):
                                raise ValueError("strict physical replay state or force differs")
                    cq = raw["canonical_qpos_after"].reshape(-1, 43)
                    cv = raw["canonical_qvel_after"].reshape(-1, 41)
                    post = native_contact_window(
                        force=raw["foot_force"].reshape(-1, 2),
                        canonical_qpos=cq,
                        canonical_qvel=cv,
                        goal=goal,
                        intended_foot=int(initial["expected_foot"][world]),
                    )
                    forbidden = bool((raw["nonfoot_force"] > 1).any())
                    stable = bool(
                        cq[:, 2].min() >= 0.65
                        and (1 - 2 * (cq[:, 4] ** 2 + cq[:, 5] ** 2)).min() >= 0.9
                    )
                    result = dict(
                        world=world,
                        source=names[sid],
                        arm=arm,
                        repeat=repeat,
                        raw_hash=digest(path),
                        controls=horizon,
                        correct_foot_first=post["correct_foot_first"],
                        forbidden=forbidden,
                        clean=bool(post["correct_foot_first"] and not forbidden),
                        stable=stable,
                        postcontact=post,
                        strict_recorded_control_replay_exact=True,
                    )
                    write_once(folder / "result.json", result)
                    results.append(result)
                    print(
                        json.dumps(
                            dict(
                                world=world,
                                arm=arm,
                                repeat=repeat,
                                clean=result["clean"],
                                stable=stable,
                                forbidden=forbidden,
                                native_steps=live_steps,
                                replay_steps=replay_steps,
                                policy_rows=nn_rows,
                            )
                        ),
                        flush=True,
                    )
        summary = {}
        for arm in ("initial", "final"):
            selected = [r for r in results if r["arm"] == arm and r["repeat"] == 0]
            summary[arm] = {
                k: sum(bool(r[k]) for r in selected)
                for k in ("clean", "correct_foot_first", "forbidden", "stable")
            }
        improved = (
            summary["final"]["clean"] > summary["initial"]["clean"]
            and summary["final"]["stable"] >= summary["initial"]["stable"]
            and summary["final"]["forbidden"] <= summary["initial"]["forbidden"]
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "NATIVE_CONTACT24_DEV64_GAIN_REQUIRES_FRESH"
                    if improved
                    else "NATIVE_CONTACT24_DEV64_NO_NET_GAIN"
                )
                if dev
                else (
                    "NATIVE_CONTACT24_TRAIN64_GAIN_REQUIRES_UNTOUCHED_DEV"
                    if improved
                    else "NATIVE_CONTACT24_TRAIN64_CANDIDATE_REJECTED"
                ),
                input_file_hashes=pins,
                checkpoint_intent=a.checkpoint_intent,
                baseline_reuse_receipt_hash=a.baseline_reuse_hash,
                reused_baseline_trajectories=len(reuse_rows),
                results=results,
                summary=summary,
                live_native_steps=live_steps,
                replay_native_steps=replay_steps,
                policy_rows=nn_rows,
                optimizer_updates=0,
                all_old_baseline_fields_bitexact=True if not dev else None,
                task_set=a.task_set,
                all_strict_replays_exact=True,
                all_representative_repeats_exact=True,
                not_independent_holdout=True,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            world=current_world,
            arm=current_arm,
            live_native_steps=live_steps,
            replay_native_steps=replay_steps,
            policy_rows=nn_rows,
            hardware_authorized=False,
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        if raw_rows:
            np.savez_compressed(
                a.output / "failed-partial.npz",
                **{k: np.array([r[k] for r in raw_rows]) for k in raw_rows[0]},
            )
        raise


if __name__ == "__main__":
    main()
