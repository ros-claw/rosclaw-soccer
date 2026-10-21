"""Qualify real contact24/recovery60 continuity, repeats and native replay.

Finite SIM probe only, no optimizer, new reward, checkpoint, or task promotion.
TRAIN64 lead10, immutable original foundation, no reference padding in tail.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("proof-root", "reference-root", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("proof-hash", "expected-head"):
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
        raise ValueError("new output and clean frozen source required")
    proof_path = args.proof_root / "complete.json"
    if digest(proof_path) != args.proof_hash:
        raise ValueError("immutable native contact-course proof required")
    proof = json.loads(proof_path.read_text())
    if proof["status"] != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL":
        raise ValueError("successful contact24 qualification required")
    pins = {**proof["input_file_hashes"], str(proof_path): args.proof_hash}
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.reference_root / "src"):
        for path in folder.rglob("*.py"):
            pins[str(path)] = digest(path)
    # The qualified contact implementation itself must remain identical.
    suffix = "/src/rosclaw_soccer/rsi/native_contact_course.py"
    old = {h for p, h in proof["input_file_hashes"].items() if p.endswith(suffix)}
    if old != {digest(Path.cwd() / suffix.lstrip("/"))}:
        raise ValueError("cannot silently change qualified contact24")
    check_pins(pins)
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_head=args.expected_head,
            input_file_hashes=pins,
            protocol="native-contact24-recovery60-probe-v1",
            worlds=64,
            lead=10,
            contact_only_baseline_episodes=1,
            independent_full_episodes=2,
            max_live_native_steps=122880,
            max_replay_native_steps=107520,
            max_neural_rows=12288,
            optimizer_updates=0,
            tail_command=[0.0, 0.0, 0.0],
            tail_suffix79="zero: no reference and no ball perception",
            original_177_foundation_frozen=True,
            reset_at_contact_tail_boundary=False,
            tail_body_guard=dict(height_m=0.65, upright=0.9),
            tail_guards_at_control_boundary=True,
            formal52_unchanged=True,
            goal_complete=False,
            hardware_authorized=False,
        ),
    )
    course = None
    neural = replay = 0
    arrays = []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.contact_replication_decision import ball_outcome
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.native_recovery_tail import NativeRecoveryTail
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
        from scripts.rsi_native_course_inputs import load_body_references

        torch.set_num_threads(1)

        def pinned(end):
            found = [Path(p) for p in pins if p.endswith(end)]
            if len(found) != 1:
                raise ValueError("exact pinned prerequisite required: " + end)
            return found[0]

        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        bp = pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz")
        cp = pinned("/ball-aware-update200.pt")
        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        config = pinned("/policy/velocity/v0/params/deploy.yaml")
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
        with np.load(bp, allow_pickle=False) as z:
            bank = {k: z[k].copy() for k in z.files}
        plans = []
        for motion in TRAIN_MOTION_NAMES:
            path = pinned(
                "/rsi-train-starts-body-bound-index-check-20261021/"
                + motion.removesuffix(".npz")
                + "-plan.npz"
            )
            with np.load(path, allow_pickle=False) as z:
                plans.append({k: z[k].copy() for k in z.files})
        refs = load_body_references(args.reference_root, model, layout, plans)
        course = NativeContactCourse(
            model=model,
            params=params,
            bank=bank,
            plans=plans,
            body_references=refs,
            worlds=64,
            lead=10,
            threads=8,
        )
        actor = VelocityPriorActor.from_onnx(
            onnx, expected_hash=pins[str(onnx)], reference_features=79
        ).eval()
        state = torch.load(cp, map_location="cpu", weights_only=True)
        actor.load_state_dict(
            {k: state["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
            strict=True,
        )
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        actor_before = {k: v.clone() for k, v in actor.state_dict().items()}

        def act(obs, ids):
            nonlocal neural
            action = np.zeros((64, 29), dtype=np.float32)
            if len(ids):
                with torch.no_grad():
                    action[ids] = actor(torch.from_numpy(obs[ids])).numpy()
                neural += len(ids)
            return action

        def head():
            course.reset()
            failed = np.zeros(64, dtype=bool)
            rows = []
            for _ in range(24):
                if course.ended.all():
                    break
                action = act(course.observation, np.flatnonzero(~course.ended))
                info = course.step(action)
                failed |= info["terminated"]
                rows.append(dict(**info, action=action))
            return {k: np.array([r[k] for r in rows]) for k in rows[0]}, failed

        baseline, _ = head()
        np.savez_compressed(args.output / "contact-only.npz", bank_index=course.index, **baseline)
        summaries, hashes = [], []
        first = None
        qi = [*range(layout.rq, layout.rq + 7), *layout.qi, *range(layout.bq, layout.bq + 7)]
        vi = [*range(layout.rv, layout.rv + 6), *layout.vi, *range(layout.bv, layout.bv + 6)]
        for episode in range(2):
            contact, failed = head()
            if set(contact) != set(baseline) or any(
                not np.array_equal(contact[k], baseline[k]) for k in baseline
            ):
                raise ValueError(
                    "first24 prefix differs from independent unchanged contact-only baseline"
                )
            snapshot = [
                (d, d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), d.ctrl.copy(), d.time)
                for d in course.batch.data
            ]
            tail = NativeRecoveryTail(course, contact_failed=failed)
            if any(
                d is not old
                or any(
                    not np.array_equal(x, y)
                    for x, y in (
                        (d.qpos, q),
                        (d.qvel, v),
                        (d.qacc_warmstart, w),
                        (d.ctrl, c),
                        (d.time, t),
                    )
                )
                for d, (old, q, v, w, c, t) in zip(course.batch.data, snapshot, strict=True)
            ):
                raise ValueError("handoff changed native state or data identity")
            tails = []
            for _ in range(60):
                ids = tail.active_ids
                if not len(ids):
                    break
                stopped = {
                    int(i): (d.qpos.copy(), d.qvel.copy(), d.time)
                    for i, d in enumerate(course.batch.data)
                    if tail.ended[i]
                }
                action = act(tail.observation, ids)
                info = tail.step(action)
                for i, (q, v, t) in stopped.items():
                    d = course.batch.data[i]
                    if (
                        not np.array_equal(d.qpos, q)
                        or not np.array_equal(d.qvel, v)
                        or d.time != t
                    ):
                        raise ValueError("ended world advanced during recovery")
                tails.append(info)
            raw = {"head_" + k: v for k, v in contact.items()}
            if tails:
                raw.update({"tail_" + k: np.array([r[k] for r in tails]) for k in tails[0]})
            raw["bank_index"] = course.index.copy()
            if first is not None and (
                set(raw) != set(first) or any(not np.array_equal(raw[k], first[k]) for k in raw)
            ):
                raise ValueError("whole independent neural/physical option repeat differs")
            first = raw
            path = args.output / f"option{episode}.npz"
            np.savez_compressed(path, **raw)
            arrays.append(raw)
            hashes.append(digest(path))
            results = []
            for world in range(64):
                index = int(course.index[world])
                rd = mujoco.MjData(model)
                rd.qpos[:], rd.qvel[:] = bank["qpos"][index], bank["qvel"][index]
                rd.time = float(bank["reference_frame"][index] * 0.02)
                rd.ctrl[:] = 0
                from rosclaw_soccer.providers.g1.velocity_policy import target29

                rd.ctrl[layout.ai] = target29(bank["previous_action"][index], params)
                mujoco.mj_forward(model, rd)
                rd.qacc_warmstart[:] = bank["qacc_warmstart"][index]
                qq, vv, ff = [], [], []
                for part in ("head_", "tail_"):
                    if part + "valid" not in raw:
                        continue
                    for step in range(len(raw[part + "valid"])):
                        if not raw[part + "valid"][step, world]:
                            continue
                        rd.ctrl[layout.ai] = raw[part + "target"][step, world]
                        for sub in range(10):
                            mujoco.mj_step(model, rd)
                            replay += 1
                            forces = course._forces(rd)
                            q = raw[part + "qpos"][step, world, sub]
                            v = raw[part + "qvel"][step, world, sub]
                            f = raw[part + "forces"][step, world, sub]
                            if (
                                not np.array_equal(rd.qpos, q)
                                or not np.array_equal(rd.qvel, v)
                                or not np.array_equal(forces, f)
                            ):
                                raise ValueError("strict recorded-control native replay differs")
                            qq.append(q[qi])
                            vv.append(v[vi])
                            ff.append(f)
                        mujoco.mj_forward(model, rd)
                cq, cv, force = np.array(qq), np.array(vv), np.array(ff)
                post = native_contact_window(
                    force=force[:, :2],
                    canonical_qpos=cq,
                    canonical_qvel=cv,
                    goal=bank["goal_world"][index],
                    intended_foot=int(bank["expected_foot"][index]),
                )
                result = dict(
                    world=world,
                    bank_index=index,
                    contact_failed=bool(failed[world]),
                    tail_failed=bool(tail.failed[world]),
                    tail_controls=int(tail.elapsed[world]),
                    postcontact=post,
                    stable=bool(
                        cq[:, 2].min() >= 0.65
                        and (1 - 2 * (cq[:, 4] ** 2 + cq[:, 5] ** 2)).min() >= 0.9
                    ),
                    forbidden=bool((force[:, 2] > 1).any()),
                )
                result["ball_outcome"] = ball_outcome(result)
                results.append(result)
            summaries.append(
                dict(
                    episode=episode,
                    results=results,
                    stable=sum(r["stable"] for r in results),
                    forbidden=sum(r["forbidden"] for r in results),
                    observed=sum(
                        r["postcontact"]["post_contact_quality_observed"] for r in results
                    ),
                    ball_outcome=sum(r["ball_outcome"] for r in results),
                )
            )
            print(
                json.dumps({k: v for k, v in summaries[-1].items() if k != "results"}), flush=True
            )
        if any(not torch.equal(v, actor.state_dict()[k]) for k, v in actor_before.items()):
            raise ValueError("foundation weights changed during probe")
        check_pins(pins)
        write_once(
            args.output / "complete.json",
            dict(
                status="NATIVE_CONTACT24_RECOVERY60_PROBE_NOT_LEARNER",
                input_file_hashes=pins,
                baseline_hash=digest(args.output / "contact-only.npz"),
                raw_hashes=hashes,
                summaries=summaries,
                live_native_steps=course.native_steps,
                replay_native_steps=replay,
                policy_rows=neural,
                prefix_exact=True,
                handoff_state_exact=True,
                repeats_exact=True,
                strict_replays_exact=True,
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failed.json",
            dict(
                error=repr(error),
                native_steps=course.native_steps if course else 0,
                replay_steps=replay,
                policy_rows=neural,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise
    finally:
        if course is not None:
            course.close()


if __name__ == "__main__":
    main()
