"""Bounded TRAIN-only whole-run expert search on actual inherited motor states.

No fabricated teacher labels, borrowed hidden state, suffix teleport, new motor
weights or qualification. Each trial starts a separately declared world and
executes the full measured run/contact/standing sequence.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("audit-root", "source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("audit-hash", "expected-head", "motor-lineage-hash"):
        p.add_argument("--" + name, required=True)
    p.add_argument(
        "--profile", choices=("paired_v1", "contextual_v2", "response_v3"), default="paired_v1"
    )
    p.add_argument("--response-root", type=Path)
    p.add_argument("--response-hash")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen expert-search source required")

    clean()
    ap, sp, dp = (
        a.audit_root / "complete.json",
        a.source_root / "complete.json",
        a.source_root / "declaration.json",
    )
    if a.output.exists() or any(x.is_symlink() for x in (ap, sp, dp)) or digest(ap) != a.audit_hash:
        raise ValueError("unused output and exact ordinary audited source required")
    audit, source, decl = [json.loads(x.read_text()) for x in (ap, sp, dp)]
    if (
        audit["status"] != "INHERITED_BASE_RUN_WHOLE_NATIVE_OWN_NN_AUDIT_PASS"
        or audit["source_hash"] != digest(sp)
        or source["declaration_hash"] != digest(dp)
        or audit["own_native_steps"] != 33920
        or not audit["independent_whole_world_replay"]
        or audit["pristine_published_R2"] is not False
    ):
        raise ValueError("audited inherited motor actual run witness required")
    pins = dict(audit["input_file_hashes"])
    for x in (ap, sp, dp):
        pins[str(x)] = digest(x)
    paths = [
        Path(x)
        for x, h in pins.items()
        if h == a.motor_lineage_hash and x.endswith("/complete.json")
    ]
    if len(paths) != 1:
        raise ValueError("exact unique canonical motor lineage required")
    lineage = json.loads(paths[0].read_text())
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    response = a.profile == "response_v3"
    response_report = None
    if response:
        if a.response_root is None or a.response_hash is None:
            raise ValueError("response trial requires exact audited diagnostic commitment")
        rr = a.response_root / "complete.json"
        if rr.is_symlink() or digest(rr) != a.response_hash:
            raise ValueError("exact response diagnostic required")
        response_report = json.loads(rr.read_text())
        import numpy as np

        if (
            response_report["status"] != "TRAIN_RUN_CONTACT_RESPONSE_DIAGNOSED_NOT_MOTOR_POLICY"
            or response_report["physical_worlds_executed"] != 0
            or response_report["predicted_success_is_evidence"] is not False
            or response_report["hardware_authorized"] is not False
            or response_report["promotion_authorized"] is not False
            or [x["course"] for x in response_report["contexts"]] != [0, 1]
            or any(len(x["proposed_latents"]) != 11 for x in response_report["contexts"])
        ):
            raise ValueError("bounded two-context predeclared response experiment required")
        for context in response_report["contexts"]:
            proposals = np.array(context["proposed_latents"], np.float64)
            if (
                proposals.shape != (11, 2, 6)
                or not np.isfinite(proposals).all()
                or (np.abs(proposals) > 1).any()
            ):
                raise ValueError("all response interventions must qualify before any allocation")
        for path, h in response_report["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("conflicting motor/body or diagnostic provenance")
            pins[path] = h
        pins[str(rr)] = a.response_hash
    elif a.response_root is not None or a.response_hash is not None:
        raise ValueError("response inputs cannot silently change legacy experiments")
    check_pins(pins)
    contextual = a.profile != "paired_v1"
    centers = [105, 105]
    if contextual:
        for ci in range(2):
            witness = next(
                x
                for x in source["records"]
                if x["course"] == ci and x["arm"] == "reference_from_start" and x["repeat"] == 0
            )
            first = witness["result"]["postcontact"]["first_foot_contact_substep"]
            if type(first) is not int or not 30 <= first // 10 <= 200:
                raise ValueError("actual TRAIN first contact phase required before allocation")
            centers[ci] = first // 10
    if response and [x["center"] for x in response_report["contexts"]] != centers:
        raise ValueError("response must preserve original physical contact phases")
    max_worlds = 28 if response else 134 if contextual else 38
    max_steps = max_worlds * 3890
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol=(
                "paid.TRAIN.inherited_run_actual_state_expert.response_v3"
                if response
                else "paid.TRAIN.inherited_run_actual_state_expert.contextual_v2"
                if contextual
                else "paid.TRAIN.inherited_run_actual_state_expert.v1"
            ),
            source_head=a.expected_head,
            response_hash=a.response_hash,
            input_file_hashes=pins,
            source_hash=digest(sp),
            audit_hash=a.audit_hash,
            motor_lineage_hash=a.motor_lineage_hash,
            source_case=decl["case"],
            courses=decl["courses"],
            center=105,
            centers=centers,
            independent_context_search=contextual,
            seed=41022,
            rounds=0 if response else 4 if contextual else 2,
            candidates_per_round=11 if response else 16 if contextual else 8,
            elite=0 if response else 4 if contextual else 2,
            initial_std=None if response else 0.2 if contextual else 0.6,
            minimum_std=None if response else 0.03 if contextual else 0.1,
            zero_baseline=True,
            selected_whole_repeats=2,
            maximum_worlds=max_worlds,
            maximum_native_steps=max_steps,
            selection=(
                "contextual task_success/useful_stable_forward_kick BEFORE precision"
                if contextual
                else (
                    "lexicographic paired healthy_quality,correct_foot,no_forbidden,"
                    "observed,negative_lateral,forward_progress"
                )
            ),
            maximum_neural_calls=max_steps // 10,
            DEV_rows=0,
            Fresh_rows=0,
            optimizer_updates=0,
            learned_policy_claim=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    nn = ik_calls = worlds = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, [decl["case"]])
        wrapper, _ = load_paid_student(
            policy,
            checkpoint=Path(lineage["checkpoint_path"]),
            checkpoint_hash=lineage["checkpoint_hash"],
            training_receipt=paths[0],
            training_receipt_hash=a.motor_lineage_hash,
            scene_hash=scene,
        )
        actor = wrapper.base
        records = []

        def collect(latent, course_index, tag):
            nonlocal nn, ik_calls, worlds
            if worlds >= max_worlds:
                raise ValueError("bounded world allocation exceeded")
            folder = a.output / tag
            folder.mkdir()
            ep = bench.make(decl["case"])
            data = ep.env.data
            q, _ = ep.layout.canonical(data)
            w, x, y, z = q[3:7]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            data.qpos[ep.layout.rq : ep.layout.rq + 2] -= decl["courses"][course_index][
                "backoff_m"
            ] * np.array([np.cos(yaw), np.sin(yaw)])
            mujoco.mj_forward(ep.env.model, data)
            initial = snapshot(data)
            intent = FootTaskSpaceIntent(
                ep.env.model,
                foot="left" if "left" in decl["case"]["source"] else "right",
                center=centers[course_index],
                latent=latent,
            )
            state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
            rows = []
            while not ep.done and ep.frame < ep.clip.length:
                obs = ep.observation()
                before = [s.numpy().copy() for s in state]
                with torch.no_grad():
                    mean, state = actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
                nn += 1
                nominal = mean[0, 0].numpy().copy()
                physical_before = snapshot(data)
                action, plan = intent(ep, nominal)
                ik_calls += 1
                if any(
                    not np.array_equal(v, snapshot(data)[k]) for k, v in physical_before.items()
                ):
                    raise ValueError("kinematic intent changed actual physics")
                row = ep.step(action)
                row.update(
                    NN_obs=obs,
                    NN_nominal_action=nominal,
                    NN_action=action.copy(),
                    NN_h_before=before[0],
                    NN_c_before=before[1],
                    NN_h_after=state[0].numpy().copy(),
                    NN_c_after=state[1].numpy().copy(),
                )
                row.update({"PLAN_" + k: v for k, v in plan.items()})
                rows.append(row)
                if ep.frame == ep.clip.length and not ep.done:
                    ep.admit_reference_recovery()
            raw = {k: np.array([v[k] for v in rows]) for k in rows[0]}
            raw.update({"initial_" + k: v for k, v in initial.items()})
            raw.update({"terminal_" + k: v for k, v in snapshot(data).items()})
            sr, stand = None, None
            if ep.termination_reason == "RECOVERY_ADMITTED":

                def count():
                    nonlocal nn
                    nn += 1

                sr, stand = NeuralStandingTask(
                    ep, actor, state=state, memory_mode="carry", count_inference=count
                ).collect(100)
            allq = np.concatenate([raw["qpos"], sr["qpos"]]) if sr is not None else raw["qpos"]
            allv = np.concatenate([raw["qvel"], sr["qvel"]]) if sr is not None else raw["qvel"]
            allf = (
                np.concatenate([raw["forces"], sr["forces"]]) if sr is not None else raw["forces"]
            )
            f = allf.reshape(-1, 3)
            post = native_contact_window(
                force=f[:, :2],
                canonical_qpos=allq.reshape(-1, 43),
                canonical_qvel=allv.reshape(-1, 41),
                goal=ep.goal,
                intended_foot=0 if "left" in decl["case"]["source"] else 1,
            )
            result = dict(
                body_failed=bool(
                    raw["body_failed"].any() or (sr is not None and sr["body_failed"].any())
                ),
                forbidden=bool((f[:, 2] > 1).any()),
                nonfoot_impulse_Ns=float(f[:, 2].sum() * 0.002),
                postcontact=post,
            )
            outcome = dict(
                tag=tag,
                course=course_index,
                latent=latent.tolist(),
                result=result,
                stand=stand,
                healthy_quality=bool(
                    contact_quality(result)
                    and stand
                    and stand["healthy"]
                    and stand["stability_observed"]
                ),
                native_steps=len(allq) * 10,
                neural_rows=len(rows) + (len(sr["qpos"]) if sr is not None else 0),
                artifacts={},
            )
            for name, z in (("kick", raw), ("stand", sr)):
                if z is not None:
                    path = folder / (name + ".npz")
                    np.savez_compressed(path, **z)
                    outcome["artifacts"][name] = dict(
                        path=str(path.relative_to(a.output)), hash=digest(path)
                    )
            write_once(folder / "complete.json", outcome)
            records.append(outcome)
            worlds += 1
            print(
                json.dumps(
                    dict(
                        world=worlds,
                        tag=tag,
                        course=course_index,
                        quality=outcome["healthy_quality"],
                        forbidden=result["forbidden"],
                        lateral=post["goal_lateral_1p2s_m"],
                    )
                ),
                flush=True,
            )
            if bench.native_steps > max_steps or nn > max_steps // 10:
                raise ValueError("charged native budget exceeded")
            return outcome, dict(kick=raw, stand=sr)

        def rank(pair):
            rr = [x[0] for x in pair]
            posts = [x["result"]["postcontact"] for x in rr]
            return (
                sum(x["healthy_quality"] for x in rr),
                sum(p["correct_foot_first"] for p in posts),
                sum(not x["result"]["forbidden"] for x in rr),
                sum(p["post_contact_quality_observed"] for p in posts),
                -sum(
                    p["goal_lateral_1p2s_m"] if p["goal_lateral_1p2s_m"] is not None else 100
                    for p in posts
                ),
                sum(
                    p["goal_progress_1p2s_m"] if p["goal_progress_1p2s_m"] is not None else -100
                    for p in posts
                ),
            )

        baseline = [collect(np.zeros((2, 6)), ci, f"baseline-c{ci}") for ci in range(2)]
        best, best_pair = np.zeros((2, 6)), baseline
        context_ranks = None
        if contextual:
            from rosclaw_soccer.rsi.run_teacher_objective import run_teacher_rank

            context_latents = []
            context_pairs = []
            context_ranks = []
            for ci in range(2):
                context_best = baseline[ci]
                context_latent = np.zeros((2, 6))
                context_rng = np.random.default_rng(41022 + ci)
                context_mean, context_std = np.zeros((2, 6)), np.full((2, 6), 0.2)
                if response:
                    for candidate, proposed in enumerate(
                        response_report["contexts"][ci]["proposed_latents"]
                    ):
                        latent = np.array(proposed, np.float64)
                        if (
                            latent.shape != (2, 6)
                            or not np.isfinite(latent).all()
                            or (np.abs(latent) > 1).any()
                        ):
                            raise ValueError("bounded preregistered response intervention required")
                        pair = collect(latent, ci, f"r-n{candidate}-c{ci}")
                        if run_teacher_rank(pair[0]) > run_teacher_rank(context_best[0]):
                            context_best, context_latent = pair, latent.copy()
                for generation in range(0 if response else 4):
                    trials = []
                    for candidate in range(16):
                        latent = np.clip(context_rng.normal(context_mean, context_std), -1, 1)
                        pair = collect(latent, ci, f"g{generation}-n{candidate}-c{ci}")
                        score = run_teacher_rank(pair[0])
                        trials.append((score, candidate, latent))
                        if score > run_teacher_rank(context_best[0]):
                            context_best, context_latent = pair, latent.copy()
                    elites = sorted(trials, key=lambda x: (x[0], -x[1]), reverse=True)[:4]
                    values = np.stack([x[2] for x in elites])
                    context_mean = values.mean(axis=0)
                    context_std = np.maximum(values.std(axis=0), 0.03)
                    write_once(
                        a.output / f"course{ci}-generation{generation}.json",
                        dict(
                            course=ci,
                            best_latent=context_latent.tolist(),
                            best_rank=run_teacher_rank(context_best[0]),
                            mean=context_mean.tolist(),
                            std=context_std.tolist(),
                            native_steps=bench.native_steps,
                            worlds=worlds,
                        ),
                    )
                context_latents.append(context_latent)
                context_pairs.append(context_best)
                context_ranks.append(run_teacher_rank(context_best[0]))
            best, best_pair = np.stack(context_latents), context_pairs
        rng = np.random.default_rng(41022)
        mean, std = np.zeros((2, 6)), np.full((2, 6), 0.6)
        for generation in range(0 if contextual else 2):
            trials = []
            for candidate in range(8):
                latent = np.clip(rng.normal(mean, std), -1, 1)
                pair = [collect(latent, ci, f"g{generation}-n{candidate}-c{ci}") for ci in range(2)]
                trials.append((rank(pair), candidate, latent))
                if rank(pair) > rank(best_pair):
                    best, best_pair = latent.copy(), pair
            elites = sorted(trials, key=lambda x: (x[0], -x[1]), reverse=True)[:2]
            values = np.stack([x[2] for x in elites])
            mean, std = values.mean(axis=0), np.maximum(values.std(axis=0), 0.1)
            write_once(
                a.output / f"generation{generation}.json",
                dict(
                    best_latent=best.tolist(),
                    best_rank=rank(best_pair),
                    mean=mean.tolist(),
                    std=std.tolist(),
                    worlds=worlds,
                    native_steps=bench.native_steps,
                ),
            )
        for ci in range(2):
            for repeat in range(2):
                trial, blobs = collect(
                    best[ci] if contextual else best, ci, f"selected-c{ci}-repeat{repeat}"
                )
                for name in blobs:
                    if blobs[name] is None:
                        if best_pair[ci][1][name] is not None:
                            raise ValueError("selected replay censored differently")
                    else:
                        identical(blobs[name], best_pair[ci][1][name])
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="INHERITED_RUN_TRAIN_EXPERT_SEARCH_CLOSED_PENDING_INDEPENDENT_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                selected_latent=best.tolist(),
                selected_rank=rank(best_pair),
                selected_context_ranks=context_ranks,
                baseline_rank=rank(baseline),
                selected_whole_repeats_exact=True,
                records=records,
                actual_native_steps=bench.native_steps,
                actual_neural_calls=nn,
                actual_kinematic_intent_calls=ik_calls,
                worlds=worlds,
                optimizer_updates=0,
                learned_policy_claim=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                actual_native_steps=bench.native_steps if bench else 0,
                actual_neural_calls=nn,
                worlds=worlds,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
