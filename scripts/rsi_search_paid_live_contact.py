"""Learn bounded contact teachers on actual post-kick bodies, not reset poses.

Full first-kick and trainer-drop physics builds sixteen parent worlds. CEM
branches native mjData copies of those parents, explicitly simulation search,
not independent full matches. Selected sequences are rerun from time zero.
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def qualified_successor_bootstrap(report, audit, *, report_path, report_hash, training_hash):
    """Select an audited global entry policy, never a per-case deployment winner."""
    if (
        report.get("status") != "PAID_LIVE_DISTILLED_CPU_FULL_WORLD_EXAM_CLOSED"
        or report.get("entry_controller") != "candidate"
        or audit.get("status") != "PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS"
        or audit.get("input_file_hashes", {}).get(str(report_path)) != report_hash
    ):
        raise ValueError("audited single-policy full-world bootstrap required")
    selected = [x for x in report["candidates"] if x["training_receipt_hash"] == training_hash]
    if len(selected) != 1:
        raise ValueError("one exact trained bootstrap policy required")
    candidate = selected[0]
    if (
        set(candidate["isolated"]) != {"old16", "dev16"}
        or not report.get("all_whole_neural_native_repeats_exact")
        or report.get("IK_executed") is not False
        or not candidate.get("single_recurrent_actor")
        or any(
            x["quality"] < 14 or x["stable"] != 16 or x["regressions"]
            for x in candidate["isolated"].values()
        )
        or candidate["continuous"]["stable"] != 16
        or len(candidate["continuous"]["results"]) != 16
        or any(
            not w["first_quality"] or not w["world_stable"] or w["second"] is None
            for w in candidate["continuous"]["results"]
        )
    ):
        raise ValueError("all initial skills and whole-world bodies must be qualified")
    return candidate


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--handoff-root", type=Path, required=True)
    p.add_argument("--handoff-hash", required=True)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--exam-root", type=Path, required=True)
    p.add_argument("--bootstrap-exam-root", type=Path)
    p.add_argument("--bootstrap-exam-hash")
    p.add_argument("--bootstrap-audit-root", type=Path)
    p.add_argument("--bootstrap-audit-hash")
    p.add_argument("--source-index", type=int, choices=range(8))
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    hp, tp = a.handoff_root / "complete.json", a.training_root / "complete.json"
    if (
        a.output.exists()
        or digest(hp) != a.handoff_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("exact completed continuous-body sources and unused output required")
    handoff = json.loads(hp.read_text())
    training = json.loads(tp.read_text())
    pins = {**handoff["input_file_hashes"], str(hp): a.handoff_hash, str(tp): digest(tp)}
    boot_args = (
        a.bootstrap_exam_root,
        a.bootstrap_exam_hash,
        a.bootstrap_audit_root,
        a.bootstrap_audit_hash,
    )
    bootstrap = None
    if any(x is not None for x in boot_args):
        if not all(x is not None for x in boot_args):
            raise ValueError("complete bootstrap exam/audit commitments required")
        bxp, bap = a.bootstrap_exam_root / "complete.json", a.bootstrap_audit_root / "complete.json"
        if digest(bxp) != a.bootstrap_exam_hash or digest(bap) != a.bootstrap_audit_hash:
            raise ValueError("bootstrap proof identity differs")
        bootstrap_report, bootstrap_audit = json.loads(bxp.read_text()), json.loads(bap.read_text())
        bootstrap = qualified_successor_bootstrap(
            bootstrap_report,
            bootstrap_audit,
            report_path=bxp,
            report_hash=a.bootstrap_exam_hash,
            training_hash=digest(tp),
        )
        if bootstrap["checkpoint_hash"] != training["final_checkpoint_hash"]:
            raise ValueError("bootstrap must execute this exact trained checkpoint")
        pins.update(bootstrap_report["input_file_hashes"])
        pins.update({str(bxp): a.bootstrap_exam_hash, str(bap): a.bootstrap_audit_hash})
    cp = a.training_root / ("live-distilled-epoch256.pt" if bootstrap else "distilled-epoch256.pt")
    if bootstrap is None and handoff["input_file_hashes"].get(str(tp)) != digest(tp):
        raise ValueError("training receipt differs")
    pins[str(tp)] = digest(tp)
    pins[str(cp)] = training["final_checkpoint_hash"]
    if bootstrap:
        pins[str(a.training_root / "declaration.json")] = digest(
            a.training_root / "declaration.json"
        )
        pins[str(a.training_root / "live-distilled-epoch0.pt")] = training[
            "initial_checkpoint_hash"
        ]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    ep_path = a.exam_root / "complete.json"
    if pins.get(str(ep_path)) != digest(ep_path):
        raise ValueError("original exam must be bound by completed handoff receipt")
    exam = json.loads(ep_path.read_text())
    edp = ep_path.parent / "declaration.json"
    if str(edp) not in pins or (
        bootstrap is None and digest(tp) != exam["candidates"][0]["training_receipt_hash"]
    ):
        raise ValueError("qualified v0 and complete original exam required")
    cases = json.loads(edp.read_text())["old_cases"]
    names = list(dict.fromkeys(c["source"] for c in cases))
    if len(cases) != 16 or len(names) != 8:
        raise ValueError("both previous first-kick conditions for each source required")
    for i, row in enumerate(exam["candidates"][0]["results"]["old16"]["results"]):
        path = ep_path.parent / "candidate0/old16" / f"case{i}-repeat0.npz"
        pins[str(path)] = row["raw_hash"]
    first_references = []
    second_references = []
    if bootstrap:
        bdp = a.bootstrap_exam_root / "declaration.json"
        pins[str(bdp)] = digest(bdp)
        if json.loads(bdp.read_text())["old_cases"] != cases:
            raise ValueError("bootstrap must preserve the original sixteen training courses")
        for w in bootstrap["continuous"]["results"]:
            pair = []
            for key in ("first", "second"):
                path = a.bootstrap_exam_root / w[key]["raw_path"]
                if not path.resolve().is_relative_to(a.bootstrap_exam_root.resolve()):
                    raise ValueError("bootstrap reference escaped the exam")
                pins[str(path)] = w[key]["raw_hash"]
                pair.append(dict(path=str(path), hash=w[key]["raw_hash"]))
            first_references.append(pair[0])
            second_references.append(pair[1])
    for row in handoff["results"]:
        if row["spec"]["transition_controls"] == 0:
            for artifact in row["artifacts"]:
                pins[str(a.handoff_root / artifact["path"])] = artifact["hash"]
    check_pins(pins)
    source_indices = list(range(8)) if a.source_index is None else [a.source_index]
    final_indices = [i for i, c in enumerate(cases) if names.index(c["source"]) in source_indices]
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            TRAIN_cases=cases,
            DEV_cases=[],
            search_source_indices=source_indices,
            final_case_indices=final_indices,
            bootstrap_first_references=first_references,
            bootstrap_second_references=second_references,
            bootstrap_training_receipt_hash=digest(tp) if bootstrap else None,
            base_model_checkpoint_hash=pins[str(cp)],
            first_kick_reference="learned_safe_successor" if bootstrap else "original_v0",
            parents_from_actual_first_kick_and_explicit_gravity_drop=True,
            branch_semantics="mj_copyData of full live parent; never call env.reset in a branch",
            first_kick_pure_neural=True,
            second_kick_composite_teacher=True,
            transition_controls=0,
            same_foot_second_kick=True,
            center_rule=(
                "mean contact controls from both baseline second kicks; "
                "declared control120 fallback if neither contacts"
            ),
            generations=3,
            population=12,
            elite=4,
            seed_base=20261022,
            z_initial_mean=0,
            z_initial_std=0.5,
            z_std_clip=[0.15, 1.2],
            smoothing=0.5,
            pose_scale=[0.04, 0.035, 0.04, 0.15, 0.2, 0.15],
            pose_bumps_relative_controls=[-12, 8],
            pose_bump_width=30,
            common_intent_per_source_both_live_bodies=True,
            max_joint_delta_rad=0.2,
            no_neural_optimizer_updates=True,
            selected_full_sequences_independent_repeats=2,
            all_generation_artifacts_sealed=True,
            formal52_unchanged=True,
            fresh_exam=False,
            team_game=False,
            goal_complete=False,
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    bench, cem_updates = None, 0
    try:
        from concurrent.futures import ThreadPoolExecutor

        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_distilled_actor import load_distilled_actor
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_live_distilled_actor import load_live_distilled_actor
        from rosclaw_soccer.rsi.paid_native_episode import NativeStepCounter
        from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench
        from rosclaw_soccer.rsi.paid_skill_handoff import practice_ball_drop, snapshot
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        xml = next(Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml"))
        contract = PaidResearchPolicy(policy, expected_hash=PUBLISHED_HASH, horizon=1)
        loader = load_live_distilled_actor if bootstrap else load_distilled_actor
        actor, _ = loader(
            policy,
            checkpoint=cp,
            checkpoint_hash=training["final_checkpoint_hash"],
            training_receipt=tp,
            training_receipt_hash=digest(tp),
            scene_hash=digest(xml),
        )
        ref = policy.parent.parent
        sys.path.insert(0, str(ref))
        from exp.mujoco_soccer.constants import BODY_NAMES, JOINT_EFFORT_LIMIT_ARRAY
        from exp.mujoco_soccer.math_utils import quat_apply
        from exp.mujoco_soccer.motion import load_motion_clips
        from exp.mujoco_soccer.observations import build_observation
        from exp.mujoco_soccer.scene import MujocoSoccer

        clips = {n: load_motion_clips(ref / "motions/soccer-standard" / n)[0] for n in names}
        bench = PaidResearchBench(
            xml=xml,
            contract=contract,
            clips=clips,
            scene_factory=MujocoSoccer,
            build_observation=build_observation,
            quat_apply=quat_apply,
            effort=JOINT_EFFORT_LIMIT_ARRAY,
            torso_index=BODY_NAMES.index("torso_link"),
        )
        for case in cases:
            bench.make(case)
        model = bench.template.model

        def save(path, raw):
            np.savez_compressed(path, **raw)
            return dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))

        def parent(i, folder):
            folder.mkdir()
            ep = bench.make(cases[i])
            raw, row = bench.collect_current(ep, actor)
            reference = (
                Path(first_references[i]["path"])
                if bootstrap
                else ep_path.parent / "candidate0/old16" / f"case{i}-repeat0.npz"
            )
            with np.load(reference, allow_pickle=False) as z:
                if set(raw) != set(z.files) or any(not np.array_equal(raw[k], z[k]) for k in raw):
                    raise ValueError(
                        "first-kick parent must reproduce its complete qualified reference"
                    )
            row.update(save(folder / "first.npz", raw))
            if ep.termination_reason != "REFERENCE_COMPLETE":
                raise ValueError("unstable first kick cannot be made into a reset parent")
            fixture = practice_ball_drop(
                ep, foot="left" if "left" in cases[i]["source"] else "right"
            )
            save(folder / "drop.npz", fixture)
            holds = []
            if fixture["performed"]:
                for _ in range(50):
                    holds.append(ep.advance_hold())
                    q, v = ep.layout.canonical(ep.env.data)
                    if holds[-1]["body_failed"] or (q[38] <= 0.12 and abs(v[37]) <= 0.1):
                        break
                save(folder / "hold.npz", {k: np.array([r[k] for r in holds]) for k in holds[0]})
                if holds[-1]["body_failed"] or not (q[38] <= 0.12 and abs(v[37]) <= 0.1):
                    raise ValueError("invalid live parent retained; no clipped/replaced course")
            values = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model, ep.env.data, values, mujoco.mjtState.mjSTATE_INTEGRATION)
            save(
                folder / "parent.npz",
                {
                    **snapshot(ep.env.data),
                    "full_integration_state": values,
                    "previous_actual_action": ep.last_action.copy(),
                },
            )
            write_once(
                folder / "complete.json",
                dict(
                    first=row,
                    body_reset_count=1,
                    native_steps=ep.native_steps,
                    no_reset_after_first=True,
                ),
            )
            return ep

        parents = [parent(i, a.output / f"parent{i}") for i in range(16)]
        centers = {}

        def branch(i):
            original = parents[i]
            ep = copy.copy(original)
            ep.env = copy.copy(original.env)
            ep.env.data = mujoco.MjData(model)
            mujoco.mj_copyData(ep.env.data, model, original.env.data)
            ep.work = NativeStepCounter()
            with bench.lock:
                bench.counters.append(ep.work)
            before = snapshot(ep.env.data)
            ep.begin_current_state(clips[cases[i]["source"]], goal=original.goal.copy())
            after = snapshot(ep.env.data)
            if any(not np.array_equal(before[k], after[k]) for k in before):
                raise ValueError("branch skill entry changed the physical body")
            return ep

        def run(i, latent, path, *, ep=None):
            ep = branch(i) if ep is None else ep
            intent = FootTaskSpaceIntent(
                model,
                foot="left" if "left" in cases[i]["source"] else "right",
                center=centers.get(cases[i]["source"], 120),
                latent=latent,
            )
            raw, row = bench.collect_current(ep, actor, transform=intent)
            row.update(save(path, raw))
            return raw, row

        baseline = []
        folder = a.output / "baseline"
        folder.mkdir()
        for i in range(16):
            raw, row = run(i, np.zeros((2, 6)), folder / f"case{i}.npz")
            if bootstrap or i % 2 == 0:
                old = (
                    Path(second_references[i]["path"])
                    if bootstrap
                    else a.handoff_root / f"world{i}-repeat0/skill1.npz"
                )
                with np.load(old, allow_pickle=False) as z:
                    if any(not np.array_equal(raw[k], z[k]) for k in z.files):
                        raise ValueError("cloned branch must match actual continuous second skill")
            baseline.append(row)
        for name in names:
            contacts = [
                r["postcontact"]["first_foot_contact_substep"]
                for r in baseline
                if r["source"] == name
            ]
            valid = [x // 10 for x in contacts if x is not None]
            centers[name] = sum(valid) // len(valid) if valid else 120
        write_once(
            folder / "complete.json",
            dict(
                results=baseline,
                centers=centers,
                all_static_full_continuous_second_skills_reproduced=True,
            ),
        )

        def search(si):
            nonlocal cem_updates
            name = names[si]
            ids = [i for i, c in enumerate(cases) if c["source"] == name]
            rng = np.random.default_rng(20261022 + si)
            best_z, mean, std = np.zeros((2, 6)), np.zeros((2, 6)), np.full((2, 6), 0.5)
            best_rows = [baseline[i] for i in ids]
            best_rank = contact_rank(best_rows)
            folder = a.output / f"source{si}"
            folder.mkdir()
            for generation in range(3):
                target = folder / f"generation{generation}"
                target.mkdir()
                zs = [best_z.copy()] + list(rng.normal(mean, std, size=(11, 2, 6)))
                candidates = []
                for ci, z in enumerate(zs):
                    rows = [
                        run(i, np.tanh(z), target / f"candidate{ci}-case{k}.npz")[1]
                        for k, i in enumerate(ids)
                    ]
                    rank = contact_rank(rows)
                    candidates.append(
                        dict(
                            z=z.tolist(), latent=np.tanh(z).tolist(), rank=list(rank), results=rows
                        )
                    )
                    if rank > best_rank:
                        best_z, best_rows, best_rank = z.copy(), rows, rank
                elite_ids = sorted(
                    range(12), key=lambda j: tuple(candidates[j]["rank"]), reverse=True
                )[:4]
                elite = np.stack([zs[j] for j in elite_ids])
                mean = 0.5 * mean + 0.5 * elite.mean(axis=0)
                std = np.clip(0.5 * std + 0.5 * elite.std(axis=0), 0.15, 1.2)
                write_once(
                    target / "complete.json",
                    dict(
                        candidates=candidates,
                        best_rank=list(best_rank),
                        next_z_mean=mean.tolist(),
                        next_z_std=std.tolist(),
                    ),
                )
                with bench.lock:
                    cem_updates += 1
                print(
                    json.dumps(dict(source=name, generation=generation, rank=list(best_rank))),
                    flush=True,
                )
            result = dict(
                source=name,
                selected_z=best_z.tolist(),
                selected_latent=np.tanh(best_z).tolist(),
                rank=list(best_rank),
                results=best_rows,
            )
            write_once(folder / "selected.json", result)
            return result

        with ThreadPoolExecutor(max_workers=8) as pool:
            selected = list(pool.map(search, source_indices))
        final = []
        for i in final_indices:
            outputs = []
            latent = np.array(
                next(s["selected_latent"] for s in selected if s["source"] == cases[i]["source"])
            )
            for repeat in (0, 1):
                folder = a.output / f"final{i}-repeat{repeat}"
                ep = parent(i, folder)
                ep.begin_current_state(clips[cases[i]["source"]], goal=ep.goal.copy())
                raw, row = run(i, latent, folder / "second.npz", ep=ep)
                outputs.append((raw, row))
            if any(not np.array_equal(outputs[0][0][k], outputs[1][0][k]) for k in outputs[0][0]):
                raise ValueError("full selected first+drop+second independent repeats differ")
            choice = next(s for s in selected if s["source"] == cases[i]["source"])
            chosen = choice["results"][i % 2]
            with np.load(a.output / chosen["raw_path"], allow_pickle=False) as z:
                if any(not np.array_equal(outputs[0][0][k], z[k]) for k in z.files):
                    raise ValueError(
                        "fresh full sequence differs from selected cloned-parent branch"
                    )
            row = outputs[0][1]
            row["independent_repeat_raw_hash"] = outputs[1][1]["raw_hash"]
            final.append(row)
        check_pins(pins)
        artifact_hashes = {
            str(f.relative_to(a.output)): digest(f) for f in a.output.rglob("*") if f.is_file()
        }
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_LIVE_BODY_CONTACT_TEACHER_RESEARCH_NOT_PROMOTED",
                input_file_hashes=pins,
                artifact_hashes=artifact_hashes,
                baseline=baseline,
                selected=selected,
                final=final,
                baseline_quality=sum(contact_rank([x, x])[1] > 0 for x in baseline),
                final_quality=sum(contact_rank([x, x])[1] > 0 for x in final),
                stable=sum(not x["body_failed"] for x in final),
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                cem_distribution_updates=cem_updates,
                neural_optimizer_updates=0,
                all_first_kick_baselines_exact=True,
                first_kick_reference="learned_safe_successor" if bootstrap else "original_v0",
                search_source_indices=source_indices,
                final_case_indices=final_indices,
                all_static_live_baselines_exact=True,
                all_selected_whole_sequences_repeated_exactly=True,
                goal_complete=False,
                fresh_exam=False,
                team_game=False,
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                actual_native_steps=0 if bench is None else bench.native_steps,
                actual_neural_rows=0 if bench is None else bench.neural_rows,
                cem_distribution_updates=cem_updates,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
