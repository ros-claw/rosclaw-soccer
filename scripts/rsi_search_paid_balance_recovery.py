"""Bounded SIM teacher pilot: actively stabilize actual post-kick bodies.

NEW explicit 100-control recovery phase, not a change to any old exam. Search
copies full live mjData; final selected worlds repeat from time zero. Only two
declared failure states (0,15) are studied. No general skill promotion follows.
"""

import argparse
import copy
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain-root", type=Path, required=True)
    p.add_argument("--chain-hash", required=True)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--seed-index", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean pinned search code required")
    from rosclaw_soccer.rsi.paid_recovery_proof import recovery_proof, unique_proof_path

    declaration = json.loads((a.chain_root / "declaration.json").read_text())
    cases = declaration["cases"]
    d, pins = recovery_proof(a.chain_root, a.chain_hash, a.audit_root, a.audit_hash, cases=cases)
    source = json.loads((a.chain_root / "complete.json").read_text())
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_cases=cases,
            pilot_cases=[0, 15],
            seed=20261112 + a.seed_index,
            generations=4,
            population=24,
            elite=6,
            latent_shape=[9, 4],
            gain_scale=2.0,
            new_phase_controls=100,
            new_phase_is_not_old_exam=True,
            max_delta_rad=0.2,
            max_delta_per_control_rad=0.02,
            preserve_original_skill_scores=True,
            body_resets=0,
            branch_semantics=(
                "full mj_copyData from actual two-skill parent after declared ball return"
            ),
            selected_full_world_repeats=2,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    bench = None
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_balance_recovery import collect_balance_recovery
        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_native_episode import NativeStepCounter
        from rosclaw_soccer.rsi.paid_practice_chain import (
            advance_practice_boundary,
            require_identical_snapshot,
        )
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_skill_handoff import practice_ball_drop, snapshot

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, cases)
        actor, _ = load_contact_residual_actor(
            policy,
            checkpoint=unique_proof_path(pins, d["checkpoint_hash"]),
            checkpoint_hash=d["checkpoint_hash"],
            training_receipt=unique_proof_path(pins, d["training_receipt_hash"]),
            training_receipt_hash=d["training_receipt_hash"],
            scene_hash=scene,
        )

        def save(folder, key, raw):
            path = folder / (key + ".npz")
            if path.exists():
                raise ValueError("physical artifact overwrite forbidden")
            np.savez_compressed(path, **raw)
            return dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))

        def read_parent(row):
            path = a.chain_root / row["raw_path"]
            if (
                not path.resolve().is_relative_to(a.chain_root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("confined qualified parent trajectory required")
            pins[str(path)] = row["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def same(x, y):
            if set(x) != set(y) or any(not np.array_equal(x[k], y[k]) for k in x):
                raise ValueError("actual qualified whole-world history changed")

        parent_records = []

        def parent(i, folder):
            folder.mkdir(parents=True)
            ep = bench.make(cases[i])
            reference = source["results"]["learned"]["worlds"][i]
            rows, artifacts = [], []

            def store(key, raw):
                artifact = save(folder, key, raw)
                artifacts.append(artifact)
                return artifact

            for k, model in enumerate((actor.base, actor)):
                if k:
                    boundary = advance_practice_boundary(
                        ep,
                        ep.clip,
                        foot="left" if "left" in cases[i]["source"] else "right",
                        store=store,
                        prefix="between0-1",
                    )
                    if boundary["status"] != "READY":
                        raise ValueError("qualified second skill boundary did not reproduce")
                raw, row = bench.collect_current(ep, model)
                same(raw, read_parent(reference["skills"][k]["result"]))
                rows.append(dict(row, **store(f"skill{k}", raw)))
                store(f"terminal{k}", snapshot(ep.env.data))
            drop = practice_ball_drop(ep, foot="left" if "left" in cases[i]["source"] else "right")
            expected = next(
                x for x in reference["artifacts"] if Path(x["raw_path"]).stem == "between1-2-drop"
            )
            same(drop, read_parent(expected))
            store("between1-2-drop", drop)
            store("recovery-entry", snapshot(ep.env.data))
            record = dict(case=i, skills=rows, artifacts=artifacts)
            write_once(folder / "complete.json", record)
            parent_records.append(record)
            return ep

        parents = {i: parent(i, a.output / f"parent{i}") for i in (0, 15)}

        def branch(i):
            ep = copy.copy(parents[i])
            ep.env = copy.copy(parents[i].env)
            ep.env.data = mujoco.MjData(ep.env.model)
            mujoco.mj_copyData(ep.env.data, ep.env.model, parents[i].env.data)
            require_identical_snapshot(snapshot(parents[i].env.data), snapshot(ep.env.data))
            ep.target, ep.last_action = parents[i].target.copy(), parents[i].last_action.copy()
            ep.work = NativeStepCounter()
            bench.counters.append(ep.work)
            return ep

        def run(ep, latent, folder, key):
            initial = snapshot(ep.env.data)
            raw, row = collect_balance_recovery(ep, latent)
            raw.update({"initial_" + k: v for k, v in initial.items()})
            return raw, dict(row, **save(folder, key, raw))

        def rank(rows):
            return (
                sum(r["healthy"] for r in rows),
                sum(r["recovery_stability_observed"] for r in rows),
                min(r["controls"] for r in rows),
                -sum(r["cost"] for r in rows),
            )

        folder = a.output / "baseline"
        folder.mkdir()
        baseline = [run(branch(i), np.zeros((9, 4)), folder, f"case{i}")[1] for i in (0, 15)]
        for i, row in zip((0, 15), baseline, strict=True):
            world = source["results"]["learned"]["worlds"][i]
            hold = next(
                (x for x in world["artifacts"] if Path(x["raw_path"]).stem == "between1-2-hold"),
                None,
            )
            if hold is not None:
                expected = read_parent(hold)
                with np.load(a.output / row["raw_path"], allow_pickle=False) as actual:
                    for k, value in expected.items():
                        if k != "post_reference_hold" and not np.array_equal(
                            actual[k][: len(value)], value
                        ):
                            raise ValueError(
                                "zero recovery does not reproduce original hold prefix"
                            )
        write_once(folder / "complete.json", dict(results=baseline, zero_hold_prefix_exact=True))
        rng = np.random.default_rng(20261112 + a.seed_index)
        best_z = mean = np.zeros((9, 4))
        std = np.full((9, 4), 0.5)
        best_rows, best_rank = baseline, rank(baseline)
        for generation in range(4):
            folder = a.output / f"generation{generation}"
            folder.mkdir()
            zs = [best_z.copy()] + list(rng.normal(mean, std, size=(23, 9, 4)))
            candidates = []
            for j, z in enumerate(zs):
                rows = [
                    run(branch(i), np.tanh(z), folder, f"candidate{j}-case{i}")[1] for i in (0, 15)
                ]
                score = rank(rows)
                candidates.append(
                    dict(z=z.tolist(), latent=np.tanh(z).tolist(), rank=list(score), results=rows)
                )
                if score > best_rank:
                    best_z, best_rows, best_rank = z.copy(), rows, score
            elite = np.stack(
                [
                    zs[j]
                    for j in sorted(
                        range(24), key=lambda j: tuple(candidates[j]["rank"]), reverse=True
                    )[:6]
                ]
            )
            mean = 0.5 * mean + 0.5 * elite.mean(axis=0)
            std = np.clip(0.5 * std + 0.5 * elite.std(axis=0), 0.15, 1.2)
            write_once(
                folder / "complete.json",
                dict(
                    candidates=candidates,
                    next_mean=mean.tolist(),
                    next_std=std.tolist(),
                    best_rank=list(best_rank),
                ),
            )
            print(json.dumps(dict(generation=generation, best_rank=best_rank)), flush=True)
        final = []
        for i, selected in zip((0, 15), best_rows, strict=True):
            repeats = []
            for repeat in range(2):
                folder = a.output / f"final{i}-repeat{repeat}"
                ep = parent(i, folder)
                raw, row = run(ep, np.tanh(best_z), folder, "recovery")
                with np.load(a.output / selected["raw_path"], allow_pickle=False) as old:
                    same(raw, {k: old[k] for k in old.files})
                third = None
                if row["healthy"]:
                    q, v = ep.layout.canonical(ep.env.data)
                    if q[38] <= 0.12 and abs(v[37]) <= 0.1:
                        before = snapshot(ep.env.data)
                        previous = ep.last_action.copy()
                        ep.begin_current_state(ep.clip, goal=ep.goal.copy())
                        require_identical_snapshot(before, snapshot(ep.env.data))
                        if not np.array_equal(ep.last_action, previous):
                            raise ValueError("recovery handoff erased the actual teacher action")
                        save(
                            folder,
                            "third-boundary",
                            {
                                **{"before_" + k: v for k, v in before.items()},
                                "previous_actual_action": previous,
                            },
                        )
                        z, third = bench.collect_current(ep, actor)
                        third.update(save(folder, "third", z))
                repeats.append(dict(case=i, recovery=row, third=third))
                write_once(folder / "final.json", repeats[-1])
            if (repeats[0]["third"] is None) != (repeats[1]["third"] is None):
                raise ValueError("selected continuation branch does not repeat")
            if repeats[0]["third"] is not None:
                raws = []
                for row in repeats:
                    with np.load(a.output / row["third"]["raw_path"], allow_pickle=False) as z:
                        raws.append({k: z[k] for k in z.files})
                same(*raws)
            final.append(repeats)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_ACTIVE_BALANCE_TEACHER_PILOT_CLOSED_PENDING_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                pilot_cases=[0, 15],
                parents=parent_records,
                baseline=baseline,
                selected_latent=np.tanh(best_z).tolist(),
                selected_rank=list(best_rank),
                selected_results=best_rows,
                final=final,
                all_selected_recovery_and_third_repeats_exact=True,
                CEM_distribution_updates=4,
                neural_optimizer_updates=0,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                artifact_hashes={
                    str(f.relative_to(a.output)): digest(f)
                    for f in a.output.rglob("*")
                    if f.is_file()
                },
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
                new_recovery_phase_not_old_exam=True,
                team_game=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                native_steps=0 if bench is None else bench.native_steps,
                neural_rows=0 if bench is None else bench.neural_rows,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
