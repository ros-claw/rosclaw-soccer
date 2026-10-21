"""Finite TRAIN-only first-contact foot teacher; all trials retained, no Fresh access."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("qualification-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("qualification-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    a = p.parse_args()
    rp = a.qualification_root / "complete.json"
    if (
        a.output.exists()
        or digest(rp) != a.qualification_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output, exact qualification and frozen source required")
    from rosclaw_soccer.rsi.paid_ancestor_study import load_ancestor_plan
    from rosclaw_soccer.rsi.paid_first_teacher import (
        FIRST_PAIRS,
        first_teacher_config,
        first_teacher_rank,
    )

    review = json.loads(rp.read_text())
    qd = json.loads((a.qualification_root / "declaration.json").read_text())
    if review["status"] != "ROUTER_FRONTIER_PASS_PENDING_NEW_FRESH" or review["primary_seed"] != 0:
        raise ValueError("fixed completed current-parent frontier required")
    pins = dict(review["input_file_hashes"])
    pins[str(rp)] = a.qualification_hash
    references = []
    cases = None
    for index, name in enumerate(qd["reference_shards"]):
        root = Path(name)
        f = root / "complete.json"
        if pins.get(str(f)) != digest(f):
            raise ValueError("qualified baseline shard changed")
        r = json.loads(f.read_text())
        d = json.loads((root / "declaration.json").read_text())
        _, plan = load_ancestor_plan(Path(d["plan_root"]), d["plan_hash"])
        if r["shard"] != index or (cases is not None and cases != plan["cases"]):
            raise ValueError("complete ordered20 TRAIN contexts required")
        cases = plan["cases"]
        references.extend((root / "grown", g["arms"]["grown"]) for g in r["groups"])
    if len(references) != 20 or cases is None:
        raise ValueError("complete TRAIN bank required")
    if (
        cases[4]["source"] != cases[5]["source"]
        or "003_left" not in cases[4]["source"]
        or cases[14]["source"] != cases[15]["source"]
        or "009_right" not in cases[14]["source"]
    ):
        raise ValueError("predeclared knee/lateral and knee/retention source pairs required")
    student_root = Path(qd["pipelines"][0]) / "trained"
    training = json.loads((student_root / "complete.json").read_text())
    if pins.get(str(student_root / "complete.json")) != digest(student_root / "complete.json"):
        raise ValueError("frozen primary actor lineage required")
    config = first_teacher_config(a.seed)
    centers = {
        str(pair[0]): sum(
            references[i][1]["kicks"][0]["result"]["postcontact"]["first_foot_contact_substep"]
            // 10
            for i in pair
        )
        // 2
        for pair in FIRST_PAIRS
    }
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            config=config,
            centers=centers,
            source_cases=cases,
            student_root=str(student_root),
            student_hash=digest(student_root / "complete.json"),
            qualification_root=str(a.qualification_root),
            qualification_hash=a.qualification_hash,
            teacher_policy="frozen_R2+bounded_scratch_only_foot_IK",
            bank="train",
            Fresh_rows_read=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    episodes = updates = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, cases)
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=student_root / "complete.json",
            training_receipt_hash=digest(student_root / "complete.json"),
            scene_hash=scene,
        )

        def read(root, row):
            f = root / row["raw_path"]
            if (
                f.is_symlink()
                or not f.resolve().is_relative_to(root.resolve())
                or digest(f) != row["raw_hash"]
            ):
                raise ValueError("exact confined baseline raw required")
            pins[str(f)] = row["raw_hash"]
            with np.load(f, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def store(path, z):
            if path.exists():
                raise ValueError("raw overwrite forbidden")
            np.savez_compressed(path, **z)
            return dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))

        def inference():
            with bench.lock:
                bench.neural_rows += 1

        def run(i, latent, folder, center, *, pure=False):
            nonlocal episodes
            ep = bench.make(cases[i])
            root, world = references[i]
            entry = snapshot(ep.env.data)
            identical(entry, read(root, world["initial"]))
            intent = (
                None
                if pure
                else FootTaskSpaceIntent(
                    ep.env.model,
                    foot="left" if "left" in cases[i]["source"] else "right",
                    center=center,
                    latent=latent,
                )
            )
            raw, row = bench.collect_current(
                ep, actor.base, transform=intent, stop_at_reference=True
            )
            if pure or not latent.any():
                identical(
                    {k: v for k, v in raw.items() if not k.startswith("PLAN_")},
                    read(root, world["kicks"][0]["result"]),
                )
            record = dict(
                case=i,
                initial=store(folder / f"case{i}-initial.npz", entry),
                kick=dict(row, **store(folder / f"case{i}-kick.npz", raw)),
                kick_terminal=store(folder / f"case{i}-kick-terminal.npz", snapshot(ep.env.data)),
                stand=None,
            )
            if not row["body_failed"]:
                task = NeuralStandingTask(
                    ep,
                    actor.base,
                    state=(
                        torch.from_numpy(raw["NN_h_after"][-1].copy()),
                        torch.from_numpy(raw["NN_c_after"][-1].copy()),
                    ),
                    memory_mode="carry",
                    count_inference=inference,
                )
                sr, ss = task.collect(100)
                record["stand"] = dict(ss, **store(folder / f"case{i}-stand.npz", sr))
                if pure or not latent.any():
                    identical(sr, read(root, world["stands"][0]["result"]))
            episodes += 1
            if episodes > config["max_episodes"] or bench.native_steps > config["max_native_steps"]:
                raise ValueError("bounded first-contact pilot budget exceeded")
            return record

        rng = np.random.default_rng(config["random_seed"])
        searches = []
        for pair in FIRST_PAIRS:
            center = centers[str(pair[0])]
            protected = [references[i][1]["quality_by_kick"][0] for i in pair]
            folder = a.output / f"pair{pair[0]}"
            folder.mkdir()
            sub = folder / "baseline"
            sub.mkdir()
            zero = np.zeros((2, 6))
            baseline = [run(i, zero, sub, center, pure=True) for i in pair]
            rank = first_teacher_rank(baseline, protected)
            best = zero.copy()
            mean, std = zero.copy(), np.ones((2, 6)) * 0.7
            history = []
            for generation in range(2):
                sampled = rng.normal(mean, std, size=(12, 2, 6))
                sampled[0] = np.arctanh(np.clip(best, -0.999999, 0.999999))
                proposals = np.tanh(sampled)
                proposals[0] = best
                records = []
                for slot, latent in enumerate(proposals):
                    sub = folder / f"generation{generation}-slot{slot}"
                    sub.mkdir()
                    rows = [run(i, latent, sub, center) for i in pair]
                    score = first_teacher_rank(rows, protected)
                    record = dict(slot=slot, latent=latent.tolist(), rank=list(score), rows=rows)
                    write_once(sub / "complete.json", record)
                    records.append(record)
                    if score > rank:
                        best, rank = latent.copy(), score
                elite = sorted(range(12), key=lambda k: tuple(records[k]["rank"]), reverse=True)[:4]
                mean, std = sampled[elite].mean(axis=0), np.maximum(sampled[elite].std(axis=0), 0.1)
                event = dict(
                    generation=generation,
                    candidates=records,
                    elite=elite,
                    next_mean=mean.tolist(),
                    next_std=std.tolist(),
                )
                write_once(folder / f"generation{generation}.json", event)
                history.append(event)
                updates += 1
                print(
                    json.dumps(dict(pair=pair, generation=generation, best_rank=list(rank))),
                    flush=True,
                )
            selected = []
            for repeat in range(2):
                sub = folder / f"selected-repeat{repeat}"
                sub.mkdir()
                rows = [run(i, best, sub, center) for i in pair]
                if first_teacher_rank(rows, protected) != rank:
                    raise ValueError("selected teacher rank failed to reproduce")
                selected.append(rows)
            searches.append(
                dict(
                    pair=list(pair),
                    center=center,
                    protected=protected,
                    baseline=baseline,
                    baseline_rank=list(first_teacher_rank(baseline, protected)),
                    selected=selected,
                    selected_rank=list(rank),
                    latent=best.tolist(),
                    history=history,
                )
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_TRAIN_FIRST_CONTACT_TEACHER_CLOSED_PENDING_OWN_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                config=config,
                searches=searches,
                episodes=episodes,
                cem_distribution_updates=updates,
                optimizer_updates=0,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                Fresh_rows_read=0,
                bank="train",
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                episodes=episodes,
                cem_distribution_updates=updates,
                actual_native_steps=bench.native_steps if bench else 0,
                actual_neural_rows=bench.neural_rows if bench else 0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
