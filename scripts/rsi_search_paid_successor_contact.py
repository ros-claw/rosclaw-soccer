"""Bounded TRAIN successor-contact teacher, with neural standing retained.

Two predeclared source-pairs diagnose knee-ball rebound and lateral-error cases.
Each common two-bump Cartesian intent is searched across both actual successor
entries. This is a privileged composite teacher, NOT parameter learning or a
continuous game. DEV is not searched; original source/score remain unchanged.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "audit-root", "replay-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("hash", "audit-hash", "replay-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or digest(a.root / "complete.json") != a.hash
        or digest(a.audit_root / "complete.json") != a.audit_hash
        or digest(a.replay_root / "complete.json") != a.replay_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen exact TRAIN proof/source required")
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate

    standing_gate(a.root, a.audit_root, variant=1, bank="train")
    source, declaration, review, replay = [
        json.loads((root / name).read_text())
        for root, name in (
            (a.root, "complete.json"),
            (a.root, "declaration.json"),
            (a.audit_root, "complete.json"),
            (a.replay_root, "complete.json"),
        )
    ]
    if (
        replay["status"] != "PAID_POST_FIRST_ALL_PHASES_STRICT_NATIVE_REPLAY_PASS"
        or replay["input_file_hashes"].get(str(a.root / "complete.json")) != a.hash
        or replay["input_file_hashes"].get(str(a.audit_root / "complete.json")) != a.audit_hash
        or replay["all_q_v_forces_terminal_exact"] is not True
        or len(source["worlds"]) != 16
        or source["summary"]["stable"] != 16
    ):
        raise ValueError("all16 exact post-first physics replay required before teacher search")
    indices = [[2, 3], [10, 11]]
    cases = declaration["cases"]
    if any(cases[i]["source"] != cases[j]["source"] for i, j in indices):
        raise ValueError("two declared same-source TRAIN pairs required")
    pins = dict(review["input_file_hashes"])
    pins.update(
        {
            str(a.root / "complete.json"): a.hash,
            str(a.audit_root / "complete.json"): a.audit_hash,
            str(a.replay_root / "complete.json"): a.replay_hash,
        }
    )
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            pairs=indices,
            source_cases=cases,
            random_seed=20261022 + a.seed,
            generations=2,
            population=12,
            elite_count=4,
            latent_shape=[2, 6],
            distribution_initial_std=0.7,
            retains_best_slot0=True,
            common_intent_for_both_conditions=True,
            actual_training_state_restore=True,
            uninterrupted_game=False,
            standing_controls=100,
            zero_baseline_must_match_source_third_arrays=True,
            no_DEV_search=True,
            optimizer_updates=0,
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    bench = None
    updates = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_continuity_audit import identical, snapshot_fields
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.paid_recovery_proof import unique_proof_path
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
        from rosclaw_soccer.rsi.paid_successor_branch import begin_successor_branch
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, cases)
        actor, _ = load_contact_residual_actor(
            policy,
            checkpoint=unique_proof_path(pins, declaration["actor_checkpoint_hash"]),
            checkpoint_hash=declaration["actor_checkpoint_hash"],
            training_receipt=unique_proof_path(pins, declaration["actor_training_receipt_hash"]),
            training_receipt_hash=declaration["actor_training_receipt_hash"],
            scene_hash=scene,
        )

        def read(row):
            path = a.root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(a.root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("confined exact source physical branch required")
            pins[str(path)] = row["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        branches = {}
        centers = {}
        for pair in indices:
            for i in pair:
                world = source["worlds"][i]
                boundary = read(world["boundaries"][1]["boundary"])
                third = read(world["kicks"][2]["result"])
                branches[i] = (
                    snapshot_fields(boundary, "after_"),
                    boundary["previous_actual_action"],
                    third,
                )
            centers[pair[0]] = (
                sum(
                    source["worlds"][i]["kicks"][2]["result"]["postcontact"][
                        "first_foot_contact_substep"
                    ]
                    // 10
                    for i in pair
                )
                // 2
            )

        def inference():
            with bench.lock:
                bench.neural_rows += 1

        def store(path, raw):
            if path.exists():
                raise ValueError("no trajectory overwrites")
            np.savez_compressed(path, **raw)
            return dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))

        def run(i, latent, folder, *, pure=False):
            entry, previous, reference = branches[i]
            ep = begin_successor_branch(bench, cases[i], entry, previous)
            transform = (
                None
                if pure
                else FootTaskSpaceIntent(
                    ep.env.model,
                    foot="left" if "left" in cases[i]["source"] else "right",
                    center=centers[next(pair[0] for pair in indices if i in pair)],
                    latent=latent,
                )
            )
            raw, row = bench.collect_current(
                ep, actor.base, transform=transform, stop_at_reference=True
            )
            if pure or not latent.any():
                identical({k: v for k, v in raw.items() if not k.startswith("PLAN_")}, reference)
            result = dict(
                case=i,
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
                result["stand"] = dict(ss, **store(folder / f"case{i}-stand.npz", sr))
            return result

        def score(rows):
            return (
                int(all(r["stand"] and r["stand"]["healthy"] for r in rows)),
                int(all(r["stand"] and r["stand"]["stability_observed"] for r in rows)),
                *contact_rank([r["kick"] for r in rows]),
            )

        rng = np.random.default_rng(20261022 + a.seed)
        searches = []
        for pair in indices:
            folder = a.output / f"pair{pair[0]}"
            folder.mkdir()
            zero = np.zeros((2, 6))
            baseline_dir = folder / "baseline"
            baseline_dir.mkdir()
            baselines = [run(i, zero, baseline_dir, pure=True) for i in pair]
            best, best_rank = zero.copy(), score(baselines)
            mean, std = np.zeros((2, 6)), np.ones((2, 6)) * 0.7
            history = []
            for generation in range(2):
                sampled = rng.normal(mean, std, size=(12, 2, 6))
                sampled[0] = np.arctanh(np.clip(best, -0.999999, 0.999999))
                candidates = np.tanh(sampled)
                candidates[0] = best
                records = []
                for slot, latent in enumerate(candidates):
                    path = folder / f"generation{generation}-slot{slot}"
                    path.mkdir()
                    rows = [run(i, latent, path) for i in pair]
                    rank = score(rows)
                    record = dict(slot=slot, latent=latent.tolist(), rank=list(rank), rows=rows)
                    write_once(path / "complete.json", record)
                    records.append(record)
                    if rank > best_rank:
                        best, best_rank = latent.copy(), rank
                elite = sorted(range(12), key=lambda k: tuple(records[k]["rank"]), reverse=True)[:4]
                mean, std = sampled[elite].mean(axis=0), np.maximum(sampled[elite].std(axis=0), 0.1)
                updates += 1
                event = dict(
                    generation=generation,
                    candidates=records,
                    elite=elite,
                    next_mean=mean.tolist(),
                    next_std=std.tolist(),
                )
                write_once(folder / f"generation{generation}.json", event)
                history.append(event)
                print(
                    json.dumps(dict(pair=pair, generation=generation, best_rank=list(best_rank))),
                    flush=True,
                )
            selected = folder / "selected"
            selected.mkdir()
            final = [run(i, best, selected) for i in pair]
            if score(final) != best_rank:
                raise ValueError("selected common teacher failed to reproduce its rank")
            searches.append(
                dict(
                    pair=pair,
                    center=centers[pair[0]],
                    baseline=baselines,
                    baseline_rank=list(score(baselines)),
                    latent=best.tolist(),
                    selected=final,
                    selected_rank=list(best_rank),
                    history=history,
                )
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_SUCCESSOR_CONTACT_TEACHER_CLOSED_PENDING_AUDIT",
                input_file_hashes=pins,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                searches=searches,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                cem_distribution_updates=updates,
                optimizer_updates=0,
                train_consumed=True,
                uninterrupted_game=False,
                old_exam_result=False,
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
                actual_native_steps=bench.native_steps if bench is not None else 0,
                actual_neural_rows=bench.neural_rows if bench is not None else 0,
                cem_distribution_updates=updates,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
