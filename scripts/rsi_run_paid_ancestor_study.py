"""One finite five-case TRAIN shard: three motor schedules, two whole repeats."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("plan-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("plan-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--shard", type=int, choices=range(4), required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused shard output and frozen source required")
    from rosclaw_soccer.rsi.paid_ancestor_study import (
        SCHEDULES,
        ancestor_comparison,
        load_ancestor_actor,
        load_ancestor_plan,
    )

    r, d = load_ancestor_plan(a.plan_root, a.plan_hash)
    pins = dict(r["input_file_hashes"])
    for n in ("complete.json", "declaration.json"):
        f = a.plan_root / n
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    cases = d["cases"][a.shard * 5 : (a.shard + 1) * 5]
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            plan_root=str(a.plan_root),
            plan_hash=a.plan_hash,
            shard=a.shard,
            cases=cases,
            schedules=d["schedules"],
            repeats=2,
            bank="train",
            Fresh_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_lifecycle_bundle import require_same_prefix
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_standing_lifecycle import run_standing_lifecycle
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # The complete20-case declaration is preflighted before ANY shard
        # rollout. A shard is execution scheduling, not a sampled easier bank.
        bench, policy, scene = open_paid_bench(pins, d["cases"])
        if scene != d["body_scene_hash"] or bench.native_steps or bench.neural_rows:
            raise ValueError("unchanged zero-step full-bank admission required")
        actor = load_ancestor_actor(policy, d)
        for mode in SCHEDULES:
            (a.output / mode).mkdir()
        groups = []
        for case in cases:
            i = case["case"]
            group = dict(case=i, arms={})
            first_raw = {}
            for mode, schedule in SCHEDULES.items():
                root = a.output / mode
                replicas, raws = [], []
                for repeat in range(2):
                    folder = root / f"case{i}-repeat{repeat}"
                    folder.mkdir()
                    artifacts = []
                    raw = {}

                    def store(key, z, folder=folder, root=root, artifacts=artifacts, raw=raw):
                        f = folder / (key + ".npz")
                        if f.exists():
                            raise ValueError("raw overwrite forbidden")
                        np.savez_compressed(f, **z)
                        raw[key] = z
                        link = dict(raw_path=str(f.relative_to(root)), raw_hash=digest(f))
                        artifacts.append(link)
                        return link

                    w = run_standing_lifecycle(
                        bench,
                        case,
                        actor,
                        variant=2,
                        skills=3,
                        store=store,
                        record_initial=True,
                        motor_schedule=schedule,
                    )
                    w.update(
                        case=i,
                        artifacts=artifacts,
                        quality_by_kick=[
                            bool(contact_rank([k["result"], k["result"]])[1] > 0)
                            for k in w["kicks"]
                        ],
                    )
                    write_once(folder / "complete.json", w)
                    replicas.append(w)
                    raws.append(raw)
                if set(raws[0]) != set(raws[1]):
                    raise ValueError("repeat coverage differs")
                for key in raws[0]:
                    identical(raws[0][key], raws[1][key])
                group["arms"][mode] = dict(replicas[0], repeat_artifacts=replicas[1]["artifacts"])
                first_raw[mode] = raws[0]
            require_same_prefix(first_raw["grown"], first_raw["ancestor_third"], before_kick=2)
            require_same_prefix(first_raw["grown"], first_raw["r2_all"], before_kick=1)
            group["shared_actual_prefix_exact"] = True
            groups.append(group)
            print(
                json.dumps(
                    dict(
                        case=i, quality={k: v["quality_by_kick"] for k, v in group["arms"].items()}
                    )
                ),
                flush=True,
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_TRAIN_ANCESTOR_COUNTERFACTUAL_SHARD_PENDING_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                plan_hash=a.plan_hash,
                shard=a.shard,
                groups=groups,
                comparison=ancestor_comparison(groups),
                all_repeats_and_shared_prefixes_exact=True,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                bank="train",
                Fresh_fit_rows=0,
                optimizer_updates=0,
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
                actual_native_steps=bench.native_steps if bench else 0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
