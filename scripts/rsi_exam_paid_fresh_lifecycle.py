"""Execute one fixed Fresh arm: all32 contexts, two continuous full-world repeats."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--declaration-root", type=Path, required=True)
    p.add_argument("--declaration-hash", required=True)
    p.add_argument("--arm", choices=("r2", "parent", "progressive"), required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen clean Fresh runner required")
    from rosclaw_soccer.rsi.paid_fresh_evidence import load_fresh_actor, load_fresh_declaration
    from rosclaw_soccer.rsi.paid_fresh_protocol import fresh_summary

    preflight, d = load_fresh_declaration(a.declaration_root, a.declaration_hash)
    pins = dict(preflight["input_file_hashes"])
    for name in ("complete.json", "declaration.json"):
        f = a.declaration_root / name
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            fresh_root=str(a.declaration_root),
            fresh_hash=a.declaration_hash,
            arm=a.arm,
            cases=d["cases"],
            repeats=2,
            skills=3,
            record_initial_snapshot_fields=10,
            Fresh_episodes_fit=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_standing_lifecycle import run_standing_lifecycle
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, d["cases"])
        if scene != d["unchanged_body_scene_hash"] or bench.native_steps or bench.neural_rows:
            raise ValueError("all-case admission must precede any Fresh physics")
        actor, variant = load_fresh_actor(policy, d, a.arm)
        worlds = []
        for i, case in enumerate(d["cases"]):
            replicas, raw_replicas = [], []
            for repeat in range(2):
                folder = a.output / f"case{i}-repeat{repeat}"
                folder.mkdir()
                artifacts, raws = [], {}

                def store(key, raw, folder=folder, artifacts=artifacts, raws=raws):
                    path = folder / (key + ".npz")
                    if path.exists():
                        raise ValueError("raw overwrite forbidden")
                    np.savez_compressed(path, **raw)
                    link = dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))
                    artifacts.append(link)
                    raws[key] = raw
                    return link

                w = run_standing_lifecycle(
                    bench, case, actor, variant=variant, skills=3, store=store, record_initial=True
                )
                w.update(
                    case=i,
                    artifacts=artifacts,
                    quality_by_kick=[
                        bool(contact_rank([k["result"], k["result"]])[1] > 0) for k in w["kicks"]
                    ],
                )
                write_once(folder / "complete.json", w)
                replicas.append(w)
                raw_replicas.append(raws)
            if set(raw_replicas[0]) != set(raw_replicas[1]):
                raise ValueError("whole-world repeat coverage differs")
            for key in raw_replicas[0]:
                identical(raw_replicas[0][key], raw_replicas[1][key])
            worlds.append(dict(replicas[0], repeat_artifacts=replicas[1]["artifacts"]))
            print(
                json.dumps(dict(case=i, status=w["status"], quality=w["quality_by_kick"])),
                flush=True,
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_FRESH32_LIFECYCLE_ALL_WORLDS_CLOSED_PENDING_INDEPENDENT_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                fresh_hash=a.declaration_hash,
                arm=a.arm,
                variant=variant,
                worlds=worlds,
                summary=fresh_summary(worlds, d["cases"]),
                all_whole_world_repeats_exact=True,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                Fresh_episodes_fit=0,
                optimizer_updates=0,
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
                actual_native_steps=bench.native_steps if bench else 0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
