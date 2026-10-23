"""Close ALL4 sealed Fresh context shards using their independent whole proofs.

Publishes aggregate metrics only. No course coordinates, seed, per-case rows,
model training, sampling, threshold change or automatic policy promotion.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("seal-root", "shards-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("seal-hash", "producer-head", "auditor-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_pass_fresh_common import (
        OWN_STATUS,
        SOURCE_STATUS,
        admit_pass_seal,
        require_frozen_pass_protocol,
    )

    seal, pins, _, _ = admit_pass_seal(a.seal_root, a.seal_hash)
    if a.output.exists():
        raise ValueError("unused Fresh whole-study readback output required")
    seeds = [str(m["seed"]) for m in seal["models"]]
    records = {s: [] for s in seeds}
    proofs = []
    counts = dict(native_steps=0, actor_calls=0)
    for ctx in (1, 4, 20, 21):
        roots = [a.shards_root / f"context{ctx}" / stage for stage in ("source", "own")]
        paths = [r / "complete.json" for r in roots]
        if any(r.is_symlink() or (r / "failed.json").exists() for r in roots) or any(
            p.is_symlink() for p in paths
        ):
            raise ValueError("ALL source/own shards must close successfully")
        source, own = [json.loads(p.read_text()) for p in paths]
        if (
            source["status"] != SOURCE_STATUS
            or own["status"] != OWN_STATUS
            or source["source_head"] != a.producer_head
            or own["source_head"] != a.auditor_head
            or own["producer_head"] != a.producer_head
            or own["producer_hash"] != digest(paths[0])
            or own["manual_native_clock"] is not True
            or own["source_actions_executed"] is not False
            or own["borrowed_hidden_state"] is not False
            or own["counts"] != source["counts"]
            or any(
                v["context"] != ctx
                or v["seal_hash"] != a.seal_hash
                or v["models"] != seal["models"]
                or v["optimizer_updates"] != 0
                or v["private_case_rows_printed"] is not False
                or any(
                    v[k] is not False
                    for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
                )
                for v in (source, own)
            )
            or set(own["records"]) != set(seeds)
            or set(source["records"]) != set(seeds)
        ):
            raise ValueError("exact complete independent per-context accounting required")
        for seed in seeds:
            original, current = source["records"][seed], own["records"][seed]
            if len(original) != 24 or len(current) != 24:
                raise ValueError("all source and own tasks required")
            for src, row in zip(original, current, strict=True):
                if row != {k: v for k, v in src.items() if k != "artifacts"}:
                    raise ValueError("complete source/own measured outcomes must match")
            records[seed].extend(current)
        for path, doc in zip(paths, (source, own), strict=True):
            for key, h in doc["input_file_hashes"].items():
                if key in pins and pins[key] != h:
                    raise ValueError("conflicting immutable proof lineage")
                pins[key] = h
            pins[str(path)] = digest(path)
        for key in counts:
            counts[key] += own["counts"][key]
        proofs.append(dict(context=ctx, producer_hash=digest(paths[0]), own_hash=digest(paths[1])))
    source_pins(pins)
    require_frozen_pass_protocol(pins)
    check_pins(pins)
    from rosclaw_soccer.rsi.pass_fresh_curriculum import pass_fresh_screen

    baseline = records["65023"]
    reports = [
        dict(seed=int(seed), **pass_fresh_screen(baseline, records[seed]))
        for seed in seeds
        if seed != "65023"
    ]
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PASS_FRESH48_PAIRED_WHOLE_NATIVE_STUDY_CLOSED_NOT_PROMOTED",
            source_head=a.expected_head,
            seal_hash=a.seal_hash,
            input_file_hashes=pins,
            proofs=proofs,
            reports=reports,
            counts=counts,
            source_plus_own_native_steps=2 * counts["native_steps"],
            physical_initial_states=48,
            goal_tasks_per_policy=96,
            model_count=len(seeds),
            old_critical_retention_and_DEV_tradeoffs=seal["reports"],
            private_case_rows_published=False,
            Fresh_fit_rows=0,
            optimizer_updates=0,
            screen_passed_seeds=[r["seed"] for r in reports if r["local_pass_screen_passed"]],
            auto_promotion=False,
            legacy52_verified=False,
            strict_point_precision_verified=False,
            team_game_verified=False,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    print(json.dumps(dict(stage="WHOLE_FRESH_CLOSED", reports=reports)), flush=True)


if __name__ == "__main__":
    main()
