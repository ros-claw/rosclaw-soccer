"""Close four matched Fresh shards only after every independent native replay."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("seal-root", "shards-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.native_fresh_curriculum import native_fresh_summary
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, read_seal, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused complete matched Fresh readback required")
    seal, pins, courses = read_seal(a.seal_root, a.expected_head)
    records = []
    producer_native = producer_neural = own_native = own_neural = 0
    for shard in range(4):
        proot, aroot = a.shards_root / f"producer{shard}", a.shards_root / f"audit{shard}"
        pp, ap = proot / "complete.json", aroot / "complete.json"
        if any(p.is_symlink() for p in (pp, ap)) or any(
            (r / "failed.json").exists() for r in (proot, aroot)
        ):
            raise ValueError("all four complete producer and own-native audit shards required")
        production, audit = [json.loads(p.read_text()) for p in (pp, ap)]
        if (
            production["status"] != "NATIVE_FRESH_SHARD_COLLECTED_PENDING_OWN_REPLAY"
            or audit["status"] != "NATIVE_FRESH_SHARD_ALL_OWN_NATIVE_AUDIT_PASS"
            or audit["producer_hash"] != digest(pp)
            or audit["records"] != production["records"]
            or audit["independent_whole_world_replay"] is not True
            or any(
                r["source_head"] != a.expected_head
                or r["shard"] != shard
                or r["seal_hash"] != digest(a.seal_root / "complete.json")
                or r["optimizer_updates"] != 0
                or r["FRESH_fit_rows"] != 0
                or r["hardware_authorized"] is not False
                or r["promotion_authorized"] is not False
                for r in (production, audit)
            )
            or audit["counts"]["native"] != production["counts"]["native"]
            or audit["counts"]["neural"] != production["counts"]["neural"]
        ):
            raise ValueError(
                "same sealed courses, exact source binding and all native replays required"
            )
        for path, h in audit["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("conflicting independent proof pins")
            pins[path] = h
        for path in (pp, ap):
            pins[str(path)] = digest(path)
        records.extend(production["records"])
        producer_native += production["counts"]["native"]
        producer_neural += production["counts"]["neural"]
        own_native += audit["counts"]["native"]
        own_neural += audit["counts"]["neural"]
    seeds = [c["seed"] for c in seal["candidates"]]
    summary = native_fresh_summary(records, courses, seeds)
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(mode=0o700, parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="SEALED_NATIVE_FRESH56_ALL_MATCHED_OWN_REPLAYS_CLOSED_NOT_PROMOTED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            seal_hash=digest(a.seal_root / "complete.json"),
            summary=summary,
            producer_native_steps=producer_native,
            producer_rollout_decisions=producer_neural,
            independent_native_steps=own_native,
            independent_rollout_decisions=own_neural,
            course_rows_published=False,
            consumed_Fresh_no_longer_available_for_future_blind_tuning=True,
            all_historical_gate_rejected_seeds_preserved_in_frontier=True,
            training_updates=0,
            old52_verified=False,
            continuous_team_verified=False,
            no_auto_activation=True,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
