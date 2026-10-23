"""Whole private-state / manual native-clock audit of one sealed Fresh shard.

All policies and all cases are replayed, not selected successes. The producer
integrator, recorded controls and recorded h/c are never used for execution.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("seal-root", "source-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("seal-hash", "producer-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-hash")
    group.add_argument(
        "--completed-predecessor",
        action="store_true",
        help="Resolve completed predecessor hash once, then verify full source identity.",
    )
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_pass_fresh_common import (
        OWN_STATUS,
        SOURCE_STATUS,
        admit_pass_seal,
        confined_artifact,
        open_pass_seal_setup,
        require_frozen_pass_protocol,
        verify_trial_initial,
    )

    seal, pins, private, initial = admit_pass_seal(a.seal_root, a.seal_hash)
    rp, dp = a.source_root / "complete.json", a.source_root / "declaration.json"
    source_hash = digest(rp) if a.completed_predecessor else a.source_hash
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (rp, dp))
        or a.source_root.is_symlink()
        or (a.source_root / "failed.json").exists()
        or digest(rp) != source_hash
    ):
        raise ValueError("completed original source and unused whole-own namespace required")
    s, d = [json.loads(p.read_text()) for p in (rp, dp)]
    context = s["context"]
    courses = [c for c in private["courses"] if c["context"] == context]
    seeds = [str(m["seed"]) for m in seal["models"]]
    worlds = 24 * len(seeds)
    if (
        s["status"] != SOURCE_STATUS
        or context not in (1, 4, 20, 21)
        or len(courses) != 24
        or s["declaration_hash"] != digest(dp)
        or s["input_file_hashes"] != d["input_file_hashes"]
        or set(s["records"]) != set(seeds)
        or d["context"] != context
        or d["worlds"] != worlds
        or d["maximum_native_steps"] != worlds * 4000
        or d["maximum_actor_calls"] != worlds * 800
        or d["teacher_enabled"] is not False
        or d["preflight_only"] is not False
        or any(
            v["source_head"] != a.producer_head
            or v["seal_hash"] != a.seal_hash
            or v["models"] != seal["models"]
            or v["optimizer_updates"] != 0
            or v["private_case_rows_printed"] is not False
            or any(
                v[k] is not False
                for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
            )
            for v in (s, d)
        )
    ):
        raise ValueError(
            "whole declared source, frozen models, scope and privacy boundary required"
        )
    for key, h in s["input_file_hashes"].items():
        if key in pins and pins[key] != h:
            raise ValueError("conflicting sealed source lineage")
        pins[key] = h
    for path in (rp, dp):
        pins[str(path)] = digest(path)
    for seed in seeds:
        rows = s["records"][seed]
        if len(rows) != 24:
            raise ValueError("ALL24 original goal tasks per policy required")
        for i, row in enumerate(rows):
            if (
                row["index"] != i
                or row["course"] != courses[i]
                or set(row["artifacts"]) != {"initial", "physical", "A", "B"}
            ):
                raise ValueError("source must retain every whole sealed case and raw artifact")
            for art in row["artifacts"].values():
                path = confined_artifact(a.source_root / f"policy{seed}", art["path"], art["hash"])
                pins[str(path)] = art["hash"]
    source_pins(pins)
    require_frozen_pass_protocol(pins)
    check_pins(pins)
    a.output.mkdir(mode=0o700, parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_head=a.producer_head,
            producer_hash=source_hash,
            seal_hash=a.seal_hash,
            context=context,
            models=seal["models"],
            input_file_hashes=pins,
            manual_native_clock=True,
            source_actions_executed=False,
            borrowed_hidden_state=False,
            private_case_rows_printed=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    counts = dict(native_steps=0, actor_calls=0)
    records = {}
    try:
        from scripts.rsi_pass_transfer_replay_trial import replay_trial

        setup, models = open_pass_seal_setup(seal, pins, private, initial)
        for seed, actor in models.items():
            target = a.output / f"policy{seed}"
            target.mkdir()
            prior = None
            rows = []
            for i, c in enumerate(courses):
                verify_trial_initial(setup, c, initial[c["state_id"]])
                row, prior = replay_trial(
                    setup,
                    actor,
                    c,
                    s["records"][str(seed)][i],
                    raw_root=a.source_root / f"policy{seed}",
                    output=target,
                    index=i,
                    counts=counts,
                    prior_initial=prior,
                    emit_course_rows=False,
                )
                rows.append(row)
            records[str(seed)] = rows
        if counts != s["counts"] or any(len(v) != 24 for v in records.values()):
            raise ValueError("all native/actor accounting and whole policy set must match")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=OWN_STATUS,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                producer_head=a.producer_head,
                producer_hash=source_hash,
                seal_hash=a.seal_hash,
                context=context,
                models=seal["models"],
                input_file_hashes=pins,
                records=records,
                counts=counts,
                optimizer_updates=0,
                manual_native_clock=True,
                source_actions_executed=False,
                borrowed_hidden_state=False,
                private_case_rows_printed=False,
                Fresh_success_claim=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
