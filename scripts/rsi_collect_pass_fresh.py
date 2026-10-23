"""One bounded actual-physics context shard of the already sealed pass Fresh48.

No policy fitting, course sampling, scene changes, skips, or promotion. All
three frozen policies execute the same24 goal trials with private h/c.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("seal-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("seal-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--context", type=int, choices=(1, 4, 20, 21), required=True)
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_pass_fresh_common import (
        SOURCE_STATUS,
        admit_pass_seal,
        open_pass_seal_setup,
        require_frozen_pass_protocol,
        verify_trial_initial,
    )

    seal, pins, private, initial = admit_pass_seal(a.seal_root, a.seal_hash)
    if a.output.exists():
        raise ValueError("unused bounded Fresh execution namespace required")
    courses = [c for c in private["courses"] if c["context"] == a.context]
    if len(courses) != 24:
        raise ValueError("whole balanced context shard required")
    worlds = 24 * len(seal["models"])
    source_pins(pins)
    require_frozen_pass_protocol(pins)
    check_pins(pins)
    a.output.mkdir(mode=0o700, parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            seal_hash=a.seal_hash,
            input_file_hashes=pins,
            context=a.context,
            models=seal["models"],
            worlds=worlds,
            preflight_only=a.preflight_only,
            maximum_native_steps=worlds * 4000,
            maximum_actor_calls=worlds * 800,
            teacher_enabled=False,
            optimizer_updates=0,
            private_case_rows_printed=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    counts = dict(native_steps=0, actor_calls=0)
    records = {}
    try:
        import numpy as np

        from scripts.rsi_pass_transfer_trial import collect_trial

        setup, models = open_pass_seal_setup(seal, pins, private, initial)
        if a.preflight_only:
            check_pins(pins)
            clean(a.expected_head)
            write_once(
                a.output / "complete.json",
                dict(
                    status="PASS_FRESH_ALL48_WORKER_PREFLIGHT_NOT_PHYSICS",
                    source_head=a.expected_head,
                    seal_hash=a.seal_hash,
                    input_file_hashes=pins,
                    declaration_hash=digest(a.output / "declaration.json"),
                    all48_compiled_states_match=True,
                    model_seeds=list(models),
                    physical_steps=0,
                    optimizer_updates=0,
                    hardware_authorized=False,
                    promotion_authorized=False,
                    goal_complete=False,
                ),
            )
            print(
                json.dumps(dict(stage="PREFLIGHT_COMPLETE", states=48, physical_steps=0)),
                flush=True,
            )
            return
        for seed, actor in models.items():
            root = a.output / f"policy{seed}"
            root.mkdir()
            rows = []
            for i, c in enumerate(courses):
                verify_trial_initial(setup, c, initial[c["state_id"]])
                row = collect_trial(
                    setup, actor, c, output=root, index=i, counts=counts, emit_course_rows=False
                )
                # Check the actually persisted initialization, not only preflight.
                with np.load(root / row["artifacts"]["initial"]["path"], allow_pickle=False) as z:
                    from rosclaw_soccer.sim.contracts import hash_json

                    if (
                        hash_json(dict(qpos=z["qpos"].tolist(), qvel=z["qvel"].tolist()))
                        != initial[c["state_id"]]
                    ):
                        raise ValueError(
                            "actual trial initial arrays differ from sealed paired state"
                        )
                rows.append(row)
            records[str(seed)] = rows
            print(
                json.dumps(dict(policy=seed, completed_trials=len(rows), context=a.context)),
                flush=True,
            )
        if (
            len(records) != len(seal["models"])
            or any(len(v) != 24 for v in records.values())
            or counts["native_steps"] > worlds * 4000
            or counts["actor_calls"] > worlds * 800
        ):
            raise ValueError("complete bounded paired context required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=SOURCE_STATUS,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                input_file_hashes=pins,
                seal_hash=a.seal_hash,
                context=a.context,
                models=seal["models"],
                records=records,
                counts=counts,
                physical_steps=counts["native_steps"],
                optimizer_updates=0,
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
