"""Spawn-safe native worker; no model, simulator, GPU or device on import.

One model/setup per CPU child, private h/c and RNG per new world. Two adjacent
same-initial worlds per task preserve the existing paired-initial contract.
Failures are kept; no task or physics retry is provided.
"""

import os
import pickle
import sys

_STATE = None


def bootstrap_probe():
    return dict(
        module_registered=__name__ in sys.modules,
        worker_pickle_roundtrip=pickle.loads(pickle.dumps(run_pair)) is run_pair,
        model_allocated=_STATE is not None,
        pid=os.getpid(),
    )


def initialize(args, admitted, config, source):
    global _STATE
    if _STATE is not None:
        raise ValueError("worker cannot replace its bound policy")
    from scripts.rsi_native_fresh_common import clean

    clean(args.expected_head)
    import torch

    from scripts.rsi_learned_refinement_common import open_models

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    setup, _, actor, courses, preflight = open_models(args, admitted)
    _STATE = dict(
        args=args,
        setup=setup,
        actor=actor,
        courses=courses,
        preflight=preflight,
        config=config,
        source=source,
    )


def run_pair(pair):
    if _STATE is None:
        raise ValueError("explicit admitted CPU worker initialization required")
    from rosclaw_soccer.rsi.parallel_refinement_batch import pair_plan
    from scripts.rsi_atomic_artifacts import write_once

    s = _STATE
    args, cfg = s["args"], s["config"]
    plan = pair_plan(cfg, s["courses"], pair)
    counts, records = dict(native_steps=0, actor_calls=0), []
    write_once(
        args.output / f"pair{pair}.started.json",
        dict(
            pair=pair,
            pid=os.getpid(),
            indices=[r["index"] for r in plan],
            maximum_native_steps=8000,
            maximum_actor_calls=1600,
            source_head=args.expected_head,
            training_hash=args.training_hash,
        ),
    )
    try:
        prior = None
        for row in plan:
            common = dict(
                output=args.output,
                index=row["index"],
                counts=counts,
                emit_course_rows=False,
                sampling_seed=row["sampling_seed"],
                sampling_profile=cfg["sampling_profile"],
            )
            if s["source"] is None:
                from scripts.rsi_pass_transfer_trial import collect_trial

                r = collect_trial(s["setup"], s["actor"], row["course"], **common)
            else:
                from scripts.rsi_pass_transfer_replay_trial import replay_trial

                claim = s["source"]["records"][row["index"]]
                r, prior = replay_trial(
                    s["setup"],
                    s["actor"],
                    row["course"],
                    claim,
                    raw_root=args.audit_source_root,
                    prior_initial=prior,
                    **common,
                )
            records.append(r)
            write_once(args.output / f"record{r['index']}.json", r)
        result = dict(pair=pair, pid=os.getpid(), counts=counts, records=records)
        write_once(args.output / f"pair{pair}.complete.json", result)
        return result
    except BaseException as error:
        write_once(
            args.output / f"pair{pair}.failed.json",
            dict(
                pair=pair,
                pid=os.getpid(),
                error=repr(error),
                observed_collector_counts=counts,
                records=records,
                incomplete_world_native_steps_unknown=True,
                reserved_native_step_ceiling=8000,
                goal_complete=False,
                hardware_authorized=False,
            ),
        )
        raise
