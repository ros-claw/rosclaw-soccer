"""Paired OLD/learned own stochastic worlds on NEW seeds; independent own replay.

All cases are consumed PUBLIC TRAIN initials, not sealed Fresh. Both policies
receive the same declared random seed but each regenerates its own mean,
recurrent history, bounded AR probability law and actually executed action.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("experience-root", "own-root", "training-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("own-hash", "training-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(25, 26), required=True)
    p.add_argument("--audit-source-root", type=Path)
    p.add_argument("--producer-head")
    a = p.parse_args()
    if (a.audit_source_root is None) != (a.producer_head is None):
        p.error("own replay requires predeclared source/head pair")
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from rosclaw_soccer.rsi.refinement_sampling_exam import (
        paired_rewards,
        sampling_exam_config,
        verify_sampling_exam,
    )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_learned_refinement_common import admit, open_models
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only sampling exam output required")
    cfg = sampling_exam_config(a.state_id)
    admitted = admit(a)
    pins = admitted[1]
    source, source_hash = None, None
    if a.audit_source_root is not None:
        root = a.audit_source_root
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("ordinary complete paired sampling source required")
        sp, dp = root / "complete.json", root / "declaration.json"
        source_hash = file_hash(sp)
        source, declaration = [json.loads(f.read_text()) for f in (sp, dp)]
        if (
            source["status"] != "PPO_NEW_SEED_PAIRED48_COLLECTED_PENDING_OWN_NATIVE_LAW"
            or source["source_head"] != a.producer_head
            or declaration["source_head"] != a.producer_head
            or source["config"] != cfg
            or declaration["config"] != cfg
            or source["training_hash"] != a.training_hash
            or declaration["training_hash"] != a.training_hash
            or source["declaration_hash"] != file_hash(dp)
            or source["input_file_hashes"] != declaration["input_file_hashes"]
            or source["optimizer_updates"] != 0
            or any(
                source[k] is not False
                for k in (
                    "goal_complete",
                    "hardware_authorized",
                    "promotion_authorized",
                    "Fresh_verified",
                    "team_game_verified",
                    "actor_training_admitted",
                )
            )
        ):
            raise ValueError("exact predeclared complete sampling48 source required")
        for k, v in source["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting paired sampling ancestry")
            pins[k] = v
        for path in (sp, dp):
            pins[str(path)] = file_hash(path)
        for r in source["records"]:
            if set(r["artifacts"]) != {"initial", "physical", "A", "B", "sampling"}:
                raise ValueError("all physical/private stochastic primitives required")
            for artifact in r["artifacts"].values():
                path = root / artifact["path"]
                if (
                    path.resolve().parent != (root / f"trial{r['index']}").resolve()
                    or file_hash(path) != artifact["hash"]
                ):
                    raise ValueError("confined exact sampling source primitives required")
                pins[str(path)] = artifact["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            config=cfg,
            training_hash=a.training_hash,
            input_file_hashes=pins,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    records, counts = [], dict(native_steps=0, actor_calls=0)
    try:
        setup, old, learned, courses, preflight = open_models(a, admitted)
        write_once(a.output / "preflight.json", preflight)
        if source is None:
            from scripts.rsi_pass_transfer_trial import collect_trial

            for course in courses:
                for seed in cfg["seeds"]:
                    for arm in cfg["arms"]:
                        r = collect_trial(
                            setup,
                            old if arm == "old" else learned,
                            course,
                            output=a.output,
                            index=len(records),
                            counts=counts,
                            sampling_seed=seed,
                            sampling_profile=cfg["sampling_profile"],
                            emit_course_rows=False,
                        )
                        r["arm"] = arm
                        records.append(r)
                        write_once(a.output / f"record{r['index']}.json", r)
            status = "PPO_NEW_SEED_PAIRED48_COLLECTED_PENDING_OWN_NATIVE_LAW"
        else:
            from scripts.rsi_pass_transfer_replay_trial import replay_trial

            if preflight != source["preflight"]:
                raise ValueError("all independent same-course preflight must match")
            verify_sampling_exam(source, courses)
            prior = None
            for claim in source["records"]:
                arm = claim["arm"]
                r, prior = replay_trial(
                    setup,
                    old if arm == "old" else learned,
                    claim["course"],
                    {k: v for k, v in claim.items() if k != "arm"},
                    raw_root=a.audit_source_root,
                    output=a.output,
                    index=claim["index"],
                    counts=counts,
                    prior_initial=prior,
                    emit_course_rows=False,
                    sampling_seed=claim["sampling_seed"],
                    sampling_profile=cfg["sampling_profile"],
                )
                r["arm"] = arm
                records.append(r)
            if (
                records
                != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
                or counts != source["counts"]
            ):
                raise ValueError("all own sampling/native results must match")
            status = "PPO_NEW_SEED_PAIRED48_ALL_OWN_NATIVE_AND_LAW_MATCH"
        verify_sampling_exam(dict(config=cfg, records=records, counts=counts), courses)
        summary = paired_rewards(records)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=status,
                source_head=a.expected_head,
                config=cfg,
                training_hash=a.training_hash,
                input_file_hashes=pins,
                declaration_hash=file_hash(a.output / "declaration.json"),
                source_hash=source_hash,
                preflight=preflight,
                records=records,
                counts=counts,
                paired_summary=summary,
                independent_whole_world_replay=source is not None,
                sampling_law_audit_passed=source is not None,
                optimizer_updates=0,
                actor_training_admitted=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
                Fresh_verified=False,
                team_game_verified=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                counts=counts,
                completed_worlds=len(records),
                records=records,
                optimizer_updates=0,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
