"""Collect or regenerate all24 private stochastic two-G1 TRAIN worlds.

No fitting, teacher rescue or promotion. Audit derives the exact receipt hash
only after its predeclared producer is closed, and reexecutes own RNG, NN and
independent physics, never applying saved actions/h/c to the audit world.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("expert-own-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expert-own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(25, 26), required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--audit-source-root", type=Path)
    p.add_argument("--producer-head")
    a = p.parse_args()
    if (a.audit_source_root is None) != (a.producer_head is None) or (
        a.audit_source_root is not None and a.preflight_only
    ):
        p.error("audit requires exact producer head/root and is not a preflight")
    from rosclaw_soccer.rsi.stochastic_refinement_pilot import pilot_config, verify_study
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_joint_delivery_pilot_common import admit, local_preflight
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only stochastic output required")
    cfg = pilot_config(a.state_id)
    pins, scope, bank, pf, inherited_root, baseline = admit(
        a.expert_own_root, a.expert_own_hash, cfg
    )
    source, source_hash = None, None
    if a.audit_source_root is not None:
        root = a.audit_source_root
        sp, dp = root / "complete.json", root / "declaration.json"
        if (
            root.is_symlink()
            or (root / "failed.json").exists()
            or any(
                x.is_symlink() or not x.is_file() or x.stat().st_size > 128 * 1024 * 1024
                for x in (sp, dp)
            )
        ):
            raise ValueError("closed bounded ordinary stochastic source required")
        source_hash = digest(sp)
        source, declaration = [json.loads(x.read_text()) for x in (sp, dp)]
        if (
            source["status"] != "OWN_STOCHASTIC_REFINEMENT24_COLLECTED_PENDING_NATIVE_LAW_AUDIT"
            or source["source_head"] != a.producer_head
            or declaration["source_head"] != a.producer_head
            or source["config"] != cfg
            or declaration["config"] != cfg
            or source["declaration_hash"] != digest(dp)
            or source["expert_own_hash"] != a.expert_own_hash
            or declaration["expert_own_hash"] != a.expert_own_hash
            or source["input_file_hashes"] != declaration["input_file_hashes"]
            or declaration["preflight_only"] is not False
            or len(source["records"]) != 24
            or source["optimizer_updates"] != 0
            or any(
                source[k] is not False
                for k in (
                    "hardware_authorized",
                    "promotion_authorized",
                    "goal_complete",
                    "Fresh_verified",
                    "team_game_verified",
                    "actor_training_admitted",
                )
            )
        ):
            raise ValueError("exact predeclared pure stochastic24 source required")
        for k, v in source["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting stochastic ancestry")
            pins[k] = v
        pins.update({str(sp): source_hash, str(dp): digest(dp)})
        for r in source["records"]:
            if set(r["artifacts"]) != {"initial", "physical", "A", "B", "sampling"}:
                raise ValueError("complete stochastic NN/RNG/native primitives required")
            for art in r["artifacts"].values():
                path = root / art["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (root / f"trial{r['index']}").resolve()
                    or digest(path) != art["hash"]
                ):
                    raise ValueError("confined exact stochastic primitives required")
                pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            config=cfg,
            input_file_hashes=pins,
            expert_own_root=str(a.expert_own_root),
            expert_own_hash=a.expert_own_hash,
            source_hash=source_hash,
            preflight_only=a.preflight_only,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    counts, records = dict(native_steps=0, actor_calls=0), []
    try:
        from scripts.rsi_self_teacher_search_common import open_setup

        setup, actor = open_setup(scope, pf, pins, bank, inherited_root / "scene.xml")
        variants, preflight = local_preflight(setup, baseline, cfg, inherited_root)
        courses = [baseline["course"], *variants]
        write_once(a.output / "preflight.json", preflight)
        if a.preflight_only:
            payload = dict(
                status="OWN_STOCHASTIC_REFINEMENT24_ALL_COURSES_PREFLIGHT_NO_PHYSICS",
                physical_steps_executed=0,
            )
        elif source is None:
            from scripts.rsi_pass_transfer_trial import collect_trial

            for course in courses:
                for seed in cfg["seeds"]:
                    row = collect_trial(
                        setup,
                        actor,
                        course,
                        output=a.output,
                        index=len(records),
                        counts=counts,
                        emit_course_rows=False,
                        sampling_seed=seed,
                        sampling_profile=cfg["sampling_profile"],
                    )
                    records.append(row)
                    write_once(a.output / f"record{row['index']}.json", row)
            verify_study(dict(config=cfg, records=records, counts=counts), courses)
            payload = dict(
                status="OWN_STOCHASTIC_REFINEMENT24_COLLECTED_PENDING_NATIVE_LAW_AUDIT",
                records=records,
                counts=counts,
            )
        else:
            from scripts.rsi_pass_transfer_replay_trial import replay_trial

            if preflight != source["preflight"]:
                raise ValueError("own stochastic course regeneration differs")
            verify_study(source, courses)
            prior = None
            for row in source["records"]:
                actual, prior = replay_trial(
                    setup,
                    actor,
                    row["course"],
                    row,
                    raw_root=a.audit_source_root,
                    output=a.output,
                    index=row["index"],
                    counts=counts,
                    prior_initial=prior,
                    emit_course_rows=False,
                    sampling_seed=row["sampling_seed"],
                    sampling_profile=cfg["sampling_profile"],
                )
                records.append(actual)
            if counts != source["counts"]:
                raise ValueError("all stochastic own native and NN clocks differ")
            verify_study(dict(config=cfg, records=records, counts=counts), courses)
            payload = dict(
                status="OWN_STOCHASTIC_REFINEMENT24_ALL_NATIVE_PRIVATE_RNG_LAW_MATCH",
                records=records,
                counts=counts,
                source_hash=source_hash,
                independent_whole_world_replay=True,
                sampling_law_audit_passed=True,
            )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            payload
            | dict(
                source_head=a.expected_head,
                config=cfg,
                preflight=preflight,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                expert_own_hash=a.expert_own_hash,
                optimizer_updates=0,
                actor_training_admitted=False,
                hardware_authorized=False,
                promotion_authorized=False,
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
