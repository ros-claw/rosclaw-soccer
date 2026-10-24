"""Collect or whole-own replay10 bounded causal foot/ball feedback TRAIN worlds."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("expert-own-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expert-own-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--state-id", type=int, choices=(25, 26), required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--source-root", type=Path)
    p.add_argument("--source-hash")
    if_missing = "audit requires both source root/hash and is not preflight-only"
    a = p.parse_args()
    if (a.source_root is None) != (a.source_hash is None) or (
        a.source_root is not None and a.preflight_only
    ):
        p.error(if_missing)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_joint_delivery_pilot_common import admit, local_preflight
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only causal study output required")
    from rosclaw_soccer.rsi.causal_feedback_pilot import (
        pilot_config,
        selection,
        teacher_qualified,
        verify_study,
    )

    cfg = pilot_config(a.state_id)
    pins, scope, bank, pf, inherited_root, baseline = admit(
        a.expert_own_root, a.expert_own_hash, cfg
    )
    if baseline["course"]["foot"] != "left":
        raise ValueError("diagnosed original left-foot TRAIN history required")
    source = None
    if a.source_root is not None:
        sp, dp = a.source_root / "complete.json", a.source_root / "declaration.json"
        if (
            a.source_root.is_symlink()
            or any(x.is_symlink() for x in (sp, dp))
            or (a.source_root / "failed.json").exists()
            or digest(sp) != a.source_hash
        ):
            raise ValueError("complete exact ordinary feedback source required")
        source, declaration = [json.loads(x.read_text()) for x in (sp, dp)]
        if (
            source["status"] != "CAUSAL_FOOT_BALL_TRAIN10_COLLECTED_PENDING_OWN"
            or source["config"] != cfg
            or source["expert_own_hash"] != a.expert_own_hash
            or source["source_head"] != a.expected_head
            or source["declaration_hash"] != digest(dp)
            or source["input_file_hashes"] != declaration["input_file_hashes"]
            or declaration["config"] != cfg
            or declaration["expert_own_hash"] != a.expert_own_hash
            or declaration["preflight_only"] is not False
            or len(source["records"]) != 10
            or source["optimizer_updates"] != 0
            or any(
                source[k] is not False
                for k in (
                    "goal_complete",
                    "hardware_authorized",
                    "promotion_authorized",
                    "Fresh_verified",
                    "team_game_verified",
                    "teacher_qualified",
                )
            )
        ):
            raise ValueError("unchanged complete10-world feedback source admission required")
        for k, v in source["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting feedback source ancestry")
            pins[k] = v
        pins.update({str(sp): a.source_hash, str(dp): digest(dp)})
        for row in source["records"]:
            if set(row["artifacts"]) != {"initial", "physical", "A", "B", "feedback"}:
                raise ValueError("all own-history and measured feedback archives required")
            for art in row["artifacts"].values():
                path = a.source_root / art["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (a.source_root / f"trial{row['index']}").resolve()
                    or digest(path) != art["hash"]
                ):
                    raise ValueError("exact confined physical feedback primitives required")
                pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    declared = dict(
        source_head=a.expected_head,
        input_file_hashes=pins,
        config=cfg,
        expert_own_root=str(a.expert_own_root),
        expert_own_hash=a.expert_own_hash,
        source_hash=a.source_hash,
        preflight_only=a.preflight_only,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    write_once(a.output / "declaration.json", declared)
    counts, records = dict(native_steps=0, actor_calls=0), []
    try:
        from scripts.rsi_self_teacher_search_common import compare_common, open_setup

        setup, actor = open_setup(scope, pf, pins, bank, inherited_root / "scene.xml")
        variants, preflight = local_preflight(setup, baseline, cfg, inherited_root)
        write_once(a.output / "preflight.json", preflight)
        if a.preflight_only:
            payload = dict(
                status="CAUSAL_FEEDBACK_ALL_ORIGINAL40_AND_LOCAL3_PREFLIGHT_NOT_PHYSICS",
                physical_steps_executed=0,
            )
        elif source is None:
            from scripts.rsi_pass_transfer_trial import collect_trial

            def run(course, variant):
                if len(records) >= cfg["maximum_worlds"]:
                    raise ValueError("finite causal10-world budget exhausted")
                row = collect_trial(
                    setup,
                    actor,
                    course,
                    output=a.output,
                    index=len(records),
                    counts=counts,
                    emit_course_rows=False,
                    feedback_variant=variant,
                )
                records.append(row)
                row["prefix_identity_fields"] = (
                    compare_common(a.output, row, inherited_root, baseline, prefix=80)
                    if row["index"] < 6
                    else None
                )
                write_once(a.output / f"record{row['index']}.json", row)

            for v in cfg["search_variants"]:
                for _ in range(2):
                    run(baseline["course"], v)
            selected = selection(baseline, records)
            for c in variants:
                for _ in range(2):
                    run(c, selected["variant"])
            study = dict(config=cfg, records=records, selected=selected)
            verify_study(study, baseline, variants)
            payload = dict(
                status="CAUSAL_FOOT_BALL_TRAIN10_COLLECTED_PENDING_OWN",
                records=records,
                selected=selected,
                counts=counts,
                research_screen_before_own=teacher_qualified(study),
            )
        else:
            from scripts.rsi_pass_transfer_replay_trial import replay_trial

            if preflight != source["preflight"]:
                raise ValueError("independently regenerated feedback course bank differs")
            verify_study(source, baseline, variants)
            previous = None
            for row in source["records"]:
                claim = {k: v for k, v in row.items() if k != "prefix_identity_fields"}
                actual, previous = replay_trial(
                    setup,
                    actor,
                    row["course"],
                    claim,
                    raw_root=a.source_root,
                    output=a.output,
                    index=row["index"],
                    counts=counts,
                    prior_initial=previous,
                    emit_course_rows=False,
                    feedback_variant=row["feedback_variant"],
                )
                prefix = (
                    compare_common(a.source_root, row, inherited_root, baseline, prefix=80)
                    if row["index"] < 6
                    else None
                )
                if prefix != row["prefix_identity_fields"]:
                    raise ValueError("independently verified zero-feedback prefix differs")
                records.append(actual | dict(prefix_identity_fields=prefix))
            if counts != source["counts"]:
                raise ValueError("all own physical/inference counters differ")
            study = source | dict(records=records)
            verify_study(study, baseline, variants)
            payload = dict(
                status="CAUSAL_FOOT_BALL_TRAIN10_ALL_WHOLE_OWN_NATIVE_MATCH",
                records=records,
                selected=source["selected"],
                counts=counts,
                source_hash=a.source_hash,
                research_screen_after_own=teacher_qualified(study),
                independent_whole_world_replay=True,
            )
        if (
            counts["native_steps"] > cfg["maximum_native_steps"]
            or counts["actor_calls"] > cfg["maximum_actor_calls"]
        ):
            raise ValueError("actual causal study clock exceeded preregistered budget")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            payload
            | dict(
                source_head=a.expected_head,
                config=cfg,
                preflight=preflight,
                declaration_hash=digest(a.output / "declaration.json"),
                input_file_hashes=pins,
                expert_own_hash=a.expert_own_hash,
                optimizer_updates=0,
                teacher_qualified=False,
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
