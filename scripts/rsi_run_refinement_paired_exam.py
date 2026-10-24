"""Own-state OLD vs learned-PPO mean-policy pairs, with independent native replay.

Consumed TRAIN only. No sampling, optimizer updates, teacher rescue, physical
retention certification, Fresh access, policy promotion or team-game claims.
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
        p.error("own replay requires exact predeclared producer head/root")
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import (
        file_hash,
    )
    from rosclaw_soccer.rsi.refinement_paired_exam import exam_config, verify_exam
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_joint_delivery_pilot_common import local_preflight
    from scripts.rsi_learned_refinement_common import open_learned_model
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_refinement_onpolicy_data import admit

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only physical exam output required")
    cfg = exam_config(a.state_id)
    experience, _, pins, scope, bank, pf, behavior_root = admit(
        a.experience_root, a.own_root, a.own_hash, a.state_id
    )
    tp = a.training_root / "complete.json"
    if file_hash(tp) != a.training_hash:
        raise ValueError("exact completed training required before physical admission")
    trained = json.loads(tp.read_text())
    if trained["config"]["state_id"] != a.state_id:
        raise ValueError("training state differs from preregistered exam")
    for k, v in trained["input_file_hashes"].items():
        if k in pins and pins[k] != v:
            raise ValueError("conflicting learned ancestry")
        pins[k] = v
    for f in (tp, a.training_root / "declaration.json", Path(trained["checkpoint_path"])):
        pins[str(f)] = file_hash(f)
    source = None
    if a.audit_source_root is not None:
        root = a.audit_source_root
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("ordinary closed paired source required")
        sp, dp = root / "complete.json", root / "declaration.json"
        source, declaration = [json.loads(f.read_text()) for f in (sp, dp)]
        if (
            source["status"] != "PPO_PAIRED_MEAN12_COLLECTED_PENDING_OWN_NATIVE"
            or source["config"] != cfg
            or declaration["config"] != cfg
            or source["source_head"] != a.producer_head
            or declaration["source_head"] != a.producer_head
            or source["training_hash"] != a.training_hash
            or declaration["training_hash"] != a.training_hash
            or source["declaration_hash"] != file_hash(dp)
            or source["input_file_hashes"] != declaration["input_file_hashes"]
            or any(
                source[k] is not False
                for k in (
                    "goal_complete",
                    "promotion_authorized",
                    "hardware_authorized",
                    "Fresh_verified",
                    "team_game_verified",
                )
            )
        ):
            raise ValueError("exact predeclared source and provenance required")
        for k, v in source["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting paired source ancestry")
            pins[k] = v
        for f in (sp, dp):
            pins[str(f)] = file_hash(f)
        for r in source["records"]:
            if set(r["artifacts"]) != {"initial", "physical", "A", "B"}:
                raise ValueError("all unassisted source primitives required")
            for artifact in r["artifacts"].values():
                f = root / artifact["path"]
                if (
                    f.resolve().parent != (root / f"trial{r['index']}").resolve()
                    or digest(f) != artifact["hash"]
                    or f.is_symlink()
                ):
                    raise ValueError("confined exact source primitives required")
                pins[str(f)] = artifact["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            config=cfg,
            source_head=a.expected_head,
            training_hash=a.training_hash,
            input_file_hashes=pins,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    counts, records = dict(native_steps=0, actor_calls=0), []
    try:
        from scripts.rsi_self_teacher_search_common import open_setup

        setup, old = open_setup(scope, pf, pins, bank, behavior_root / "scene.xml")
        learned, _ = open_learned_model(old, a, pins, scope)
        baseline = next(r for r in experience["records"] if r["index"] == 0)
        pre_cfg = dict(unsearched_TRAIN_offsets_forward_m=[-0.005, 0.005])
        # Baseline initial primitives belong to the stochastic source, not its ancestor.
        variants, preflight = local_preflight(setup, baseline, pre_cfg, a.experience_root)
        courses = [baseline["course"], *variants]
        write_once(a.output / "preflight.json", preflight)
        if source is None:
            from scripts.rsi_pass_transfer_trial import collect_trial

            for course in courses:
                for arm in cfg["arms"]:
                    r = collect_trial(
                        setup,
                        old if arm == "old" else learned,
                        course,
                        output=a.output,
                        index=len(records),
                        counts=counts,
                        emit_course_rows=False,
                    )
                    r["arm"] = arm
                    records.append(r)
                    write_once(a.output / f"record{r['index']}.json", r)
            status = "PPO_PAIRED_MEAN12_COLLECTED_PENDING_OWN_NATIVE"
        else:
            from scripts.rsi_pass_transfer_replay_trial import replay_trial

            if preflight != source["preflight"]:
                raise ValueError("independent original initial preflight differs")
            verify_exam(source, courses)
            prior = None
            for claim in source["records"]:
                arm = claim["arm"]
                bare = {k: v for k, v in claim.items() if k != "arm"}
                r, prior = replay_trial(
                    setup,
                    old if arm == "old" else learned,
                    claim["course"],
                    bare,
                    raw_root=a.audit_source_root,
                    output=a.output,
                    index=claim["index"],
                    counts=counts,
                    prior_initial=prior,
                    emit_course_rows=False,
                )
                r["arm"] = arm
                records.append(r)
            if (
                records
                != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
                or counts != source["counts"]
            ):
                raise ValueError("all paired own-native results must match")
            status = "PPO_PAIRED_MEAN12_ALL_OWN_NATIVE_MATCH"
        verify_exam(dict(config=cfg, records=records, counts=counts), courses)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=status,
                config=cfg,
                source_head=a.expected_head,
                training_hash=a.training_hash,
                preflight=preflight,
                input_file_hashes=pins,
                records=records,
                counts=counts,
                declaration_hash=digest(a.output / "declaration.json"),
                source_hash=None
                if source is None
                else digest(a.audit_source_root / "complete.json"),
                independent_whole_world_replay=source is not None,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
                Fresh_verified=False,
                team_game_verified=False,
                optimizer_updates=0,
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
                goal_complete=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
