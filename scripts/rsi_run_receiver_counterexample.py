"""Reuse two authenticated factual failures; execute only the B-role intervention.

No action/hidden-state playback. Every new draw uses own NN observations and
the original private seed. Own audit uses the independent manual native clock.
Four selected TRAIN worlds and their repeats are NOT a generalization exam.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in (
        "experience-root",
        "own-root",
        "training-root",
        "factual-source-root",
        "factual-own-root",
        "output",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("own-hash", "training-hash", "factual-own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--audit-source-root", type=Path)
    p.add_argument("--producer-head")
    p.add_argument("--projected-capture", action="store_true")
    a = p.parse_args()
    if (a.audit_source_root is None) != (a.producer_head is None):
        p.error("exact own producer root/head pair required")
    a.state_id = 26
    from rosclaw_soccer.rsi.continuing_refinement_contract import KIND, select_parallel_experience
    from rosclaw_soccer.rsi.receiver_counterexample import (
        OWN,
        SOURCE,
        config,
        projected_config,
        verify,
    )
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_learned_refinement_common import admit, open_models
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only counterexample output required")
    admitted = admit(a)
    pins = admitted[1]
    cfg = projected_config() if a.projected_capture else config()
    if a.projected_capture:
        SOURCE, OWN = (
            "PROJECTED_RECEIVER6_PENDING_OWN_NATIVE",
            "PROJECTED_RECEIVER6_ALL_NATIVE_LAW_MATCH",
        )
    trained = json.loads((a.training_root / "complete.json").read_text())
    if (
        trained["checkpoint_kind"] != KIND
        or trained["config"]["generation_index"] != 2
        or trained["cumulative_actor_steps"] != 12
        or trained["cumulative_critic_steps"] != 12
    ):
        raise ValueError("exact fixed actual generation2 A required")
    factual_reports = []
    for root in (a.factual_source_root, a.factual_own_root):
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("closed ordinary factual source/own required")
        path = root / "complete.json"
        pins[str(path)] = file_hash(path)
        r = json.loads(path.read_text())
        factual_reports.append(r)
        for k, v in r["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting factual ancestry")
            pins[k] = v
    factual, factual_own = factual_reports
    if pins[str(a.factual_own_root / "complete.json")] != a.factual_own_hash:
        raise ValueError("exact predeclared factual own proof required")
    select_parallel_experience(
        factual,
        factual_own,
        source_hash=pins[str(a.factual_source_root / "complete.json")],
        training_hash=a.training_hash,
        state=26,
        full_recovery=True,
    )

    def raw_pins(root, records):
        for r in records:
            if set(r["artifacts"]) != {"initial", "physical", "A", "B", "sampling"}:
                raise ValueError("all actual primitives required")
            for art in r["artifacts"].values():
                path = root / art["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (root / f"trial{r['index']}").resolve()
                    or file_hash(path) != art["hash"]
                ):
                    raise ValueError("exact confined primitives required")
                pins[str(path)] = art["hash"]

    facts = [factual["records"][i] for i in cfg["source_case_indices"]]
    raw_pins(a.factual_source_root, facts)
    source = None
    if a.audit_source_root is not None:
        root = a.audit_source_root
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("closed counterexample source required")
        sp, dp = root / "complete.json", root / "declaration.json"
        source, decl = [json.loads(path.read_text()) for path in (sp, dp)]
        if (
            source["status"] != SOURCE
            or source["config"] != cfg
            or decl["config"] != cfg
            or source["source_head"] != a.producer_head
            or decl["source_head"] != a.producer_head
            or source["declaration_hash"] != file_hash(dp)
            or source["input_file_hashes"] != decl["input_file_hashes"]
            or source["training_hash"] != a.training_hash
            or source["factual_own_hash"] != a.factual_own_hash
        ):
            raise ValueError("exact counterexample producer required")
        verify(source, factual)
        for k, v in source["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting counterexample ancestry")
            pins[k] = v
        for path in (sp, dp):
            pins[str(path)] = file_hash(path)
        raw_pins(root, source["records"])
    source_pins(pins)
    check_pins(pins)
    setup, old, actor, courses, preflight = open_models(a, admitted)
    if any(r["course"] not in courses for r in facts):
        raise ValueError("all source courses must bind whole-bank preflight")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            config=cfg,
            source_head=a.expected_head,
            training_hash=a.training_hash,
            factual_own_hash=a.factual_own_hash,
            input_file_hashes=pins,
            preflight=preflight,
            goal_complete=False,
            hardware_authorized=False,
        ),
    )
    counts, records = dict(native_steps=0, actor_calls=0), []
    try:
        prior = None
        for index in range(cfg["new_worlds"]):
            fact = facts[index // 2]
            common = dict(
                output=a.output,
                index=index,
                counts=counts,
                emit_course_rows=False,
                sampling_seed=fact["sampling_seed"],
                sampling_profile="contact-v1",
                sampling_include_recovery=True,
                receiver_intent_mode=cfg["receiver_intent_mode"],
                receiver_role_ablation=True,
            )
            if source is None:
                from scripts.rsi_pass_transfer_trial import collect_trial

                row = collect_trial(setup, actor, fact["course"], **common)
            else:
                from scripts.rsi_pass_transfer_replay_trial import replay_trial

                claim = {
                    k: v for k, v in source["records"][index].items() if k != "source_case_index"
                }
                row, prior = replay_trial(
                    setup,
                    actor,
                    fact["course"],
                    claim,
                    raw_root=a.audit_source_root,
                    prior_initial=prior,
                    **common,
                )
            row["source_case_index"] = cfg["source_case_indices"][index // 2]
            records.append(row)
            write_once(a.output / f"record{index}.json", row)
        verify(dict(config=cfg, records=records, counts=counts), factual)
        if source is not None and (
            records != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
            or counts != source["counts"]
        ):
            raise ValueError("all own native/RNG results must match")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=SOURCE if source is None else OWN,
                config=cfg,
                source_head=a.expected_head,
                training_hash=a.training_hash,
                factual_own_hash=a.factual_own_hash,
                factual_source_hash=pins[str(a.factual_source_root / "complete.json")],
                input_file_hashes=pins,
                declaration_hash=file_hash(a.output / "declaration.json"),
                source_hash=None
                if source is None
                else file_hash(a.audit_source_root / "complete.json"),
                independent_whole_world_replay=source is not None,
                sampling_law_audit_passed=source is not None,
                records=records,
                counts=counts,
                factual_worlds_reused=len(cfg["source_case_indices"]),
                factual_worlds_reexecuted=0,
                actor_training_admitted=False,
                optimizer_updates=0,
                Fresh_verified=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
                team_game_verified=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                records=records,
                counts=counts,
                incomplete_world_steps_unknown=True,
                reserved_native_step_ceiling=cfg["maximum_native_steps"],
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
