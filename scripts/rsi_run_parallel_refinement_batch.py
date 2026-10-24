"""Finite spawn-parallel native MuJoCo experience, with independent own replay.

Model/source admission and ALL course preflight precede physical workers. No
task retry, source-action replay, GPU approximation or optimizer update here.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import pickle
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("experience-root", "own-root", "training-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("own-hash", "training-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(25, 26), required=True)
    p.add_argument("--seeds-per-initial", type=int, choices=(4, 32), required=True)
    p.add_argument("--workers", type=int, choices=(1, 2, 4), required=True)
    p.add_argument("--full-recovery", action="store_true")
    p.add_argument("--receiver-role-ablation", action="store_true")
    p.add_argument("--projected-capture", action="store_true")
    p.add_argument("--projected-pilot-own-root", type=Path)
    p.add_argument("--projected-pilot-producer-head")
    p.add_argument("--audit-source-root", type=Path)
    p.add_argument("--producer-head")
    p.add_argument("--calibration-own-root", type=Path)
    p.add_argument("--calibration-producer-head")
    a = p.parse_args()
    if (
        a.projected_capture != (a.projected_pilot_own_root is not None)
        or a.projected_capture != (a.projected_pilot_producer_head is not None)
        or (a.projected_capture and not a.receiver_role_ablation)
    ):
        p.error("projected role requires exact closed pilot root/head and explicit role ablation")
    if a.receiver_role_ablation and (not a.full_recovery or a.state_id != 26):
        p.error("separate full-recovery state26 receiver-role protocol required")
    if (a.audit_source_root is None) != (a.producer_head is None):
        p.error("own audit requires exact source root/head")
    if (a.calibration_own_root is None) != (a.calibration_producer_head is None) or (
        a.seeds_per_initial == 32 and a.calibration_own_root is None
    ):
        p.error("large batches require a predeclared closed own-native calibration")
    from rosclaw_soccer.rsi.parallel_refinement_batch import (
        FULL_OWN,
        FULL_SOURCE,
        PROJECTED_OWN,
        PROJECTED_SOURCE,
        ROLE_OWN,
        ROLE_SOURCE,
        parallel_config,
        verify_parallel_batch,
    )
    from rosclaw_soccer.rsi.refinement_continual_checkpoint import KIND, STATUS
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_learned_refinement_common import admit, open_models
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_parallel_refinement_worker import bootstrap_probe, initialize, run_pair

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only parallel output required")
    cfg = parallel_config(
        a.state_id,
        a.seeds_per_initial,
        full_recovery=a.full_recovery,
        receiver_role_ablation=a.receiver_role_ablation,
        projected_capture=a.projected_capture,
    )
    source_status = (
        FULL_SOURCE
        if a.full_recovery
        else "GEN1_PARALLEL_OWN_POLICY_BATCH_COLLECTED_PENDING_NATIVE_LAW"
    )
    own_status = (
        FULL_OWN if a.full_recovery else "GEN1_PARALLEL_OWN_POLICY_BATCH_ALL_NATIVE_LAW_MATCH"
    )
    if a.receiver_role_ablation:
        source_status, own_status = ROLE_SOURCE, ROLE_OWN
    if a.projected_capture:
        source_status, own_status = PROJECTED_SOURCE, PROJECTED_OWN
    admitted = admit(a)
    pins = admitted[1]
    if a.projected_capture:
        from rosclaw_soccer.rsi.receiver_counterexample import projected_config, verify
        from scripts.rsi_native_fresh_common import receipt

        root = a.projected_pilot_own_root
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("closed projected pilot own evidence required")
        rp = root / "complete.json"
        pilot = json.loads(rp.read_text())
        if (
            pilot["status"] != "PROJECTED_RECEIVER6_ALL_NATIVE_LAW_MATCH"
            or pilot["config"] != projected_config()
            or pilot["source_head"] != a.projected_pilot_producer_head
            or pilot["training_hash"] != a.training_hash
            or pilot["independent_whole_world_replay"] is not True
            or pilot["sampling_law_audit_passed"] is not True
            or len(pilot["records"]) != 6
            or any(
                not r["outcome"]["B_legal_functional_receive_before_deadline"]
                for r in pilot["records"]
            )
            or any(
                not r["outcome"]["combined_precision_and_functional_receive"]
                for r in pilot["records"][:4]
            )
        ):
            raise ValueError(
                "all6 repaired functional receptions and4 retained strict passes required"
            )
        for key, value in pilot["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting projected pilot ancestry")
            pins[key] = value
        pins[str(rp)] = file_hash(rp)
        _, factual = receipt(pins, pilot["factual_source_hash"])
        _, producer = receipt(pins, pilot["source_hash"])
        verify(pilot, factual)
        if (
            pilot["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in producer["records"]]
            or producer["config"] != projected_config()
        ):
            raise ValueError("exact all6 independent producer histories required")
    training = json.loads((a.training_root / "complete.json").read_text())
    if a.full_recovery:
        from rosclaw_soccer.rsi.continuing_refinement_contract import (
            KIND as FULL_BEHAVIOR_KIND,
        )
        from rosclaw_soccer.rsi.continuing_refinement_contract import (
            STATUS as FULL_BEHAVIOR_STATUS,
        )

        expected_kind, expected_status, steps = FULL_BEHAVIOR_KIND, FULL_BEHAVIOR_STATUS, 12
        if training["config"]["generation_index"] != 2:
            raise ValueError("full recovery pilot requires the exact generation2 behavior")
    else:
        expected_kind, expected_status, steps = KIND, STATUS, 8
    if (
        training["checkpoint_kind"] != expected_kind
        or training["status"] != expected_status
        or training["cumulative_actor_steps"] != steps
        or training["cumulative_critic_steps"] != steps
    ):
        raise ValueError("exact fully trained versioned behavior required for new rollouts")
    if a.calibration_own_root is not None:
        cr = a.calibration_own_root
        cp = cr / "complete.json"
        if cr.is_symlink() or (cr / "failed.json").exists():
            raise ValueError("closed ordinary parallel calibration required")
        ch = file_hash(cp)
        cal = json.loads(cp.read_text())
        if (
            cal["status"] != own_status
            or cal["source_head"] != a.calibration_producer_head
            or cal["training_hash"] != a.training_hash
            or cal["config"]
            != parallel_config(
                a.state_id,
                4,
                full_recovery=a.full_recovery,
                receiver_role_ablation=a.receiver_role_ablation,
                projected_capture=a.projected_capture,
            )
            or cal["independent_whole_world_replay"] is not True
            or cal["sampling_law_audit_passed"] is not True
        ):
            raise ValueError("all12 same-policy native/RNG calibration worlds required")
        verify_parallel_batch(cal, [cal["records"][i]["course"] for i in (0, 4, 8)])
        for k, v in cal["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting calibration ancestry")
            pins[k] = v
        pins[str(cp)] = ch
    source, sh = None, None
    if a.audit_source_root is not None:
        root = a.audit_source_root
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("ordinary complete source required")
        sp, dp = root / "complete.json", root / "declaration.json"
        sh = file_hash(sp)
        source, decl = [json.loads(x.read_text()) for x in (sp, dp)]
        if (
            source["status"] != source_status
            or source["config"] != cfg
            or decl["config"] != cfg
            or source["source_head"] != a.producer_head
            or decl["source_head"] != a.producer_head
            or source["training_hash"] != a.training_hash
            or decl["training_hash"] != a.training_hash
            or source["declaration_hash"] != file_hash(dp)
            or source["input_file_hashes"] != decl["input_file_hashes"]
            or source["optimizer_updates"] != 0
            or any(
                source[k] is not False
                for k in (
                    "hardware_authorized",
                    "promotion_authorized",
                    "goal_complete",
                    "Fresh_verified",
                    "actor_training_admitted",
                )
            )
        ):
            raise ValueError("exact complete generation1 own-policy source required")
        for k, v in source["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting parallel ancestry")
            pins[k] = v
        for path in (sp, dp):
            pins[str(path)] = file_hash(path)
        for r in source["records"]:
            if set(r["artifacts"]) != {"initial", "physical", "A", "B", "sampling"}:
                raise ValueError("all own-state native and sampling primitives required")
            for art in r["artifacts"].values():
                path = root / art["path"]
                if (
                    path.resolve().parent != (root / f"trial{r['index']}").resolve()
                    or file_hash(path) != art["hash"]
                ):
                    raise ValueError("confined exact source primitives required")
                pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    # Exact runner __mp_main__ import and designated worker pickle, no physics.
    code = (
        "import runpy,pickle,sys\n"
        "runpy.run_module('scripts.rsi_run_parallel_refinement_batch',"
        "run_name='__mp_main__',alter_sys=True)\n"
        "import scripts.rsi_parallel_refinement_worker as w\n"
        "assert sys.modules[w.__name__] is w\n"
        "assert pickle.loads(pickle.dumps(w.run_pair)) is w.run_pair\n"
        "assert not w.bootstrap_probe()['model_allocated']\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    if pickle.loads(pickle.dumps(run_pair)) is not run_pair:
        raise ValueError("worker pickle identity differs")
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as bootstrap:
        proof = bootstrap.submit(bootstrap_probe).result()
    if (
        not proof["module_registered"]
        or not proof["worker_pickle_roundtrip"]
        or proof["model_allocated"]
    ):
        raise ValueError("spawn bootstrap failed before physical allocation")
    setup, old, actor, courses, preflight = open_models(a, admitted)
    if source is not None:
        if preflight != source["preflight"]:
            raise ValueError("all independent course preflight must match")
        verify_parallel_batch(source, courses)
    del setup, old, actor
    # Exact initializer capsule consists of JSON/path metadata, never live
    # MuJoCo/Torch objects, and must survive a real pickle roundtrip.
    roundtrip = pickle.loads(pickle.dumps((a, admitted, cfg, source)))
    if roundtrip != (a, admitted, cfg, source):
        raise ValueError("actual initializer metadata capsule pickle differs")
    del roundtrip
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            config=cfg,
            source_head=a.expected_head,
            training_hash=a.training_hash,
            input_file_hashes=pins,
            workers=a.workers,
            bootstrap=proof,
            bootstrap_physics_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    write_once(a.output / "preflight.json", preflight)
    records, counts, errors, owners = [], dict(native_steps=0, actor_calls=0), [], []
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=a.workers,
            mp_context=context,
            initializer=initialize,
            initargs=(a, admitted, cfg, source),
        ) as pool:
            futures = {pool.submit(run_pair, i): i for i in range(cfg["worlds"] // 2)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    row = future.result()
                    if row["pair"] != futures[future]:
                        raise ValueError("worker returned wrong pair identity")
                    records.extend(row["records"])
                    owners.append(dict(pair=row["pair"], pid=row["pid"]))
                    for k in counts:
                        counts[k] += row["counts"][k]
                except Exception as error:
                    # Drain all other declared work. No implicit resubmission.
                    errors.append(dict(pair=futures[future], error=repr(error)))
        records.sort(key=lambda r: r["index"])
        if errors:
            raise ValueError("failed parallel tasks preserved; batch cannot train")
        verify_parallel_batch(dict(config=cfg, records=records, counts=counts), courses)
        if source is not None and (
            counts != source["counts"]
            or records
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
        ):
            raise ValueError("all independent own native/RNG histories must match")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=source_status if source is None else own_status,
                config=cfg,
                source_head=a.expected_head,
                training_hash=a.training_hash,
                input_file_hashes=pins,
                declaration_hash=file_hash(a.output / "declaration.json"),
                preflight=preflight,
                source_hash=sh,
                records=records,
                counts=counts,
                worker_owners=sorted(owners, key=lambda x: x["pair"]),
                requested_workers=a.workers,
                participating_worker_pids=sorted({x["pid"] for x in owners}),
                independent_whole_world_replay=source is not None,
                sampling_law_audit_passed=source is not None,
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
                task_errors=errors,
                records=records,
                known_completed_counts=counts,
                actual_steps_in_incomplete_worlds_unknown=True,
                total_reserved_native_step_ceiling=cfg["maximum_native_steps"],
                automatic_retries=0,
                optimizer_updates=0,
                goal_complete=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
