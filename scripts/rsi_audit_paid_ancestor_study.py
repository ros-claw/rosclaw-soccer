"""Independently check own-NN, all phase physics and shared counterfactual prefixes."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "plan-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("plan-hash", "producer-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("frozen audit source required")

    clean()
    if a.output.exists():
        raise ValueError("unused audit output required")
    deadline = time.monotonic() + 7200
    while not (a.root / "complete.json").exists():
        clean()
        if (a.root / "failed.json").exists():
            raise ValueError("failed experiment, no retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("closed shard required")
        time.sleep(10)
    clean()
    from rosclaw_soccer.rsi.paid_ancestor_study import (
        SCHEDULES,
        ancestor_comparison,
        load_ancestor_actor,
        load_ancestor_plan,
    )

    _, plan = load_ancestor_plan(a.plan_root, a.plan_hash)
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    shard = d["shard"]
    if (
        (a.root / "failed.json").exists()
        or type(shard) is not int
        or shard not in range(4)
        or r["status"] != "PAID_TRAIN_ANCESTOR_COUNTERFACTUAL_SHARD_PENDING_AUDIT"
        or r["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or r["declaration_hash"] != digest(dp)
        or r["plan_hash"] != a.plan_hash
        or d["plan_hash"] != a.plan_hash
        or r["shard"] != shard
        or d["cases"] != plan["cases"][shard * 5 : (shard + 1) * 5]
        or [g["case"] for g in r["groups"]] != [c["case"] for c in d["cases"]]
        or d["schedules"] != {k: list(v) for k, v in SCHEDULES.items()}
        or d["repeats"] != 2
        or r["all_repeats_and_shared_prefixes_exact"] is not True
        or any(o["bank"] != "train" or o["Fresh_fit_rows"] != 0 for o in (r, d))
        or any(
            o[k] is not False
            for o in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("exact whole-shard TRAIN counterfactual contract required")
    pins = dict(r["input_file_hashes"])
    for f in (rp, dp, a.plan_root / "complete.json", a.plan_root / "declaration.json"):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_hash=digest(rp),
            plan_hash=a.plan_hash,
            shard=shard,
            own_native_integration=True,
            own_neural_recompute=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, own_neural=0, worlds=0)
    replay = None
    try:
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_lifecycle_bundle import (
            audit_saved_lifecycle_replica,
            require_same_prefix,
        )
        from rosclaw_soccer.rsi.paid_native_phase_replay import PaidNativePhaseReplay
        from rosclaw_soccer.rsi.paid_neural_stand_audit import StandingObservationAuditor
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, plan["cases"])
        observer = StandingObservationAuditor(auditor)
        policy = next(Path(p) for p in pins if p.endswith("/ckp/policy_30000.onnx"))
        actor = load_ancestor_actor(policy, plan)
        replay = PaidNativePhaseReplay(auditor)
        groups = []
        phase_records = []

        def inference():
            counts["own_neural"] += 1

        for group, case in zip(r["groups"], d["cases"], strict=True):
            checked = dict(case=case["case"], arms={})
            first_raw = {}
            if (
                set(group["arms"]) != set(SCHEDULES)
                or group["shared_actual_prefix_exact"] is not True
            ):
                raise ValueError("all modes required")
            for mode, schedule in SCHEDULES.items():
                world = group["arms"][mode]
                results = []
                raws = []
                for repeat in range(2):
                    result, raw, phases = audit_saved_lifecycle_replica(
                        a.root / mode,
                        world,
                        repeat,
                        case,
                        auditor=auditor,
                        observer=observer,
                        actor=actor,
                        replay=replay,
                        motion=motions[case["source"]],
                        schedule=schedule,
                        pins=pins,
                        count_inference=inference,
                    )
                    for k in ("native", "neural"):
                        counts[k] += result[k]
                    counts["worlds"] += 1
                    results.append(result)
                    raws.append(raw)
                    phase_records.extend(
                        dict(case=case["case"], mode=mode, repeat=repeat, **p) for p in phases
                    )
                if results[0] != results[1] or set(raws[0]) != set(raws[1]):
                    raise ValueError("counterfactual repeated worlds differ")
                for key in raws[0]:
                    identical(raws[0][key], raws[1][key])
                checked["arms"][mode] = results[0]
                first_raw[mode] = raws[0]
            require_same_prefix(first_raw["grown"], first_raw["ancestor_third"], before_kick=2)
            require_same_prefix(first_raw["grown"], first_raw["r2_all"], before_kick=1)
            groups.append(checked)
            print(
                json.dumps(
                    dict(case=case["case"], status="ALL_MODES_NN_PHYSICS_AND_PREFIX_CHECKED")
                ),
                flush=True,
            )
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_rows"]
            or counts["own_neural"] != counts["neural"]
            or counts["worlds"] != 30
            or replay.native_steps != counts["native"]
            or any(str(f) not in pins for f in a.root.rglob("*.npz"))
            or ancestor_comparison(groups) != r["comparison"]
        ):
            raise ValueError("complete own physics/NN/accounting differs")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_TRAIN_ANCESTOR_COUNTERFACTUAL_OWN_NN_AND_PHYSICS_AUDIT_PASS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=digest(rp),
                plan_hash=a.plan_hash,
                shard=shard,
                counts=counts,
                groups=groups,
                comparison=ancestor_comparison(groups),
                own_native_steps=replay.native_steps,
                own_trajectory_neural_rows=counts["own_neural"],
                phases=phase_records,
                first_kick_replayed=True,
                independent_whole_world_replay=False,
                shared_actual_prefixes_exact=True,
                bank="train",
                Fresh_fit_rows=0,
                optimizer_updates=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                own_native_steps=replay.native_steps if replay else 0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
