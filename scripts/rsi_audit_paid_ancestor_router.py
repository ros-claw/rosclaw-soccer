"""Own network and all-phase native integration for complete TRAIN20 exams."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("producer-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen audit source required")

    clean()
    if a.output.exists():
        raise ValueError("unused audit output required")
    deadline = time.monotonic() + 7200
    while not (a.root / "complete.json").exists():
        clean()
        if (a.root / "failed.json").exists():
            raise ValueError("failed exam, no retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("closed exam required")
        time.sleep(10)
    from rosclaw_soccer.rsi.paid_ancestor_study import load_ancestor_plan

    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    _, plan = load_ancestor_plan(Path(d["plan_root"]), d["plan_hash"])
    schedule = ("base", "student", "student")
    if (
        (a.root / "failed.json").exists()
        or r["status"] != "PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT"
        or any(o["source_head"] != a.producer_head for o in (r, d))
        or r["declaration_hash"] != digest(dp)
        or r["plan_hash"] != d["plan_hash"]
        or r["student_hash"] != d["student_hash"]
        or d["cases"] != plan["cases"]
        or [w["case"] for w in r["worlds"]] != list(range(20))
        or d["schedule"] != list(schedule)
        or d["repeats"] != 2
        or r["repeats_exact"] is not True
        or any(o["bank"] != "train" or o["Fresh_fit_rows"] != 0 for o in (r, d))
        or any(
            o[k] is not False
            for o in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete committed TRAIN20 motor experiment required")
    srp = Path(d["student_root"]) / "complete.json"
    if digest(srp) != d["student_hash"]:
        raise ValueError("student changed")
    student = json.loads(srp.read_text())
    pins = dict(r["input_file_hashes"])
    for f in (rp, dp):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_hash=digest(rp),
            schedule=schedule,
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

        from rosclaw_soccer.rsi.paid_ancestor_study import ancestor_outcome
        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_lifecycle_bundle import audit_saved_lifecycle_replica
        from rosclaw_soccer.rsi.paid_native_phase_replay import PaidNativePhaseReplay
        from rosclaw_soccer.rsi.paid_neural_stand_audit import StandingObservationAuditor
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, plan["cases"])
        observer = StandingObservationAuditor(auditor)
        policy = next(Path(p) for p in pins if p.endswith("/ckp/policy_30000.onnx"))
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(student["checkpoint_path"]),
            checkpoint_hash=student["checkpoint_hash"],
            training_receipt=srp,
            training_receipt_hash=d["student_hash"],
            scene_hash=plan["body_scene_hash"],
        )
        replay = PaidNativePhaseReplay(auditor)
        worlds, phases = [], []

        def inference():
            counts["own_neural"] += 1

        for world, case in zip(r["worlds"], plan["cases"], strict=True):
            results, raws = [], []
            for repeat in range(2):
                checked, raw, records = audit_saved_lifecycle_replica(
                    a.root,
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
                for key in ("native", "neural"):
                    counts[key] += checked[key]
                counts["worlds"] += 1
                results.append(checked)
                raws.append(raw)
                phases.extend(dict(case=case["case"], repeat=repeat, **v) for v in records)
            if results[0] != results[1] or set(raws[0]) != set(raws[1]):
                raise ValueError("independent repeated-world audit differs")
            for key in raws[0]:
                identical(raws[0][key], raws[1][key])
            worlds.append(results[0])
            print(
                json.dumps(dict(case=case["case"], status="OWN_NN_ALL_PHASES_CHECKED")), flush=True
            )
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["native"] != replay.native_steps
            or counts["neural"] != r["actual_neural_rows"]
            or counts["own_neural"] != counts["neural"]
            or counts["worlds"] != 40
            or any(str(f) not in pins for f in a.root.rglob("*.npz"))
            or ancestor_outcome(worlds) != r["summary"]
        ):
            raise ValueError("all-world native/NN/accounting differs")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_ROUTER_TRAIN20_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=digest(rp),
                student_hash=d["student_hash"],
                counts=counts,
                worlds=worlds,
                phases=phases,
                summary=ancestor_outcome(worlds),
                own_native_steps=replay.native_steps,
                own_trajectory_neural_rows=counts["own_neural"],
                first_kick_replayed=True,
                independent_whole_world_replay=False,
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
