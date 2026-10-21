"""All20 TRAIN contexts with an admitted motor; no tuning or Fresh data."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("plan-root", "student-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("plan-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--wait-for-student", action="store_true")
    p.add_argument("--first-enabled", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen exam source required")

    clean()
    if a.output.exists():
        raise ValueError("unused exam output required")
    deadline = time.monotonic() + 7200
    rp = a.student_root / "complete.json"
    while not rp.exists():
        clean()
        if (a.student_root / "failed.json").exists():
            raise ValueError("failed fit, no implicit retry")
        if not a.wait_for_student or time.monotonic() > deadline:
            raise TimeoutError("closed student required")
        time.sleep(10)
    from rosclaw_soccer.rsi.paid_ancestor_study import load_ancestor_plan

    plan_receipt, plan = load_ancestor_plan(a.plan_root, a.plan_hash)
    training = json.loads(rp.read_text())
    pins = dict(plan_receipt["input_file_hashes"])
    for path, h in training["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("conflicting evidence lineage")
        pins[path] = h
    cp = Path(training["checkpoint_path"])
    pins[str(cp)] = training["checkpoint_hash"]
    for f in (rp, a.plan_root / "complete.json", a.plan_root / "declaration.json"):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    schedule = (
        ("student", "student", "student") if a.first_enabled else ("base", "student", "student")
    )
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            plan_root=str(a.plan_root),
            plan_hash=a.plan_hash,
            student_root=str(a.student_root),
            student_hash=digest(rp),
            cases=plan["cases"],
            schedule=schedule,
            repeats=2,
            bank="train",
            Fresh_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
            **(dict(first_enabled=True) if a.first_enabled else {}),
        ),
    )
    bench = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_ancestor_study import ancestor_outcome
        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_standing_lifecycle import run_standing_lifecycle
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, plan["cases"])
        if scene != plan["body_scene_hash"] or bench.native_steps or bench.neural_rows:
            raise ValueError("full bank must preflight before any rollout")
        actor, _ = load_paid_student(
            policy,
            checkpoint=cp,
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=rp,
            training_receipt_hash=digest(rp),
            scene_hash=scene,
        )
        worlds = []
        for case in plan["cases"]:
            i = case["case"]
            replicas, raws = [], []
            for repeat in range(2):
                folder = a.output / f"case{i}-repeat{repeat}"
                folder.mkdir()
                artifacts, raw = [], {}

                def store(key, z, folder=folder, artifacts=artifacts, raw=raw):
                    f = folder / (key + ".npz")
                    if f.exists():
                        raise ValueError("raw overwrite forbidden")
                    np.savez_compressed(f, **z)
                    raw[key] = z
                    link = dict(raw_path=str(f.relative_to(a.output)), raw_hash=digest(f))
                    artifacts.append(link)
                    return link

                w = run_standing_lifecycle(
                    bench,
                    case,
                    actor,
                    variant=2,
                    skills=3,
                    store=store,
                    record_initial=True,
                    motor_schedule=schedule,
                )
                w.update(
                    case=i,
                    artifacts=artifacts,
                    quality_by_kick=[
                        bool(contact_rank([k["result"], k["result"]])[1] > 0) for k in w["kicks"]
                    ],
                )
                write_once(folder / "complete.json", w)
                replicas.append(w)
                raws.append(raw)
            if set(raws[0]) != set(raws[1]):
                raise ValueError("repeat phase coverage differs")
            for key in raws[0]:
                identical(raws[0][key], raws[1][key])
            worlds.append(dict(replicas[0], repeat_artifacts=replicas[1]["artifacts"]))
            print(
                json.dumps(dict(case=i, quality=w["quality_by_kick"], stable=w["stable"])),
                flush=True,
            )
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                plan_hash=a.plan_hash,
                student_hash=digest(rp),
                worlds=worlds,
                summary=ancestor_outcome(worlds),
                repeats_exact=True,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
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
                actual_native_steps=bench.native_steps if bench else 0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
