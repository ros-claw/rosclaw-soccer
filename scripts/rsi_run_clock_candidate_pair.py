"""Execute each learned clock candidate with zero/full intervention and own replay."""

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(926051, 926052, 926053), required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.plan.is_symlink() or digest(a.plan) != a.plan_hash or a.output.is_symlink():
        raise ValueError("exact candidate evaluation plan required")
    plan = json.loads(a.plan.read_text())
    if (
        plan["protocol"] != "robonaldo.reference_clock_authority.v1"
        or plan["source_commit"] != a.expected_head
        or plan["reference_rates"] != [1.18]
        or plan["hardware_authorized"] is not False
        or [r["seed"] for r in plan["candidates"]] != [926051, 926052, 926053]
    ):
        raise ValueError("all three predeclared fast-clock candidates required")
    candidate = next(r for r in plan["candidates"] if r["seed"] == a.seed)
    training_path = Path(candidate["root"]) / "complete.json"
    if digest(training_path) != candidate["hash"]:
        raise ValueError("exact physically unqualified learned candidate required")
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(
            a.output / "declaration.json",
            dict(
                plan_hash=a.plan_hash,
                seed=a.seed,
                source_head=a.expected_head,
                observer_only=False,
                maximum_worker_executions=4,
                blind_retry=False,
                hardware_authorized=False,
            ),
        )
        results = []
        for fraction in (0.0, 1.0):
            tag = "zero" if fraction == 0 else "full"
            produced, audited = a.output / tag, a.output / (tag + "-manual")
            args = [
                sys.executable,
                "-m",
                "scripts.rsi_probe_live_neural_transition",
                "--clock-authority",
                "--contact-release",
                "--single-player-only",
                "--reference-rate=1.18",
                "--clock-plan",
                str(a.plan),
                "--clock-plan-hash",
                a.plan_hash,
                "--expected-head",
                a.expected_head,
                "--output",
                str(produced),
                "--torque-candidate-root",
                candidate["root"],
                "--torque-candidate-hash",
                candidate["hash"],
                "--torque-intervention-fraction=" + str(fraction),
            ]
            for option, key in (
                ("source", "source_root"),
                ("source-hash", "source_hash"),
                ("own", "own_root"),
                ("own-hash", "own_hash"),
                ("identity-reference", "identity_source_root"),
                ("identity-reference-hash", "identity_source_hash"),
                ("identity-own", "identity_own_root"),
                ("identity-own-hash", "identity_own_hash"),
                ("clock-parent-root", "clock_parent_root"),
                ("clock-parent-hash", "clock_parent_hash"),
                ("clock-parent-own-root", "clock_parent_own_root"),
                ("clock-parent-own-hash", "clock_parent_own_hash"),
                ("core-source-root", "core_root"),
                ("core-expected-head", "core_commit"),
            ):
                args += ["--" + option, plan[key]]
            if not produced.exists():
                subprocess.run(args, check=True)
            path = produced / "complete.json"
            if not path.is_file() or (produced / "failed.json").exists():
                raise ValueError("partial/failed candidate retained, no blind restart")
            report = json.loads(path.read_text())
            decl = json.loads((produced / "declaration.json").read_text())
            if (
                report["source_head"] != a.expected_head
                or len(report["records"]) != 16
                or decl["clock_plan_hash"] != a.plan_hash
                or decl["torque_candidate_hash"] != candidate["hash"]
                or decl["torque_intervention_fraction"] != fraction
            ):
                raise ValueError("completed candidate differs from commitment")
            if fraction == 0 and not all(
                all(r["zero_intervention_full_identity"].values()) for r in report["records"]
            ):
                raise ValueError("zero intervention changed actual fast parent")
            source_hash = digest(path)
            if not audited.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.rsi_audit_live_neural_transition",
                        "--source",
                        str(produced),
                        "--source-hash",
                        source_hash,
                        "--producer-head",
                        a.expected_head,
                        "--expected-head",
                        a.expected_head,
                        "--output",
                        str(audited),
                    ],
                    check=True,
                )
            own_path = audited / "complete.json"
            if not own_path.is_file() or (audited / "failed.json").exists():
                raise ValueError("partial/failed own replay retained, no blind restart")
            own = json.loads(own_path.read_text())
            if own["source_hash"] != source_hash or own["all_exact"] is not True:
                raise ValueError("entire exact own trajectory/neural episode required")
            result = dict(
                fraction=fraction,
                source_hash=source_hash,
                own_hash=digest(own_path),
                records=report["records"],
                source_root=str(produced),
                own_root=str(audited),
            )
            write_once(a.output / (tag + "-reviewed.json"), result)
            results.append(result)
        clean(a.expected_head)
        if digest(a.plan) != a.plan_hash:
            raise ValueError("evaluation plan changed")
        write_once(
            a.output / "complete.json",
            dict(
                status="FAST_CLOCK_LEARNED_CANDIDATE_PAIRED_NOT_PROMOTED",
                seed=a.seed,
                source_head=a.expected_head,
                plan_hash=a.plan_hash,
                results=results,
                optimizer_updates=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )


if __name__ == "__main__":
    main()
