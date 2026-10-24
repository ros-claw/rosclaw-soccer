"""Bounded SIM succession: identity, own replay, then each declared clock rate.

This executes real workers, unlike the read-only loop observer. It does not
invent follow-up experiments, certify skill, or blindly restart failed runs.
"""

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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.plan.is_symlink() or digest(a.plan) != a.plan_hash or a.output.is_symlink():
        raise ValueError("ordinary exact campaign plan and evidence directory required")
    plan = json.loads(a.plan.read_text())
    if (
        plan["protocol"] != "robonaldo.reference_clock_authority.v1"
        or plan["source_commit"] != a.expected_head
        or plan["reference_rates"] != [1.0, 1.06, 1.12, 1.18]
        or plan["hardware_authorized"] is not False
    ):
        raise ValueError("bounded predeclared identity-first SIM family required")
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(
            a.output / "declaration.json",
            dict(
                plan_hash=a.plan_hash,
                source_head=a.expected_head,
                hardware_authorized=False,
                observer_only=False,
                maximum_worker_executions=8,
                blind_retry=False,
            ),
        )
        results = []
        for index, rate in enumerate(plan["reference_rates"]):
            produced, audited = a.output / f"rate-{index}", a.output / f"rate-{index}-manual"
            producer = [
                sys.executable,
                "-m",
                "scripts.rsi_probe_live_neural_transition",
                "--clock-authority",
                "--contact-release",
                "--single-player-only",
                "--reference-rate=" + str(rate),
                "--clock-plan",
                str(a.plan),
                "--clock-plan-hash",
                a.plan_hash,
                "--expected-head",
                a.expected_head,
                "--output",
                str(produced),
                "--core-source-root",
                plan["core_root"],
                "--core-expected-head",
                plan["core_commit"],
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
            ):
                producer += ["--" + option, plan[key]]
            if not produced.exists():
                subprocess.run(producer, check=True)
            complete = produced / "complete.json"
            if not complete.is_file() or (produced / "failed.json").exists():
                raise ValueError("partial or failed producer retained; no blind restart")
            report = json.loads(complete.read_text())
            declaration = json.loads((produced / "declaration.json").read_text())
            if (
                report["source_head"] != a.expected_head
                or declaration["clock_plan_hash"] != a.plan_hash
                or declaration["reference_rate"] != rate
                or len(report["records"]) != 4
            ):
                raise ValueError("completed producer differs from exact campaign commitment")
            if rate == 1.0 and not all(
                row["zero_intervention_full_identity"]
                and all(row["zero_intervention_full_identity"].values())
                for row in report["records"]
            ):
                raise ValueError("identity failed: accelerated successors prohibited")
            source_hash = digest(complete)
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
                raise ValueError("partial or failed own replay retained; no blind restart")
            own = json.loads(own_path.read_text())
            if own["source_hash"] != source_hash or own["all_exact"] is not True:
                raise ValueError("exact independent replay required before next stage")
            result = dict(
                reference_rate=rate,
                source_hash=source_hash,
                own_hash=digest(own_path),
                records=report["records"],
                source_root=str(produced),
                own_root=str(audited),
            )
            write_once(a.output / f"stage-{index}.json", result)
            results.append(result)
        clean(a.expected_head)
        if digest(a.plan) != a.plan_hash:
            raise ValueError("campaign plan changed during execution")
        write_once(
            a.output / "complete.json",
            dict(
                status="REFERENCE_CLOCK_FAMILY_REPLAYED_NOT_PROMOTED",
                results=results,
                source_head=a.expected_head,
                plan_hash=a.plan_hash,
                optimizer_updates=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )


if __name__ == "__main__":
    main()
