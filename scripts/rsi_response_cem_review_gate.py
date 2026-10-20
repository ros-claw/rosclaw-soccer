"""Fixed downstream review queue; never invents hypotheses or promotes a policy.

Run only as an After= dependency of the four already-started producer units.
No polling loop, no producer restart, no neural or physical execution here.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def review_decision(rows):
    if len(rows) != 16 or {
        (r["course"]["parent_index"], r["course"]["variant_index"]) for r in rows
    } != {(c, v) for c in range(4) for v in range(4)}:
        raise ValueError("all16 distinct original conditions required")
    for row in rows:
        for score in (row["score"], row["old_fixed_six_teacher_score_comparison_only"]):
            if any(
                type(score[k]) is not bool for k in ("qualified", "physical_prefix_and_suffix_safe")
            ):
                raise ValueError("typed producer-claimed response required")
    old = sum(r["old_fixed_six_teacher_score_comparison_only"]["qualified"] for r in rows)
    if old != 6:
        raise ValueError("original six-success teacher comparison required")
    qualified = sum(r["score"]["qualified"] for r in rows)
    regressions = sum(
        r["old_fixed_six_teacher_score_comparison_only"]["qualified"]
        and not r["score"]["qualified"]
        for r in rows
    )
    unsafe = sum(
        r["old_fixed_six_teacher_score_comparison_only"]["physical_prefix_and_suffix_safe"]
        and not r["score"]["physical_prefix_and_suffix_safe"]
        for r in rows
    )
    allowed = qualified >= 7 and regressions == unsafe == 0
    return dict(
        status="REVIEW_CPU_AUTHORIZED" if allowed else "CEM_FAMILY_REJECTED_STOP",
        producer_claimed_qualified=qualified,
        original_teacher_qualified=old,
        regressions=regressions,
        added_unsafe=unsafe,
        independent_review_required=True,
        metadata_is_physical_qualification=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=("response_cem", "matched_entry"), default="response_cem"
    )
    for name in ("reviewer-source", "world", "motion-csv", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-reviewer-head",
        "expected-producer-head",
        "expected-bank-hash",
        "review-unit-prefix",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--producer", nargs=2, action="append", required=True, metavar=("UNIT", "ROOT")
    )
    parser.add_argument("--previous-producer", type=Path, action="append", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("sealed gate source and unused receipt directory required")
    for name in (Path(__file__).name, "rsi_native_context_recovery_common.py"):
        if (
            subprocess.check_output(["git", "show", f"{head}:scripts/{name}"])
            != (Path.cwd() / "scripts" / name).read_bytes()
        ):
            raise ValueError("gate bootstrap differs from sealed source")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,90}", args.review_unit_prefix):
        raise ValueError("bounded review-only unit prefix required")
    if (
        subprocess.check_output(
            ["git", "-C", str(args.reviewer_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != args.expected_reviewer_head
    ):
        raise ValueError("sealed reviewer checkout required")
    subprocess.run(
        [
            "git",
            "-C",
            str(args.reviewer_source),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "src",
            "scripts",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    pins = {}
    for folder in (
        Path.cwd() / "scripts",
        args.reviewer_source / "src",
        args.reviewer_source / "scripts",
        args.core_source,
    ):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    complete_rows, roots, clusters, units = [], [], set(), set()
    original = {}
    for root in args.previous_producer:
        data = json.loads((root / "complete.json").read_text())
        if data["cluster"] in original:
            raise ValueError("unique original comparison clusters required")
        original[data["cluster"]] = (root, data)
    if set(original) != set(range(4)):
        raise ValueError("all four original teacher clusters required")
    for unit, path in args.producer:
        root = Path(path)
        terminal = dict(
            line.split("=", 1)
            for line in subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "-p",
                    "MainPID",
                    "-p",
                    "Result",
                    "-p",
                    "SubState",
                    "-p",
                    "ExecMainStatus",
                ],
                text=True,
            ).splitlines()
        )
        if terminal != dict(MainPID="0", Result="success", SubState="exited", ExecMainStatus="0"):
            raise ValueError("all producers must be terminal; never restart them")
        declaration = json.loads((root / "declaration.json").read_text())
        complete = json.loads((root / "complete.json").read_text())
        c = complete["cluster"]
        if (
            c not in range(4)
            or c in clusters
            or unit in units
            or any(root.rglob("failure.json"))
            or declaration["schema"]
            != (
                "soccer.rsi.bounded_response_cem16.v1"
                if args.study == "response_cem"
                else "soccer.rsi.matched_entry_bridge16.v1"
            )
            or declaration["source_head"] != args.expected_producer_head
            or complete["source_head"] != args.expected_producer_head
            or declaration["bank_hash"] != args.expected_bank_hash
            or complete["bank_hash"] != args.expected_bank_hash
            or len(complete["results"]) != 4
            or complete["actual_native_steps"] != 12000
            or (args.study == "response_cem" and complete["forecast_native_steps"] != 336000)
            or complete["neural_optimizer_updates"] != 0
        ):
            raise ValueError("exact closed CEM cluster commitment required")
        check_pins(declaration["input_file_hashes"])
        pins.update(declaration["input_file_hashes"])
        oldroot, old = original[c]
        if pins.get(str(oldroot / "complete.json")) != digest(oldroot / "complete.json"):
            raise ValueError("original teacher root not bound")
        for v, row in enumerate(complete["results"]):
            folder = root / f"variant{v}"
            if (
                row != json.loads((folder / "complete.json").read_text())
                or row["course"] != old["results"][v]["course"]
                or row["course"]["parent_index"] != c
                or row["course"]["variant_index"] != v
                or (
                    len(row["candidates"]) != 30
                    if args.study == "response_cem"
                    else not 6 <= len(row["candidates"]) <= 12
                )
                or row["old_fixed_six_teacher_score_comparison_only"]
                != old["results"][v]["results"][1]["score"]
            ):
                raise ValueError("exact original condition/comparison required")
            if (
                row["forecast_native_steps"] != 2800 * len(row["candidates"])
                or row["actual_native_steps"] != 3000
            ):
                raise ValueError("complete per-case forecast/actual accounting required")
            for receipt in [row] + row["candidates"]:
                p = (folder / receipt["artifact"]).resolve()
                if not p.is_relative_to(folder.resolve()) or digest(p) != receipt["artifact_hash"]:
                    raise ValueError("producer raw output identity differs")
            complete_rows.append(row)
        if complete["forecast_native_steps"] != sum(
            row["forecast_native_steps"] for row in complete["results"]
        ):
            raise ValueError("complete cluster forecast sum differs")
        for p in (args.world, args.motion_csv):
            if pins.get(str(p)) != digest(p):
                raise ValueError("bound original assets required")
        pins.update({str(p): digest(p) for p in root.rglob("*") if p.is_file()})
        roots.append((c, unit, root, digest(root / "complete.json")))
        clusters.add(c)
        units.add(unit)
    if clusters != set(range(4)) or len(roots) != 4:
        raise ValueError("exact four closed producers required")
    decision = review_decision(complete_rows)
    decision["study"] = args.study
    if args.study == "matched_entry" and decision["status"] == "CEM_FAMILY_REJECTED_STOP":
        decision["status"] = "MATCHED_ENTRY_FAMILY_REJECTED_STOP"
    check_pins(pins)
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(source_head=head, input_file_hashes=pins, metadata_gate_only=True),
    )
    write_once(args.output / "decision.json", decision)
    launches = []
    if decision["status"] == "REVIEW_CPU_AUTHORIZED":
        for c, _, root, expected_hash in sorted(roots):
            unit = f"{args.review_unit_prefix}-c{c}"
            output = args.output / f"review-c{c}"
            command = [
                "systemd-run",
                "--user",
                "--unit=" + unit,
                "--no-block",
                "--property=Type=oneshot",
                "--property=RemainAfterExit=yes",
                "--property=MemoryMax=8G",
                "--property=NoNewPrivileges=yes",
                "--working-directory=" + str(args.reviewer_source),
            ]
            for k, v in dict(
                PYTHONPATH="src:.:" + str(args.core_source),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                CUDA_VISIBLE_DEVICES="",
            ).items():
                command.append("--setenv=" + k + "=" + v)
            import sys

            command += [
                sys.executable,
                "scripts/rsi_verify_response_cem.py",
                "--study",
                args.study,
                "--producer",
                str(root),
                "--producer-unit",
                dict((i, u) for i, u, _, _ in roots)[c],
                "--expected-head",
                args.expected_reviewer_head,
                "--expected-producer-head",
                args.expected_producer_head,
                "--expected-producer-hash",
                expected_hash,
                "--world",
                str(args.world),
                "--motion-csv",
                str(args.motion_csv),
                "--core-source",
                str(args.core_source),
                "--output",
                str(output),
            ]
            for p in args.previous_producer:
                command += ["--previous-producer", str(p)]
            subprocess.run(command, check=True)
            launches.append(
                dict(
                    unit=unit + ".service", output=str(output), producer_complete_hash=expected_hash
                )
            )
            write_once(args.output / f"launch-c{c}.json", launches[-1])
    check_pins(pins)
    write_once(
        args.output / "complete.json",
        dict(
            decision,
            launched_reviews=launches,
            review_jobs_completed_or_verified=False,
            new_native_steps_by_gate=0,
            new_neural_calls_by_gate=0,
            scientific_successor_generation=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(dict(decision, launched_reviews=launches), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
