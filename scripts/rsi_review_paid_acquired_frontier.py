"""Compare audited student's each kick/case to R2 AND its learned parent.

Read-only evidence arithmetic, no new physics, no candidate selection or promotion.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate-root",
        "candidate-audit-root",
        "parent-root",
        "parent-audit-root",
        "r2-root",
        "r2-audit-root",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("immutable clean reviewer source required")

    clean()
    if a.output.exists():
        raise ValueError("unused review output and immutable source required")
    roots = [
        a.candidate_root,
        a.candidate_audit_root,
        a.parent_root,
        a.parent_audit_root,
        a.r2_root,
        a.r2_audit_root,
    ]
    deadline = time.monotonic() + 7200
    while not all((r / "complete.json").exists() for r in roots):
        clean()
        if any((r / "failed.json").exists() for r in roots):
            raise ValueError("upstream failed; preserve and diagnose, no review or physics retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("complete physics and audit receipts required")
        time.sleep(10)
    if any((r / "failed.json").exists() for r in roots):
        raise ValueError("failed evidence is not a completed experiment")
    clean()
    from rosclaw_soccer.rsi.paid_skill_frontier import compare_skill_frontier
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate
    from rosclaw_soccer.rsi.paid_student_evidence import qualify_student_worlds

    pins = {}

    def student(root, audit_root):
        rp, dp, ap = root / "complete.json", root / "declaration.json", audit_root / "complete.json"
        r, d, review = [json.loads(path.read_text()) for path in (rp, dp, ap)]
        qualify_student_worlds(
            r, d, review, report_path=str(rp), report_hash=digest(rp), declaration_hash=digest(dp)
        )
        for path, h in {
            **review["input_file_hashes"],
            str(rp): digest(rp),
            str(dp): digest(dp),
            str(ap): digest(ap),
        }.items():
            if path in pins and pins[path] != h:
                raise ValueError("candidate and parent lineage conflict")
            pins[path] = h
        return r, d

    candidate, cd = student(a.candidate_root, a.candidate_audit_root)
    parent, pd = student(a.parent_root, a.parent_audit_root)
    standing_gate(a.r2_root, a.r2_audit_root, variant=1, bank=cd["bank"])
    baseline = json.loads((a.r2_root / "complete.json").read_text())
    bd = json.loads((a.r2_root / "declaration.json").read_text())
    training_path = Path(cd["student_root"]) / "complete.json"
    training = json.loads(training_path.read_text())
    if (
        digest(training_path) != cd["student_hash"]
        or training["parent_hash"] != pd["student_hash"]
        or cd["cases"] != pd["cases"]
        or cd["cases"] != bd["cases"]
        or Path(cd["source_root"]).resolve() != a.r2_root.resolve()
        or Path(pd["source_root"]).resolve() != a.r2_root.resolve()
    ):
        raise ValueError("exact same cases, base world and frozen learned parent required")
    for path in (
        training_path,
        a.r2_root / "complete.json",
        a.r2_root / "declaration.json",
        a.r2_audit_root / "complete.json",
    ):
        pins[str(path)] = digest(path)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    comparisons = dict(
        baseline=compare_skill_frontier(baseline, candidate),
        acquired_parent=compare_skill_frontier(parent, candidate),
    )
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="PAID_AUDITED_BASELINE_AND_ACQUIRED_FRONTIER_REVIEW_CLOSED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            candidate_hash=digest(a.candidate_root / "complete.json"),
            parent_hash=digest(a.parent_root / "complete.json"),
            baseline_hash=digest(a.r2_root / "complete.json"),
            bank=cd["bank"],
            comparisons=comparisons,
            no_baseline_or_acquired_regressions=all(
                c["no_regression"] for c in comparisons.values()
            ),
            own_native_steps=0,
            own_neural_rows=0,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
