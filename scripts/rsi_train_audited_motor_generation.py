"""Bind a queued, frozen fresh-generation producer before on-policy learning."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("generation-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("generation-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--generation-index", type=int, choices=range(1, 100), required=True)
    p.add_argument("--curriculum", choices=("run2", "retention-run64"), required=True)
    p.add_argument("--critic-regression-steps", type=int, choices=(0, 32), default=0)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest

    if (
        subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("clean frozen queued generation trainer required")
    sp = a.generation_root / "complete.json"
    ep = a.generation_root / "experience" / "complete.json"
    ap = a.generation_root / "audit" / "complete.json"
    dp = ep.parent / "declaration.json"
    if a.output.exists() or any(x.is_symlink() for x in (sp, ep, ap, dp)):
        raise ValueError("ordinary complete sources and unused output required")
    s, e, ar, d = [json.loads(x.read_text()) for x in (sp, ep, ap, dp)]
    from rosclaw_soccer.rsi.motor_exploration_curriculum import declared_motor_curriculum

    c = declared_motor_curriculum(d)
    if (
        s["status"] != "FRESH_AR_GENERATION_COLLECTED_AND_AUDITED_NOT_TRAINED"
        or ar["status"] != c.audit_status
        or ar["source_hash"] != digest(ep)
        or e["declaration_hash"] != digest(dp)
        or c.name != a.curriculum
        or d["generation_index"] != a.generation_index
        or any(
            r["source_head"] != a.generation_head
            or r["hardware_authorized"] is not False
            or r["promotion_authorized"] is not False
            for r in (s, e, ar)
        )
    ):
        raise ValueError("exact new policy generation and independent native/law witness required")
    write_once(
        a.output.parent / (a.output.name + "-source-binding.json"),
        dict(
            source_head=a.expected_head,
            generation_head=a.generation_head,
            generation_hash=digest(sp),
            experience_hash=digest(ep),
            audit_hash=digest(ap),
            generation_index=a.generation_index,
            curriculum=c.name,
            worlds=c.worlds,
            automatic_retry=False,
            hardware_authorized=False,
        ),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_train_dual_onpolicy",
            "--experience-root",
            str(ep.parent),
            "--audit-root",
            str(ap.parent),
            "--audit-hash",
            digest(ap),
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
            "--critic-regression-steps",
            str(a.critic_regression_steps),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
