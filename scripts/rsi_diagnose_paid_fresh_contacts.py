"""Explain independently audited Fresh failures; never fit or rewrite scores."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("review-root", "r2-root", "parent-root", "progressive-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("review-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    rp = a.review_root / "complete.json"
    if (
        a.output.exists()
        or (a.review_root / "failed.json").exists()
        or digest(rp) != a.review_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("exact closed paired review and frozen diagnostic source required")
    review = json.loads(rp.read_text())
    if (
        review["status"] != "PAID_FRESH32_PAIRED_REVIEW_CLOSED"
        or review["independent_audits_and_native_replays_all_closed"] is not True
        or review["first_R2_prefix_exact_across_all_arms"] is not True
        or review["hardware_authorized"] is not False
        or review["promotion_authorized"] is not False
    ):
        raise ValueError("complete independently verified Fresh evidence required")
    from rosclaw_soccer.rsi.paid_contact_diagnosis import summarize_contact_failures

    pins = dict(review["input_file_hashes"])
    pins[str(rp)] = a.review_hash
    results = {}
    for arm in ("r2", "parent", "progressive"):
        root = getattr(a, arm + "_root")
        path = root / "complete.json"
        if pins.get(str(path)) != digest(path) or (root / "failed.json").exists():
            raise ValueError("world not bound to paired review")
        world = json.loads(path.read_text())
        decl = json.loads((root / "declaration.json").read_text())
        if world["arm"] != arm or world["fresh_hash"] != review["fresh_hash"]:
            raise ValueError("wrong paired arm")
        per_kick = {}
        for k in range(3):
            classes = {}
            for label in ("near_toward", "near_away", "far_toward", "far_away"):
                indices = [i for i, c in enumerate(decl["cases"]) if c["fresh_class"] == label]
                present = [i for i in indices if len(world["worlds"][i]["kicks"]) > k]
                if not present:
                    classes[label] = dict(
                        declared_cases=indices, executed_cases=[], unreached_cases=indices
                    )
                    continue
                rows = [world["worlds"][i]["kicks"][k]["result"] for i in present]
                diagnostic = summarize_contact_failures(rows)
                if diagnostic["quality_pass"] != sum(
                    world["worlds"][i]["quality_by_kick"][k] for i in present
                ):
                    raise ValueError("diagnostic categories disagree with frozen score")
                classes[label] = dict(
                    **diagnostic,
                    declared_cases=indices,
                    executed_cases=present,
                    unreached_cases=sorted(set(indices) - set(present)),
                )
            per_kick[str(k)] = classes
        results[arm] = per_kick
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="PAID_AUDITED_FRESH_CONTACT_FAILURES_DIAGNOSED_WITHOUT_SCORE_CHANGE",
            source_head=a.expected_head,
            input_file_hashes=pins,
            review_hash=a.review_hash,
            results=results,
            diagnostic_tags_can_overlap=True,
            Fresh_episodes_fit=0,
            own_native_steps=0,
            own_neural_rows=0,
            optimizer_updates=0,
            training_authorized_by_this_diagnostic=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
