"""Research-only cross-validated risk selection from bound paired SIM outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.receiver_risk_cv import grouped_risk_selection
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("review", "ball-audit", "baseline", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("review-hash", "ball-audit-hash", "baseline-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    pins = (
        (args.review, args.review_hash),
        (args.ball_audit, args.ball_audit_hash),
        (args.baseline, args.baseline_hash),
    )
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != expected for path, expected in pins)
    ):
        raise ValueError("unchanged pinned inputs and unused external output required")
    review, ball, baseline = (json.loads(path.read_text()) for path, _ in pins)
    if (
        review.get("status") != "LONG_RECEIVER_PAIRED_SIM_DIAGNOSIS_NOT_POSSESSION"
        or ball.get("status") != "SIM_BALL_OBSERVATION_FINITE_DIFFERENCE_AUDIT_NOT_POLICY"
        or review.get("baseline_hash") != args.baseline_hash
        or review.get("snapshot_hash") != ball.get("snapshot_hash")
        or baseline.get("snapshot_hash") != ball.get("snapshot_hash")
        or any(
            receipt.get(k) is not False
            for receipt in (review, ball, baseline)
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "independent_fresh",
            )
        )
    ):
        raise ValueError("same unpromoted paired outcomes and causal snapshot required")
    rows = {r["index"]: r for r in review["rows"]}
    balls = {r["index"]: r for r in ball["rows"]}
    bases = {r["index"]: r for r in baseline["rows"]}
    if (
        any(len(d["rows"]) != 24 for d in (review, ball, baseline))
        or len(rows) != 24
        or set(rows) != set(balls)
        or set(rows) != set(bases)
    ):
        raise ValueError("all unique paired source cases required")
    xs, bs, gs, failed_b, failed_c, observed, indices = [], [], [], [], [], [], []
    for index in sorted(rows):
        r, br, source = rows[index], balls[index], bases[index]
        if r["context"] != br["context"] or r["context"] != source["context"]:
            raise ValueError("matched source context required")
        path = args.baseline.parent / source["physical_trajectory"]
        if (
            path.is_symlink()
            or path.resolve().parent != args.baseline.parent.resolve()
            or digest(path) != source["physical_trajectory_hash"]
        ):
            raise ValueError("original matched pre-intervention observation required")
        with np.load(path, allow_pickle=False) as data:
            x = data["receiver_observation"][0].copy()
        if (
            digest(path) != source["physical_trajectory_hash"]
            or x.dtype != np.float32
            or x.shape != (98,)
            or not np.isfinite(x).all()
        ):
            raise ValueError("stable finite current receiver observation required")
        feature = np.asarray(br["feature"], dtype=np.float32)
        if feature.shape != (6,) or not np.isfinite(feature).all():
            raise ValueError("finite current ball feature required")
        ready = r["baseline_ready"] and r["candidate_ready"]
        # A nonfailed return must survive all 460 post-snapshot controls;
        # early no-handoff traces are censored rather than negative examples.
        complete = ready and all(
            r[f"{arm}_body_failure"] or r[f"{arm}_controls"] == 460
            for arm in ("baseline", "candidate")
        )
        xs.append(x)
        bs.append(feature)
        gs.append(r["context"])
        indices.append(index)
        failed_b.append(r["baseline_body_failure"])
        failed_c.append(r["candidate_body_failure"])
        observed.append(complete)
    x = np.stack(xs)
    group = np.asarray(gs, np.int64)
    base = np.asarray(failed_b, np.bool_)
    candidate = np.asarray(failed_c, np.bool_)
    mask = np.asarray(observed, np.bool_)
    torch.set_num_threads(1)
    own = grouped_risk_selection(x, group, base, candidate, mask)
    augmented = grouped_risk_selection(
        np.concatenate((x, np.stack(bs)), axis=1), group, base, candidate, mask
    )
    if any(digest(path) != expected for path, expected in pins):
        raise ValueError("pinned source changed during training")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="GROUPED_FAILURE_RISK_SELECTOR_RESEARCH_NOT_PHYSICALLY_VALIDATED",
            source_head=args.expected_head,
            review_hash=args.review_hash,
            ball_audit_hash=args.ball_audit_hash,
            baseline_hash=args.baseline_hash,
            indices=indices,
            groups=gs,
            observed=observed,
            baseline_failed=failed_b,
            candidate_failed=failed_c,
            proprioception_only=own,
            ball_augmented=augmented,
            threshold=0.8,
            negative_loss_weight=3.0,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
