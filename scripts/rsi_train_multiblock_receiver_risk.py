"""Group-held-out research on complete paired developmental SIM returns."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.receiver_risk_cv import grouped_risk_selection
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def assemble(blocks: list[tuple[dict, dict]]) -> tuple:
    """Join by bound snapshot and within-block index; never use future observations."""
    xs, balls, groups, baseline, candidate, observed, identities = [], [], [], [], [], [], []
    seen_contexts: set[int] = set()
    for review, features in blocks:
        if (
            review.get("status") != "LONG_RECEIVER_PAIRED_SIM_DIAGNOSIS_NOT_POSSESSION"
            or features.get("status") != "CAUSAL_RECEIVER_FEATURES_DEV_ONLY_NOT_TRAINED"
            or review.get("snapshot_hash") != features.get("snapshot_hash")
            or any(
                d.get(k) is not False
                for d in (review, features)
                for k in (
                    "independent_fresh",
                    "hardware_authorized",
                    "promotion_authorized",
                    "goal_complete",
                )
            )
        ):
            raise ValueError("matched unpromoted same-snapshot evidence required")
        rr = {r["index"]: r for r in review["rows"]}
        ff = {r["index"]: r for r in features["rows"]}
        if (
            len(review["rows"]) != 24
            or len(features["rows"]) != 24
            or len(rr) != 24
            or set(rr) != set(ff)
        ):
            raise ValueError("all unique paired cases required")
        contexts = {r["context"] for r in ff.values()}
        if len(contexts) != 4 or contexts & seen_contexts:
            raise ValueError("disjoint four-context development blocks required")
        seen_contexts.update(contexts)
        for index in sorted(rr):
            r, f = rr[index], ff[index]
            if r["context"] != f["context"]:
                raise ValueError("paired source context mismatch")
            x, ball = (
                np.asarray(f["observation"], np.float32),
                np.asarray(f["ball_feature"], np.float32),
            )
            if (
                x.shape != (98,)
                or ball.shape != (6,)
                or not np.isfinite(x).all()
                or not np.isfinite(ball).all()
            ):
                raise ValueError("finite current own and ball features required")
            if any(
                type(r.get(k)) is not bool
                for k in (
                    "baseline_ready",
                    "candidate_ready",
                    "baseline_body_failure",
                    "candidate_body_failure",
                )
            ):
                raise ValueError("explicit physical readiness and body outcomes required")
            complete = (
                r["baseline_ready"]
                and r["candidate_ready"]
                and all(
                    r[f"{arm}_body_failure"] or r[f"{arm}_controls"] == 460
                    for arm in ("baseline", "candidate")
                )
            )
            xs.append(x)
            balls.append(ball)
            groups.append(r["context"])
            baseline.append(r["baseline_body_failure"])
            candidate.append(r["candidate_body_failure"])
            observed.append(complete)
            identities.append(
                dict(snapshot_hash=review["snapshot_hash"], index=index, context=r["context"])
            )
    return (
        np.stack(xs),
        np.stack(balls),
        np.asarray(groups, np.int64),
        np.asarray(baseline, np.bool_),
        np.asarray(candidate, np.bool_),
        np.asarray(observed, np.bool_),
        identities,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block",
        nargs=4,
        action="append",
        required=True,
        metavar=("REVIEW", "REVIEW_HASH", "FEATURES", "FEATURE_HASH"),
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        len(args.block) != 3
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("three complete development blocks and unique external result required")
    pins = [
        (Path(path), expected)
        for block in args.block
        for path, expected in ((block[0], block[1]), (block[2], block[3]))
    ]
    if len({p.resolve() for p, _ in pins}) != 6 or any(
        p.is_symlink() or digest(p) != h for p, h in pins
    ):
        raise ValueError("six exact distinct immutable input receipts required")
    blocks = [
        (json.loads(pins[i][0].read_text()), json.loads(pins[i + 1][0].read_text()))
        for i in range(0, 6, 2)
    ]
    x, ball, groups, base, cand, observed, ids = assemble(blocks)
    if set(groups.tolist()) != {2, 3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 18}:
        raise ValueError("only declared non-Fresh context blocks permitted")
    torch.set_num_threads(1)
    own = grouped_risk_selection(x, groups, base, cand, observed)
    augmented = grouped_risk_selection(
        np.concatenate((x, ball), axis=1), groups, base, cand, observed
    )
    if any(digest(p) != h for p, h in pins):
        raise ValueError("input changed during training")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="MULTIBLOCK_GROUPED_RISK_RESEARCH_NOT_PHYSICALLY_VALIDATED",
            source_head=args.expected_head,
            inputs=[dict(path=str(p), hash=h) for p, h in pins],
            identities=ids,
            observed=observed.tolist(),
            baseline_failed=base.tolist(),
            candidate_failed=cand.tolist(),
            proprioception_only=own,
            ball_augmented=augmented,
            threshold=0.8,
            negative_loss_weight=3.0,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
