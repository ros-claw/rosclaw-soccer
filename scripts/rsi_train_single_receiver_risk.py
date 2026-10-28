"""Train one unpromoted 104D SIM research selector with explicit failure protection."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.receiver_risk_cv import grouped_risk_selection
from rosclaw_soccer.rsi.single_receiver_risk import fit_single_receiver_risk
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_train_multiblock_receiver_risk import assemble


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
        raise ValueError("three complete development banks and new external output required")
    pins = [
        (Path(path), expected)
        for block in args.block
        for path, expected in ((block[0], block[1]), (block[2], block[3]))
    ]
    if len({p.resolve() for p, _ in pins}) != 6 or any(
        p.is_symlink() or digest(p) != h for p, h in pins
    ):
        raise ValueError("six exact distinct immutable receipts required")
    blocks = [
        (json.loads(pins[i][0].read_text()), json.loads(pins[i + 1][0].read_text()))
        for i in range(0, 6, 2)
    ]
    x, ball, groups, base, candidate, complete, ids = assemble(blocks)
    if set(groups.tolist()) != {2, 3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 18}:
        raise ValueError("only explicitly declared non-Fresh development contexts allowed")
    features = np.concatenate((x, ball), axis=1)
    admitted = complete | candidate
    torch.set_num_threads(1)
    cv = grouped_risk_selection(
        features, groups, base, candidate, admitted, predict_unobserved=True
    )
    prediction = np.asarray(cv["predictions"], dtype=np.int64)
    cv["complete_pair_baseline_failures"] = int(base[complete].sum())
    cv["complete_pair_selected_failures"] = int(
        np.where(prediction == 1, candidate, base)[complete].sum()
    )
    cv["additional_direct_failure_examples"] = int((candidate & ~complete).sum())
    cv["selected_additional_direct_candidate_failures"] = int(
        ((prediction == 1) & candidate & ~complete).sum()
    )
    # These default aggregate fields would incorrectly compare a candidate's
    # known failure to a censored baseline. Retain only complete-pair metrics.
    for key in (
        "baseline_failures",
        "always_candidate_failures",
        "selected_counterfactual_failures",
        "selected_regressions",
        "selected_rescues",
    ):
        cv.pop(key)
    model, rehearsal = fit_single_receiver_risk(features, base, candidate, complete)
    if any(digest(p) != h for p, h in pins):
        raise ValueError("training source changed during fit")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="SINGLE_RECEIVER_RISK_DEV_MODEL_NOT_PHYSICALLY_VALIDATED",
            source_head=args.expected_head,
            inputs=[dict(path=str(p), hash=h) for p, h in pins],
            identities=ids,
            complete_pair=complete.tolist(),
            admitted=admitted.tolist(),
            baseline_failed=base.tolist(),
            candidate_failed=candidate.tolist(),
            label_contract="complete_rescue_positive_or_direct_candidate_failure_negative",
            threshold=0.8,
            model=model,
            cross_validation=cv,
            in_sample_rehearsal=rehearsal,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
