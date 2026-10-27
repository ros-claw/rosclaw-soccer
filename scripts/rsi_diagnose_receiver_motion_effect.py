"""Read actual own-state velocity traces after full independent paired review."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.receiver_motion_effect import paired_receiver_motion_effect
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SOURCE", "AUDIT"))
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--source-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.pair) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.review.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("three audited sources and exact paired review required")
    review = json.loads(a.review.read_text())
    if (
        review.get("status") != "RECEIVER_LEARNED_PAIR_PHYSICAL_REPLAY_REVIEWED"
        or review.get("producer_head") != a.source_head
        or review.get("conditions") != 24
        or any(
            review.get(k) is not False
            for k in ("promotion_authorized", "hardware_authorized", "goal_complete")
        )
    ):
        raise ValueError("full reviewed physical comparison required")
    expected = {(r["context"], r["side"]): r for r in review["records"]}
    if len(expected) != 24:
        raise ValueError("24 unique reviewed paired conditions required")
    rows, frozen, blocks = [], [], set()
    for source_name, audit_name in a.pair:
        root, audit = Path(source_name), Path(audit_name)
        declaration = json.loads((root / "declaration.json").read_text())
        block = declaration["development_context_block"]
        if block in blocks or block not in (0, 1, 2):
            raise ValueError("unique reviewed blocks required")
        blocks.add(block)
        ah = digest(audit)
        source, _, sh = report(
            root, audit, ah, "NEURAL_RELAY_RECEIVER_LEARNED_PAIR16_PENDING_OWN_REPLAY", worlds=16
        )
        if source["source_head"] != a.source_head:
            raise ValueError("exact paired producer required")
        paired = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["receiver_learned"],
            ): r
            for r in source["records"]
        }
        if len(paired) != 16:
            raise ValueError("complete two-arm physical factorial required")
        for condition in sorted({(x, y) for x, y, _ in paired}):
            base, candidate = paired[(*condition, False)], paired[(*condition, True)]
            row = expected[condition]
            if (
                row["block"] != block
                or measured(root, base) != row["baseline"]
                or measured(root, candidate) != row["candidate"]
                or row["historical_baseline_identical"] is not True
            ):
                raise ValueError("full physical outcomes must match reviewed evidence")
            paths = [bind(root, r, "receiver_velocity") for r in (base, candidate)]
            if any(path is None for path in paths):
                raise ValueError("actual full receiver actions required")
            with (
                np.load(paths[0], allow_pickle=False) as left,
                np.load(paths[1], allow_pickle=False) as right,
            ):
                effect = paired_receiver_motion_effect(dict(left), dict(right))
            rows.append(
                dict(
                    context=condition[0],
                    side=condition[1],
                    block=block,
                    baseline_termination=base["termination"],
                    candidate_termination=candidate["termination"],
                    **effect,
                )
            )
        frozen.extend(((root / "complete.json", sh), (audit, ah)))
    if (
        blocks != {0, 1, 2}
        or len(rows) != 24
        or {(r["context"], r["side"]) for r in rows} != set(expected)
        or digest(a.review) != a.review_hash
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("all unchanged original complete physical pairs required")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_PAIRED_OWN_MOTION_DIAGNOSIS",
            source_head=a.expected_head,
            review_hash=a.review_hash,
            conditions=len(rows),
            records=rows,
            root_ball_proxy_not_foot_contact=True,
            aligned_states_are_not_fixed_state_causal_interventions=True,
            independent_new_physics=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
