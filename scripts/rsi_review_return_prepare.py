"""Review every preregistered paired preparation world and own replay."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.return_prepare_contact import return_prepare_contact
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def outcome(root, row, horizon):
    path = bind(root, row, "physics")
    if path is None:
        raise ValueError("exact replayed physical history required")
    with np.load(path, allow_pickle=False) as data:
        forces, failed = data["ball_forces"][:horizon], data["body_failed"][:horizon]
        handoff = row["handoff"]["control"] if row["handoff"] is not None else None
        if handoff is not None and handoff >= horizon:
            handoff = None
        return return_prepare_contact(
            forces,
            failed,
            handoff=handoff,
            outgoing_foot_strike=row["result"]["outgoing_foot_strike"],
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("SOURCE", "AUDIT", "PRIOR", "PRIOR_AUDIT"),
    )
    p.add_argument("--audit-hash", action="append", required=True)
    p.add_argument("--prior-audit-hash", action="append", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.group) != 3 or len(a.audit_hash) != 3 or len(a.prior_audit_hash) != 3:
        raise ValueError("all three declared paired blocks and replay pins required")
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    records, blocks, pins = [], set(), []
    for group, ah, ph in zip(a.group, a.audit_hash, a.prior_audit_hash, strict=True):
        root, audit, previous, prior_audit = map(Path, group)
        src, _, source_hash = report(
            root, audit, ah, "NEURAL_RELAY_RETURN_PLAYER_PREPARE16_PENDING_OWN_REPLAY", worlds=16
        )
        prior, _, prior_hash = report(
            previous,
            prior_audit,
            ph,
            "NEURAL_RELAY_CHILD_RETURN_ROLE16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        if src["source_head"] == prior["source_head"]:
            raise ValueError("explicit distinct current and old protocols required")
        trial = src["records"]
        block = tuple(sorted({r["course"]["context"] for r in trial}))
        if block not in ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18)) or block in blocks:
            raise ValueError("unique complete declared development block required")
        blocks.add(block)
        for i in range(0, 16, 2):
            old, new, earlier = trial[i], trial[i + 1], prior["records"][i + 1]
            if (
                old["course"]["context"] != new["course"]["context"]
                or old["course"]["lateral_m"] != new["course"]["lateral_m"]
                or old["course"]["arm"]["passer_prepare"] is not False
                or new["course"]["arm"]["passer_prepare"] is not True
                or old["artifacts"]["initial"]["hash"] != new["artifacts"]["initial"]["hash"]
                or old["artifacts"]["physics"]["hash"] != earlier["artifacts"]["physics"]["hash"]
            ):
                raise ValueError(
                    "exact matched context, initial state and old role baseline required"
                )
            common = min(old["controls"], new["controls"])
            records.append(
                dict(
                    context=old["course"]["context"],
                    side=old["course"]["lateral_m"],
                    common_controls=common,
                    baseline_termination=old["termination"],
                    candidate_termination=new["termination"],
                    baseline_full=outcome(root, old, old["controls"]),
                    candidate_full=outcome(root, new, new["controls"]),
                    baseline_common=outcome(root, old, common),
                    candidate_common=outcome(root, new, common),
                )
            )
        pins.append(dict(source=source_hash, audit=ah, prior=prior_hash, prior_audit=ph))
    counts = Counter((r["baseline_termination"], r["candidate_termination"]) for r in records)
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RETURN_PREPARATION_REPLAYED_DEVELOPMENT_ONLY",
            source_head=a.expected_head,
            blocks=3,
            conditions=len(records),
            common_horizon_contact_proxy=True,
            b_event_is_first_post_handoff_foot_not_isolated_impulse=True,
            baseline_matches_prior_physics=True,
            termination_pairs=[
                dict(baseline=k[0], candidate=k[1], count=v) for k, v in sorted(counts.items())
            ],
            records=records,
            source_pins=pins,
            learned_policy=False,
            possession_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
