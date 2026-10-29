"""Review every B-child candidate pair after whole-world own-history replay."""

import argparse
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_receiver_match_prepared import physical_outcome
from scripts.rsi_review_relay_balance_counterfactual import bind, report


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
        raise ValueError("all declared candidate blocks and exact replay pins required")
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    records, pins, blocks = [], [], set()
    for group, ah, ph in zip(a.group, a.audit_hash, a.prior_audit_hash, strict=True):
        root, audit, prior_root, prior_audit = map(Path, group)
        src, _, sh = report(
            root,
            audit,
            ah,
            "NEURAL_RELAY_CHILD_VELOCITY_CANDIDATE_FIXED16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        prior, _, oldhash = report(
            prior_root,
            prior_audit,
            ph,
            "NEURAL_RELAY_RECEIVER_VELOCITY_MATCH_PREPARED16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        context = tuple(sorted({r["course"]["context"] for r in src["records"]}))
        if context not in ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18)) or context in blocks:
            raise ValueError("all unique predeclared development blocks required")
        blocks.add(context)
        scene = root / "scene.xml"
        scene_hash = digest(scene)
        if scene.is_symlink() or digest(prior_root / "scene.xml") != scene_hash:
            raise ValueError("exact independently replayed shared scene required")
        model = mujoco.MjModel.from_xml_path(str(scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        observed_candidate_calls = 0
        for i in range(0, 16, 2):
            old, new, expected = src["records"][i], src["records"][i + 1], prior["records"][i + 1]
            if (
                old["course"]["context"] != new["course"]["context"]
                or old["course"]["lateral_m"] != new["course"]["lateral_m"]
                or old["course"]["arm"]["receiver_child_candidate"] is not False
                or new["course"]["arm"]["receiver_child_candidate"] is not True
                or old["artifacts"]["initial"]["hash"] != new["artifacts"]["initial"]["hash"]
                or old["artifacts"]["physics"]["hash"] != expected["artifacts"]["physics"]["hash"]
            ):
                raise ValueError("exact paired initial conditions and v51 baseline required")
            oldv, newv = bind(root, old, "receiver_velocity"), bind(root, new, "receiver_velocity")
            if oldv is None or newv is None:
                raise ValueError("actual candidate execution traces required")
            with np.load(oldv, allow_pickle=False) as x, np.load(newv, allow_pickle=False) as y:
                if x["candidate_actor_selected"].any() or not y["candidate_actor_selected"].all():
                    raise ValueError("recorded actual candidate actor assignment required")
                observed_candidate_calls += len(y["candidate_actor_selected"])
                n = min(len(x["action"]), len(y["action"]))
                delta = np.max(np.abs(x["action"][:n] - y["action"][:n]), axis=1)
                indices = np.flatnonzero(delta != 0)
                first = int(indices[0]) if len(indices) else None
                if first is not None and any(
                    not np.array_equal(x[k][first], y[k][first])
                    for k in ("obs", "context14", "reference_action")
                ):
                    raise ValueError(
                        "first candidate action change must start from identical B state"
                    )
                first_delta = float(delta[first]) if first is not None else 0.0
            common = min(old["controls"], new["controls"])
            records.append(
                dict(
                    context=old["course"]["context"],
                    side=old["course"]["lateral_m"],
                    first_different_action=first,
                    first_action_difference_policy_units=first_delta,
                    common_controls=common,
                    baseline_controls=old["controls"],
                    candidate_controls=new["controls"],
                    baseline_termination=old["termination"],
                    candidate_termination=new["termination"],
                    baseline_full=physical_outcome(root, old, old["controls"], model, layout),
                    candidate_full=physical_outcome(root, new, new["controls"], model, layout),
                    baseline_common=physical_outcome(root, old, common, model, layout),
                    candidate_common=physical_outcome(root, new, common, model, layout),
                    baseline_return=old["result"],
                    candidate_return=new["result"],
                )
            )
        if (
            observed_candidate_calls == 0
            or observed_candidate_calls != src["counts"]["receiver_candidate_calls"]
        ):
            raise ValueError("actual candidate inference count must match verified source")
        if digest(scene) != scene_hash:
            raise ValueError("audited scene changed during review")
        pins.append(
            dict(
                source=sh,
                audit=ah,
                prior=oldhash,
                prior_audit=ph,
                scene=scene_hash,
                candidate_calls=observed_candidate_calls,
            )
        )
    terms = Counter((r["baseline_termination"], r["candidate_termination"]) for r in records)
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="B_CHILD_VELOCITY_CANDIDATE_REPLAYED_DEVELOPMENT_ONLY",
            source_head=a.expected_head,
            conditions=len(records),
            source_pins=pins,
            records=records,
            termination_pairs=[
                dict(baseline=k[0], candidate=k[1], count=v) for k, v in sorted(terms.items())
            ],
            parent_baselines_match_v51=True,
            candidate_execution_and_own_replay_verified=True,
            physical_retention_proven=False,
            possession_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
