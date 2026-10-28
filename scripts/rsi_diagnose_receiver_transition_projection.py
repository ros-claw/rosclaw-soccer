"""Inspect first-step reference-bounded handoff projection without acting."""

import argparse
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
from rosclaw_soccer.rsi.paid_velocity_shadow import policy_action_from_hg_target
from rosclaw_soccer.rsi.receiver_transition_blend import receiver_transition_action
from rosclaw_soccer.rsi.relay_snapshot_cases import select_relay_snapshot_cases
from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_context_recovery_common import check_pins
from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt
from scripts.rsi_review_relay_balance_counterfactual import bind, report
from scripts.rsi_stroke_search_common import open_motor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "frontier", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "frontier-head", "frontier-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    root, audit = a.source / "source", a.source / "audit/complete.json"
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external result required")
    source, _, sh = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier, a.frontier_head, a.frontier_hash
    )
    if (
        sh != a.source_hash
        or [c["seed"] for c in candidates] != [65023]
        or any(source["input_file_hashes"].get(k) != v for k, v in pins.items())
    ):
        raise ValueError("exact audited source and original frozen actor lineage required")
    check_pins(pins)
    indices = (2, 3, 5, 7)
    bench, base = open_motor(pins, base_hash, {i: contexts[i] for i in indices})
    candidate = candidates[0]
    _, training = receipt(pins, candidate["training_hash"])
    actor, _ = load_dual_candidate_for_sim_audit(
        base.parent,
        parent_lineage_hash=training["parent_lineage_hash"],
        root=Path(candidate["training_root"]),
        expected_receipt_hash=candidate["training_hash"],
        checkpoint_kind=candidate["kind"],
    )
    setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor, context_indices=indices)
    rows = []
    for row in select_relay_snapshot_cases(source["records"], nondefault=True):
        if row["handoff"] is None:
            continue
        physical, neural = bind(root, row, "physics"), bind(root, row, "receiver_return_nn")
        if physical is None or neural is None:
            raise ValueError("own actual handoff and recurrent source traces required")
        h = row["handoff"]["control"]
        ep = setup.episodes[row["course"]["context"]]
        with (
            np.load(physical, allow_pickle=False) as physics,
            np.load(neural, allow_pickle=False) as nn,
        ):
            if int(nn["control"][0]) != h:
                raise ValueError("exact first return control required")
            previous_target = physics["target_b"][h - 1].copy()
            previous = policy_action_from_hg_target(previous_target, ep.offset, ep.scale, ep.order)
            state = (
                torch.from_numpy(nn["NN_h_before"][0].copy()),
                torch.from_numpy(nn["NN_c_before"][0].copy()),
            )
            with torch.no_grad():
                mean, after, evidence = actor.forward_with_evidence(
                    torch.from_numpy(nn["NN_obs"][0].copy()).reshape(1, 1, 160), state
                )
            proposal = mean[0, 0].numpy().copy()
            nominal = evidence["reference_nominal"][0, 0].numpy().copy()
            if (
                not np.array_equal(proposal, nn["NN_action"][0])
                or not np.array_equal(after[0].numpy(), nn["NN_h_after"][0])
                or not np.array_equal(after[1].numpy(), nn["NN_c_after"][0])
                or not np.array_equal(
                    ep.offset + proposal[ep.order] * ep.scale, physics["target_b"][h]
                )
            ):
                raise ValueError("recomputed original own return inference and target required")
            projected = receiver_transition_action(
                previous, proposal, nominal, ep.scale, ep.order, control_index=0, duration=20
            )
            projected_target = ep.offset + projected[ep.order] * ep.scale
            rows.append(
                dict(
                    index=row["index"],
                    handoff_control=h,
                    original_jump_rad=float(
                        np.max(np.abs(physics["target_b"][h] - previous_target))
                    ),
                    projected_jump_rad=float(np.max(np.abs(projected_target - previous_target))),
                    hypothetical_only=True,
                    new_physics_steps=0,
                )
            )
        if (
            digest(physical) != row["artifacts"]["physics"]["hash"]
            or digest(neural) != row["artifacts"]["receiver_return_nn"]["hash"]
        ):
            raise ValueError("source traces changed")
    if len(rows) != 11 or digest(root / "complete.json") != sh or digest(audit) != a.audit_hash:
        raise ValueError("all unchanged original recorded handoffs required")
    clean(a.expected_head)
    check_pins(pins)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="REFERENCE_BOUNDED_TRANSITION_HYPOTHESIS_NOT_PHYSICS",
            source_head=a.expected_head,
            source_hash=sh,
            audit_hash=a.audit_hash,
            frontier_hash=a.frontier_hash,
            duration_controls=20,
            rows=rows,
            new_physics_steps=0,
            optimizer_updates=0,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
