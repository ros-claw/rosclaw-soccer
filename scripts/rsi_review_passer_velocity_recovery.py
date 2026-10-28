"""Review independently replayed actual passer velocity recovery pairs."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.paired_health_exposure import paired_health_at_common_horizon
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from rosclaw_soccer.rsi.passer_velocity_memory_validation import validate_passer_velocity_history
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "prior", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "prior-hash", "prior-audit-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    root, audit = a.source / "source", a.source / "audit/complete.json"
    old, old_audit = a.prior / "source", a.prior / "audit/complete.json"
    declaration = json.loads((root / "declaration.json").read_text())
    if (
        declaration.get("protocol")
        != "soccer.neural_relay_passer_velocity_recovery.development.v30"
        or declaration.get("development_context_block") != 0
        or declaration.get("passer_velocity_recovery_pair") is not True
    ):
        raise ValueError("exact preregistered passer recovery study required")
    source, _, sh = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_PASSER_VELOCITY_RECOVERY16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    prior, _, ph = report(
        old,
        old_audit,
        a.prior_audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    if sh != a.source_hash or ph != a.prior_hash:
        raise ValueError("exact matched independently replayed source and prior required")
    configs = [
        Path(x)
        for x in source["input_file_hashes"]
        if x.endswith("/velocity/v0/params/deploy.yaml")
    ]
    if len(configs) != 1:
        raise ValueError("one pinned velocity codec required")
    params = VelocityPolicyParameters.load(
        configs[0], expected_hash=source["input_file_hashes"][str(configs[0])]
    )
    offset, scale = (
        np.asarray(params.action_offset, np.float64),
        np.asarray(params.action_scale, np.float64),
    )
    rows = {
        (
            r["course"]["context"],
            r["course"]["lateral_m"],
            r["course"]["arm"]["passer_velocity_recovery"],
        ): r
        for r in source["records"]
    }
    baseline = {
        (r["course"]["context"], r["course"]["lateral_m"]): r
        for r in prior["records"]
        if r["course"]["arm"]["receiver_lateral_offset_m"] == 0.12
    }
    if len(rows) != 16 or len(baseline) != 8:
        raise ValueError("eight unique complete matched conditions required")
    records = []
    for key in sorted(baseline):
        parent, candidate = rows[(*key, False)], rows[(*key, True)]
        if not artifact_equal(root, parent, old, baseline[key], "initial") or not artifact_equal(
            root, parent, root, candidate, "initial"
        ):
            raise ValueError("same initialized actual bodies and ball required")
        identity = artifact_equal(root, parent, old, baseline[key], "physics")
        paths = [bind(root, row, "physics") for row in (parent, candidate)]
        trace_path = bind(root, candidate, "passer_velocity")
        if any(path is None for path in paths) or trace_path is None:
            raise ValueError("complete actual physical and passer histories required")
        with (
            np.load(paths[0], allow_pickle=False) as aa,
            np.load(paths[1], allow_pickle=False) as bb,
            np.load(trace_path, allow_pickle=False) as v,
        ):
            boundary = candidate["passer_velocity_recovery_boundary"]
            memory = validate_passer_velocity_history(
                observations=v["obs"],
                proposed=v["proposed_action"],
                actual_targets=v["actual_target"],
                memory_after=v["action_memory_after"],
                active=v["active"],
                offset=offset,
                scale=scale,
            )
            if (boundary is None) != (memory["first_active_control"] is None) or (
                boundary is not None and boundary["control"] != memory["first_active_control"]
            ):
                raise ValueError("actual switch and own observed active history differ")
            common = min(len(aa["qpos"]), len(bb["qpos"]))
            prefix = common if boundary is None else min(common, boundary["control"])
            prefix_exact = all(
                np.array_equal(aa[k][:prefix], bb[k][:prefix])
                for k in (
                    "qpos",
                    "qvel",
                    "torque",
                    "ball_forces",
                    "body_failed",
                    "time_before",
                    "target_b",
                )
            )
            jump = (
                None
                if boundary is None or boundary["control"] == 0
                else float(
                    np.max(
                        np.abs(
                            v["actual_target"][boundary["control"]]
                            - v["actual_target"][boundary["control"] - 1]
                        )
                    )
                )
            )
            records.append(
                dict(
                    context=key[0],
                    side=key[1],
                    baseline_matches_prior=identity,
                    pre_switch_prefix_exact=prefix_exact,
                    pre_switch_compared_controls=prefix,
                    full_prefix_available=boundary is not None and common >= boundary["control"],
                    boundary_control=None if boundary is None else boundary["control"],
                    boundary_target_jump_rad=jump,
                    memory=memory,
                    health=paired_health_at_common_horizon(aa["body_failed"], bb["body_failed"]),
                    baseline_outcome=strict_pass_receive(
                        aa["ball_forces"],
                        aa["body_failed"],
                        handoff=parent["result"]["handoff_executed"],
                    ),
                    candidate_outcome=strict_pass_receive(
                        bb["ball_forces"],
                        bb["body_failed"],
                        handoff=candidate["result"]["handoff_executed"],
                    ),
                    baseline_precise=parent["result"]["return_0p1m_and_healthy"],
                    candidate_precise=candidate["result"]["return_0p1m_and_healthy"],
                )
            )
    if any(
        digest(p) != h
        for p, h in (
            (root / "complete.json", sh),
            (audit, a.audit_hash),
            (old / "complete.json", ph),
            (old_audit, a.prior_audit_hash),
        )
    ):
        raise ValueError("reviewed source changed")
    summary = dict(
        exact_prior=sum(r["baseline_matches_prior"] for r in records),
        exact_pre_switch=sum(r["pre_switch_prefix_exact"] for r in records),
        exact_full_pre_switch=sum(
            r["pre_switch_prefix_exact"] and r["full_prefix_available"] for r in records
        ),
        activated=sum(r["boundary_control"] is not None for r in records),
        baseline_healthy_common=sum(r["health"]["raw_healthy_common"] for r in records),
        candidate_healthy_common=sum(r["health"]["executed_healthy_common"] for r in records),
        baseline_strict=sum(r["baseline_outcome"]["strict_both_clean_handoff"] for r in records),
        candidate_strict=sum(r["candidate_outcome"]["strict_both_clean_handoff"] for r in records),
        baseline_precise=sum(r["baseline_precise"] for r in records),
        candidate_precise=sum(r["candidate_precise"] for r in records),
    )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PASSER_VELOCITY_RECOVERY_REPLAY_REVIEWED",
            source_head=a.expected_head,
            source_hash=sh,
            audit_hash=a.audit_hash,
            prior_hash=ph,
            summary=summary,
            records=records,
            independent_fresh=False,
            learned_update=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
