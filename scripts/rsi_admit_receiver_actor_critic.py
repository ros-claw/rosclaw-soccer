"""Admit exact audited receiver data and rebuild healthy identity anchors, without training."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.relay_whole_identity_comparison import compare_receiver_traces
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal, handoff_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "audit", "identity", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset-hash", "audit-hash", "identity-hash", "identity-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--identity-pair", nargs=2, action="append", required=True)
    a = p.parse_args()
    clean(a.expected_head)
    dataset_path = a.dataset / "complete.json"
    files = ((dataset_path, a.dataset_hash), (a.audit, a.audit_hash), (a.identity, a.identity_hash))
    if (
        len(a.identity_pair) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in files)
    ):
        raise ValueError("exact immutable receipts and unused external output required")
    dataset, audit, identity = (json.loads(path.read_text()) for path, _ in files)
    if (
        dataset.get("status") != "RECEIVER_ONPOLICY_PHYSICAL_EPISODES_NOT_TRAINED"
        or audit.get("status") != "RECEIVER_ONPOLICY_READBACK_AND_DENSITY_NOT_TRAINED"
        or audit.get("dataset_hash") != a.dataset_hash
        or audit.get("review_hash") != dataset.get("review_hash")
        or audit.get("exact_inputs_actions_rewards_and_source_replayed") is not True
        or audit.get("max_density_error") != 0.0
        or audit.get("optimizer_authorized") is not False
        or identity.get("status") != "RELAY_WHOLE_RECEIVER_ZERO_IDENTITY_QUALIFIED"
        or identity.get("identity_passed") is not True
        or identity.get("own_physics_and_neural_replay_complete") is not True
        or any(r.get("conditions") != 24 for r in (dataset, audit, identity))
        or any(r.get("episodes") != 72 for r in (dataset, audit))
        or any(r.get("sampled_controls") != 18482 for r in (dataset, audit))
        or any(r.get("hardware_authorized") is not False for r in (dataset, audit, identity))
        or any(r.get("promotion_authorized") is not False for r in (dataset, audit, identity))
    ):
        raise ValueError("complete independently audited unpromoted data required")
    rows = dataset.get("records", [])
    keys = {(r["context"], r["side"], r["seed"]) for r in rows}
    conditions = {(r["context"], r["side"]) for r in rows}
    positive = {r["context"] for r in rows if r["terminal_reward"] > 0}
    negative = {r["context"] for r in rows if r["terminal_reward"] < 0}
    if (
        len(rows) != 72
        or len(keys) != 72
        or len(conditions) != 24
        or len(positive) < 4
        or len(negative) < 4
        or not any(r["actual"]["return_0p1m_and_healthy"] for r in rows)
    ):
        raise ValueError("complete mixed-sign conditions and a precise return required")
    inputs, references, frozen, seen, healthy = [], [], [], set(), []
    identity_rows = {(r["context"], r["side"]): r for r in identity["records"]}
    if len(identity_rows) != 24 or set(identity_rows) != conditions:
        raise ValueError("all identity conditions must match the dataset")
    for root_name, audit_name in a.identity_pair:
        root, own = Path(root_name), Path(audit_name)
        oh = digest(own)
        source, _, sh = report(
            root, own, oh, "NEURAL_RELAY_WHOLE_RECEIVER_IDENTITY16_PENDING_OWN_REPLAY", worlds=16
        )
        if source.get("source_head") != a.identity_head:
            raise ValueError("exact original identity producer required")
        records = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["receiver_whole"],
            ): r
            for r in source["records"]
        }
        if len(records) != 16:
            raise ValueError("complete paired identity block required")
        for condition in {(x, y) for x, y, _ in records}:
            if condition in seen or condition not in identity_rows:
                raise ValueError("duplicate or unreviewed identity condition")
            seen.add(condition)
            baseline, whole = records[(*condition, False)], records[(*condition, True)]
            review = identity_rows[condition]
            if (
                not review["candidate_outcomes_exact"]
                or not review["historic_parent_all_traces_exact"]
                or not review["historic_parent_outcomes_exact"]
                or any(
                    baseline[k] != whole[k]
                    for k in ("termination", "controls", "native_steps", "result")
                )
                or not handoff_equal(root, baseline, root, whole)
                or not all(
                    artifact_equal(root, baseline, root, whole, name)
                    for name in (
                        "initial",
                        "physics",
                        "passer_nn",
                        "receiver_shadow",
                        "receiver_return_nn",
                    )
                )
            ):
                raise ValueError("exact paired identity physical evidence required")
            trace = bind(root, whole, "receiver_velocity")
            baseline_trace = bind(root, baseline, "receiver_velocity")
            if trace is None or baseline_trace is None:
                raise ValueError("exact original receiver history required")
            with (
                np.load(baseline_trace, allow_pickle=False) as left,
                np.load(trace, allow_pickle=False) as right,
            ):
                comparison = compare_receiver_traces(dict(left), dict(right))
            if comparison != review["receiver_trace"] or not all(
                comparison[k]
                for k in (
                    "common_fields_exact",
                    "action_is_actual_onnx_reference",
                    "extra_joint_delta_exact_zero",
                )
            ):
                raise ValueError("actual paired identity receiver traces must match the review")
            if baseline["termination"] != "BODY_FAILURE":
                with np.load(trace, allow_pickle=False) as f:
                    x = np.concatenate((f["obs"], f["context14"]), axis=1)
                    ref = f["reference_action"]
                    if (
                        x.dtype != np.float32
                        or ref.dtype != np.float32
                        or x.shape != (len(ref), 112)
                        or ref.shape[1:] != (29,)
                        or not np.isfinite(x).all()
                        or not np.isfinite(ref).all()
                    ):
                        raise ValueError("finite exact healthy own histories required")
                    inputs.append(x.copy())
                    references.append(ref.copy())
                    healthy.append(
                        dict(
                            context=condition[0],
                            side=condition[1],
                            controls=len(ref),
                            trace_hash=digest(trace),
                        )
                    )
        frozen.extend(((root / "complete.json", sh), (own, oh)))
    if seen != conditions or not healthy:
        raise ValueError("all 24 identity conditions and healthy anchors required")
    for row in rows:
        path = a.dataset / row["path"]
        if (
            path.is_symlink()
            or path.resolve().parent != a.dataset.resolve()
            or digest(path) != row["hash"]
        ):
            raise ValueError("all exact confined audited episodes required")
    if any(digest(path) != h for path, h in (*files, *frozen)):
        raise ValueError("source receipts changed during admission")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    anchor_path = a.output / "healthy-anchors.npz"
    np.savez_compressed(
        anchor_path, inputs=np.concatenate(inputs), reference=np.concatenate(references)
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_RESEARCH_TRAINING_DATA_ADMITTED_NOT_TRAINED",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            audit_hash=a.audit_hash,
            identity_hash=a.identity_hash,
            conditions=len(conditions),
            episodes=len(rows),
            sampled_controls=18482,
            positive_original_contexts=sorted(positive),
            negative_original_contexts=sorted(negative),
            healthy_conditions=healthy,
            healthy_anchor_controls=sum(x["controls"] for x in healthy),
            healthy_anchor_hash=digest(anchor_path),
            thresholds_selected_after_data=True,
            offline_research_training_authorized=True,
            actor_updates=0,
            critic_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
