"""Close a reference-bridge screen only after ALL own replays and baseline identity.

Zero physics/learning. Links whole128 proof and exact raw streams, preserves
all failures, compares repeated arms within8 distinct TRAIN conditions.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "audit-root", "prior-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "prior-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused immutable comparison output required")
    reports = []
    for root, h in (
        (a.source_root, a.source_hash),
        (a.audit_root, a.audit_hash),
        (a.prior_root, a.prior_hash),
    ):
        path = root / "complete.json"
        if path.is_symlink() or (root / "failed.json").exists() or digest(path) != h:
            raise ValueError("exact closed immutable physics evidence required")
        reports.append(json.loads(path.read_text()))
    source, audit, prior = reports
    dp = a.source_root / "declaration.json"
    dec = json.loads(dp.read_text())
    if (
        source["status"] != "NEURAL_RELAY_REFERENCE_BRIDGE_COLLECTED_PENDING_OWN_REPLAY"
        or source["declaration_hash"] != digest(dp)
        or dec["reference_bridge_controls"] != [0, 5, 15, 30]
        or dec["prior_closed_relay"]["relay_hash"] != a.prior_hash
        or audit["status"] != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
        or audit["producer_hash"] != a.source_hash
        or audit["worlds"] != 128
        or audit["counts"] != source["counts"]
        or len(audit["records"]) != 128
        or prior["status"] != "NEURAL_RELAY32_COLLECTED_PENDING_OWN_REPLAY"
        or len(prior["records"]) != 32
        or any(
            r[k] is not False
            for r in reports
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("whole128 own-native closure and original32 baseline required")

    from rosclaw_soccer.rsi.neural_relay_comparison import compare_relay_arms, relay_key

    bound = {}

    def bind(root, ci, item):
        if item is None:
            return None
        path = root / item["path"]
        if (
            path.is_symlink()
            or path.resolve().parent != (root / f"trial{ci}").resolve()
            or digest(path) != item["hash"]
        ):
            raise ValueError("confined exact raw stream required")
        bound[str(path)] = item["hash"]
        return item["hash"]

    old = {relay_key(r): r for r in prior["records"]}
    identity = []
    for row, own in zip(source["records"], audit["records"], strict=True):
        ci = row["index"]
        if (own["index"], own["course"], own["termination"], own["result"]) != (
            ci,
            row["course"],
            row["termination"],
            row["result"],
        ):
            raise ValueError("exact source and own replay outcomes required")
        for artifact in row["artifacts"].values():
            bind(a.source_root, ci, artifact)
        if row["handoff"]:
            bind(a.source_root, ci, row["handoff"]["boundary"])
        if row["course"]["arm"]["reference_bridge_controls"]:
            continue
        previous = old[relay_key(row)]
        if row["result"] != previous["result"] or row["termination"] != previous["termination"]:
            raise ValueError("zero-reference baseline outcome changed")
        if set(row["artifacts"]) != set(previous["artifacts"]):
            raise ValueError("zero-reference baseline raw surface changed")
        for key, artifact in row["artifacts"].items():
            if bind(a.source_root, ci, artifact) != bind(
                a.prior_root, previous["index"], previous["artifacts"][key]
            ):
                raise ValueError("zero-reference whole raw stream not byte-identical:" + key)
        if bool(row["handoff"]) != bool(previous["handoff"]):
            raise ValueError("zero-reference transition changed")
        if row["handoff"] and bind(a.source_root, ci, row["handoff"]["boundary"]) != bind(
            a.prior_root, previous["index"], previous["handoff"]["boundary"]
        ):
            raise ValueError("zero-reference physical/memory transition not byte-identical")
        identity.append(dict(current=ci, original=previous["index"], whole_raw_byte_identical=True))
    if len(identity) != 32:
        raise ValueError("all32 old baseline executions must be retained")
    result = compare_relay_arms(source["records"])
    improved = [r for r in result["summaries"][1:] if r["precise_net_change"] > 0]
    findings = dict(
        discovered="Local TRAIN reference-conditioning gain"
        if improved
        else "No precise-healthy TRAIN net gain from reference bridge",
        ruled_out=(
            "Original zero-bridge baseline unchanged; all32 whole raw streams byte-identical"
        ),
        next_investment="Validate teacher on new perturbations before own-state student learning"
        if improved
        else (
            "Pivot from reference-only family to successor-state geometry/gain/phase mechanisms; "
            "do not start blind PPO"
        ),
        falsifier=(
            "Gains vanish on new perturbations or old successes regress; "
            "no Fresh/promotion claim from these8 consumed conditions"
        ),
    )
    clean(a.expected_head)
    write_once(
        a.output,
        dict(
            schema="soccer.neural_relay_reference_comparison.v1",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            prior_hash=a.prior_hash,
            raw_file_hashes=bound,
            baseline_identity=identity,
            **result,
            findings=findings,
            physics_steps=0,
            hardware_authorized=False,
        ),
    )
    print(json.dumps(dict(summaries=result["summaries"], findings=findings)), flush=True)


if __name__ == "__main__":
    main()
