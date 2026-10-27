"""Fail-closed research admission for an independently replayed offline selector.

Admission only permits planning a further SIM_ONLY physics experiment; it is
not a controller deployment, a safety certificate, or a generalization claim.
"""

import argparse
import json
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def assess(source: dict, audit: dict) -> dict:
    if (
        source.get("status") != "RELAY_BALANCE_OFFLINE_SELECTOR_HOLDOUT_NOT_PHYSICS"
        or audit.get("status") != "RELAY_BALANCE_SELECTOR_NUMPY_REPLAY_NOT_CONTROL"
        or audit.get("dataset_hash") != source.get("dataset_hash")
        or audit.get("independent_numpy_inference") is not True
        or any(
            source.get(k) is not False
            for k in (
                "physics_evaluated_as_controller",
                "safety_certificate",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            audit.get(k) is not False
            for k in (
                "physics_evaluated_as_controller",
                "safety_certificate",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("offline training and independent inference audit required")
    contexts = source.get("contexts")
    rows, audited = source.get("rows"), audit.get("rows")
    if not isinstance(contexts, list) or not contexts or len(set(contexts)) != len(contexts):
        raise ValueError("unique held-out contexts required")
    expected = {(c, s) for c in contexts for s in (17, 43, 71)}
    if not isinstance(rows, list) or not isinstance(audited, list):
        raise ValueError("complete selector and audit rows required")
    if (
        len(rows) != len(expected)
        or {(r.get("context"), r.get("seed")) for r in rows} != expected
        or len(audited) != len(expected)
        or {(r.get("context"), r.get("seed")) for r in audited} != expected
        or any(r.get("choices_replayed") is not True for r in audited)
    ):
        raise ValueError("every context and all three seeds must be independently audited")
    audits = {(r["context"], r["seed"]): r for r in audited}
    metrics = {
        s: dict(
            decisions=0,
            interventions=0,
            precise_rescues=0,
            precise_regressions=0,
            failure_rescues=0,
            failure_regressions=0,
        )
        for s in (17, 43, 71)
    }
    for row in rows:
        choices = row.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or len(choices) != row.get("heldout_decisions")
            or len(choices) != audits[(row["context"], row["seed"])].get("heldout_decisions")
        ):
            raise ValueError("all held-out choices must be replayed")
        m = metrics[row["seed"]]
        for c in choices:
            fields = ("parent_success", "chosen_success", "parent_failure", "chosen_failure")
            if (
                any(type(c.get(k)) is not bool for k in fields)
                or type(c.get("action")) is not int
                or c["action"] not in range(4)
            ):
                raise ValueError("invalid offline outcome or action")
            m["decisions"] += 1
            m["interventions"] += c["action"] != 0
            m["precise_rescues"] += not c["parent_success"] and c["chosen_success"]
            m["precise_regressions"] += c["parent_success"] and not c["chosen_success"]
            m["failure_rescues"] += c["parent_failure"] and not c["chosen_failure"]
            m["failure_regressions"] += not c["parent_failure"] and c["chosen_failure"]
    admitted = all(
        m["interventions"] > 0
        and m["precise_rescues"] > 0
        and m["precise_regressions"] == 0
        and m["failure_regressions"] == 0
        for m in metrics.values()
    )
    return dict(
        status="SELECTOR_RESEARCH_PHYSICS_ELIGIBLE"
        if admitted
        else "SELECTOR_RESEARCH_PHYSICS_REJECTED",
        metrics=metrics,
        seed_cherry_picking=False,
        physics_evaluated_as_controller=False,
        safety_certificate=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "audit-hash", "expected-head"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    sp, ap = args.source / "complete.json", args.audit / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or sp.is_symlink()
        or ap.is_symlink()
        or digest(sp) != args.source_hash
        or digest(ap) != args.audit_hash
    ):
        raise ValueError("exact source and audit, and unused external output required")
    source, audit = json.loads(sp.read_text()), json.loads(ap.read_text())
    if audit.get("source_hash") != args.source_hash:
        raise ValueError("audit must bind exact selector")
    for row in source.get("rows", []):
        path = args.source / row["checkpoint"]
        if (
            path.is_symlink()
            or path.resolve().parent != args.source.resolve()
            or digest(path) != row["checkpoint_hash"]
        ):
            raise ValueError("selector checkpoint changed or escaped source")
    result = assess(source, audit)
    if digest(sp) != args.source_hash or digest(ap) != args.audit_hash:
        raise ValueError("source or audit changed during admission")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            **result,
            source_hash=args.source_hash,
            audit_hash=args.audit_hash,
            source_head=args.expected_head,
        ),
    )


if __name__ == "__main__":
    main()
