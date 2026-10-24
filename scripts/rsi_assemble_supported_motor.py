"""Assemble an explicit frozen neural/support reference bundle; NO training."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("training-root", "gate-audit-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("training-hash", "gate-audit-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    tp, gp = a.training_root / "complete.json", a.gate_audit_root / "complete.json"
    if (
        a.output.exists()
        or digest(tp) != a.training_hash
        or digest(gp) != a.gate_audit_hash
        or any(p.is_symlink() or (p.parent / "failed.json").exists() for p in (tp, gp))
    ):
        raise ValueError("unused output and exact closed neural/support sources required")
    t, g = [json.loads(p.read_text()) for p in (tp, gp)]
    pins = dict(t["input_file_hashes"])
    for k, v in g["input_file_hashes"].items():
        if k in pins and pins[k] != v:
            raise ValueError("conflicting neural/support ancestry")
        pins[k] = v
    pins.update({str(tp): a.training_hash, str(gp): a.gate_audit_hash})
    _, geometry = receipt(pins, g["geometry_hash"])
    if (
        t["checkpoint_kind"] != "consolidated-refinement-v1"
        or t["new_head_steps"] != 2048
        or g["status"] != "ALL_PROTECTED_FEATURES_ZERO_RESCUE_ANCHORS_ONE_GATE_AUDITED_NOT_MOTOR"
        or t["parent_training_hash"] != g["parent_training_hash"]
        or any(
            d[k] is not False
            for d in (t, g)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("same unpromoted trained neural lineage and qualified support required")
    pins[t["checkpoint_path"]] = t["checkpoint_hash"]
    source_pins(pins)
    check_pins(pins)
    from rosclaw_soccer.rsi.supported_motor_checkpoint import (
        IDENTITIES,
        KIND,
        STATUS,
        support_cpu_scope,
    )

    ids = dict(
        source_refinement_training_hash=a.training_hash,
        source_refinement_checkpoint_hash=t["checkpoint_hash"],
        **{
            k: t[k]
            for k in ("parent_training_hash", "parent_lineage_hash", "bank_hash", "feature_hash")
        },
        retention_feature_hash=geometry["retention_feature_hash"],
        geometry_hash=g["geometry_hash"],
        gate_audit_hash=a.gate_audit_hash,
    )
    if (
        set(ids) != set(IDENTITIES)
        or support_cpu_scope(dict(**ids, input_file_hashes=pins)) != 28977
    ):
        raise ValueError("complete fixed79-history support declaration required")
    a.output.mkdir()
    cp = a.output / "supported-motor.json"
    write_once(cp, dict(schema=KIND, **ids))
    declared = dict(
        **ids,
        source_head=a.expected_head,
        input_file_hashes=pins,
        config=dict(
            seed=t["config"]["seed"],
            new_optimizer_updates=0,
            reused_head_updates=2048,
            support_schema=1,
        ),
        checkpoint_path=str(cp),
        checkpoint_hash=digest(cp),
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    write_once(a.output / "declaration.json", declared)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            **declared,
            status=STATUS,
            checkpoint_kind=KIND,
            declaration_hash=digest(a.output / "declaration.json"),
            checkpoint_format="json.reference_bundle.v1",
            new_optimizer_updates=0,
            physical_steps_executed=0,
            old_retention_proven=False,
            online_RL_claim=False,
        ),
    )


if __name__ == "__main__":
    main()
