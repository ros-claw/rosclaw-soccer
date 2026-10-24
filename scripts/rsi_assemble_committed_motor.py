"""Assemble the audited three-expert neural option bundle; never fit or simulate."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("own-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument(
        "--expert", nargs=2, action="append", required=True, metavar=("SEED", "TRAINING_HASH")
    )
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    op = a.own_root / "complete.json"
    if (
        a.output.exists()
        or op.is_symlink()
        or (a.own_root / "failed.json").exists()
        or digest(op) != a.own_hash
        or [int(x[0]) for x in a.expert] != [115023, 115024, 115025]
    ):
        raise ValueError("unused output and exact own97/three ordered source experts required")
    own = json.loads(op.read_text())
    pins = dict(own["input_file_hashes"])
    pins[str(op)] = a.own_hash
    _, feasibility = receipt(pins, own["feasibility_hash"])
    _, parent = receipt(pins, feasibility["parent_training_hash"])
    experts = [receipt(pins, h)[1] for _, h in a.expert]
    from rosclaw_soccer.rsi.committed_motor_checkpoint import option_ancestry
    from rosclaw_soccer.rsi.neural_option_contract import IDENTITIES, KIND, STATUS, option_config

    ids = dict(
        parent_training_hash=feasibility["parent_training_hash"],
        parent_lineage_hash=parent["parent_lineage_hash"],
        parent_checkpoint_hash=parent["checkpoint_hash"],
        data_hash=parent["data_hash"],
        feasibility_hash=own["feasibility_hash"],
        initiation_hash=feasibility["initiation_hash"],
        own97_hash=a.own_hash,
        expert_training_hashes=[h for _, h in a.expert],
        expert_checkpoint_hashes=[r["checkpoint_hash"] for r in experts],
        generation_hash=own["generation_hash"],
    )
    if set(ids) != set(IDENTITIES):
        raise ValueError("complete fixed reference bundle identities required")
    option_ancestry(dict(**ids, input_file_hashes=pins))
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    cp = a.output / "committed-neural-motor.json"
    write_once(cp, dict(schema=KIND, **ids))
    declared = dict(
        **ids,
        source_head=a.expected_head,
        input_file_hashes=pins,
        config=option_config(),
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
