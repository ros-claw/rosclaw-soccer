"""Close a predeclared option composition against its exact whole learned parent.

No new optimizer steps, three-seed training claim, physical replay, Fresh or
promotion. Full source/own evidence must already exist for BOTH80/44 protocols.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("parent-pass-root", "parent-retention-root", "pass-root", "retention-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("parent-pass-hash", "parent-retention-hash", "training-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused read-only physical review output required")
    pins = {}

    def closed(root, h=None):
        path = root / "complete.json"
        if (
            root.is_symlink()
            or path.is_symlink()
            or (root / "failed.json").exists()
            or (h is not None and digest(path) != h)
        ):
            raise ValueError("exact closed nonfailed independent native evidence required")
        r = json.loads(path.read_text())
        for key, value in r["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting physical source identities")
            pins[key] = value
        pins[str(path)] = digest(path)
        return r

    pp, pr = (
        closed(a.parent_pass_root, a.parent_pass_hash),
        closed(a.parent_retention_root, a.parent_retention_hash),
    )
    cp, cr = closed(a.pass_root), closed(a.retention_root)
    from rosclaw_soccer.rsi.acquired_motor_retention import native_successes
    from rosclaw_soccer.rsi.committed_motor_checkpoint import option_ancestry
    from rosclaw_soccer.rsi.native_growth_evidence import qualified_native_growth_pair
    from rosclaw_soccer.rsi.neural_option_contract import KIND, option_config
    from rosclaw_soccer.rsi.refinement_growth_review import paired_refinement_growth

    parent, ps = qualified_native_growth_pair(
        pp, pr, kind="consolidated-adapter-v1", pins=pins, resolve=lambda h: receipt(pins, h)
    )
    candidate, cs = qualified_native_growth_pair(
        cp, cr, kind=KIND, pins=pins, resolve=lambda h: receipt(pins, h)
    )
    if (
        cp["training_hash"] != a.training_hash
        or candidate["parent_training_hash"] != pp["training_hash"]
        or candidate["parent_checkpoint_hash"] != parent["checkpoint_hash"]
        or candidate["parent_lineage_hash"] != parent["parent_lineage_hash"]
        or candidate["config"] != option_config()
        or candidate["new_optimizer_updates"] != 0
        or ps["contexts"] != cs["contexts"]
    ):
        raise ValueError(
            "same exact whole parent, original fixtures and zero-new-update composition required"
        )
    option_ancestry(candidate)
    protected = native_successes(pr["results"])
    growth = paired_refinement_growth(
        pp["records"],
        cp["records"],
        required_old_contexts=protected,
        achieved_old_contexts=native_successes(cr["results"]),
    )
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="COMMITTED_NEURAL_OPTIONS_WHOLE_PHYSICAL_GROWTH_REVIEWED_NOT_PROMOTED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            training_hash=a.training_hash,
            parent_training_hash=pp["training_hash"],
            pass_own_hash=digest(a.pass_root / "complete.json"),
            retention_own_hash=digest(a.retention_root / "complete.json"),
            protected_old_contexts=sorted(protected),
            physical_growth=growth,
            new_optimizer_updates=0,
            reused_neural_head_updates_per_seed=2048,
            reused_head_seeds=[115023, 115024, 115025],
            new_independent_training_seeds=0,
            physical_steps_executed=0,
            Fresh_verified=False,
            team_game_verified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
