"""Close every declared new-head seed against its whole acquired parent.

Read-only native effect review, not physics, optimization, Fresh or promotion.
No candidate may be omitted because its outcome was disappointing.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("parent-pass-root", "parent-retention-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("parent-pass-hash", "parent-retention-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument(
        "--candidate",
        nargs=4,
        action="append",
        required=True,
        metavar=("SEED", "TRAINING_HASH", "PASS_OWN_ROOT", "RETENTION_OWN_ROOT"),
    )
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists() or [int(c[0]) for c in a.candidate] != [115023, 115024, 115025]:
        raise ValueError("unused output and all three preregistered independent seeds required")
    pins = {}

    def closed(root, expected=None):
        path = root / "complete.json"
        if (
            root.is_symlink()
            or path.is_symlink()
            or (root / "failed.json").exists()
            or (expected is not None and digest(path) != expected)
        ):
            raise ValueError("exact closed nonfailed native evidence required")
        doc = json.loads(path.read_text())
        for key, value in doc["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting physical proof lineage")
            pins[key] = value
        pins[str(path)] = digest(path)
        if any(
            doc[k] is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        ):
            raise ValueError("unpromoted SIM-only physical research required")
        return doc

    def validate_pair(passed, retained, kind):
        if (
            passed["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
            or retained["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
            or len(passed["records"]) != 80
            or retained["worlds"] != 44
            or retained["independent_whole_world_replay"] is not True
            or retained["checkpoint_kind"] != kind
            or passed["training_hash"] != retained["training_hash"]
        ):
            raise ValueError("same model with complete independent pass80 and retention44 required")
        _, source = receipt(pins, passed["producer_hash"])
        old_path, old_source = receipt(pins, retained["source_hash"])
        _, training = receipt(pins, passed["training_hash"])
        if (
            source["training_hash"] != passed["training_hash"]
            or source["checkpoint_kind"] != kind
            or [{k: v for k, v in row.items() if k != "artifacts"} for row in source["records"]]
            != passed["records"]
            or old_source["training_hash"] != passed["training_hash"]
            or training["checkpoint_hash"] != retained["checkpoint_hash"]
        ):
            raise ValueError("exact source and independently reproduced own results required")
        declaration_path = old_path.parent / "declaration.json"
        if (
            declaration_path.is_symlink()
            or pins.get(str(declaration_path)) != digest(declaration_path)
            or old_source["declaration_hash"] != digest(declaration_path)
        ):
            raise ValueError("source-bound whole retention task witness required")
        declaration = json.loads(declaration_path.read_text())
        if declaration["protocol"] != "dual256.unassisted_TRAIN20_and_consumed_run2.v1":
            raise ValueError("unchanged whole22 retention protocol required")
        return training, declaration

    parent_pass = closed(a.parent_pass_root, a.parent_pass_hash)
    parent_old = closed(a.parent_retention_root, a.parent_retention_hash)
    parent_training, parent_scope = validate_pair(
        parent_pass, parent_old, "consolidated-adapter-v1"
    )
    from rosclaw_soccer.rsi.acquired_motor_retention import native_successes
    from rosclaw_soccer.rsi.refinement_growth_review import paired_refinement_growth

    protected = native_successes(parent_old["results"])
    reports = []
    for seed, training_hash, pass_root, old_root in a.candidate:
        passed, retained = closed(Path(pass_root)), closed(Path(old_root))
        training, scope = validate_pair(passed, retained, "consolidated-refinement-v1")
        if (
            passed["training_hash"] != training_hash
            or training["config"]["seed"] != int(seed)
            or training["parent_training_hash"] != parent_pass["training_hash"]
            or training["parent_checkpoint_hash"] != parent_training["checkpoint_hash"]
            or training["new_head_steps"] != 2048
            or scope["contexts"] != parent_scope["contexts"]
        ):
            raise ValueError("all three exact new heads must derive from the same whole parent")
        report = paired_refinement_growth(
            parent_pass["records"],
            passed["records"],
            required_old_contexts=protected,
            achieved_old_contexts=native_successes(retained["results"]),
        )
        reports.append(
            dict(
                seed=int(seed),
                training_hash=training_hash,
                pass_own_hash=digest(Path(pass_root) / "complete.json"),
                retention_own_hash=digest(Path(old_root) / "complete.json"),
                checkpoint_hash=training["checkpoint_hash"],
                new_head_updates=2048,
                **report,
            )
        )
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="ALL_THREE_WHOLE_MOTOR_REFINEMENTS_PHYSICALLY_REVIEWED_NOT_PROMOTED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            parent_training_hash=parent_pass["training_hash"],
            protected_old_contexts=sorted(protected),
            reports=reports,
            physical_steps_executed=0,
            optimizer_updates=0,
            Fresh_verified=False,
            team_game_verified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
