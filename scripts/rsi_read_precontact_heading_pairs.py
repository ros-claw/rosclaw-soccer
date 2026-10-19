"""Whole nine paired current-state nodes from252 verified SIM action bodies.

Standard-library source pins precede ALL project imports. This is raw
counterfactual dataset extraction, no inference, fitting or policy activation.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", required=True, type=Path)
    parser.add_argument("--physics-root", required=True, type=Path)
    parser.add_argument("--physics-unit", required=True)
    parser.add_argument("--readback", required=True, type=Path)
    parser.add_argument("--readback-file-hash", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--core-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent

    def digest(file):
        with Path(file).open("rb") as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()

    if (
        args.output.exists()
        or not args.core_source.is_absolute()
        or args.core_source.is_symlink()
        or not (args.core_source / "rosclaw/growth/canonical_json_snapshot.py").is_file()
        or (args.physics_root / "failure.json").exists()
        or digest(args.readback) != args.readback_file_hash
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("clean frozen source, closed proof and unused pair output required")
    pins = {}
    for folder in (source / "src", source / "scripts", args.core_source):
        for file in folder.rglob("*.py"):
            if file.is_symlink():
                raise ValueError("ordinary frozen source files required")
            pins[str(file)] = digest(file)
    # No project or optional numerical imports occur above this boundary.
    import numpy as np
    import rosclaw.growth.canonical_json_snapshot as snapshot

    from rosclaw_soccer.rsi.precontact_heading_pairs import PAIRS, paired_context
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_read_postcontact_full_bank import check_complete
    from scripts.rsi_replay_postcontact_full_bank_physics import require_full_physical_review
    from scripts.rsi_stage_release_contact_replay import require_terminal

    if (
        Path(snapshot.__file__).resolve()
        != (args.core_source / "rosclaw/growth/canonical_json_snapshot.py").resolve()
    ):
        raise ValueError("actually loaded Core must match the pre-import pinned source")
    require_terminal(args.physics_unit)

    def pin(file, expected=None):
        file = Path(file)
        h = digest(file)
        if (
            file.is_symlink()
            or (expected is not None and h != expected)
            or (str(file) in pins and pins[str(file)] != h)
        ):
            raise ValueError("same whole raw/physical source bytes required")
        pins[str(file)] = h

    def sealed(file, expected=None):
        pin(file, expected)
        d = json.loads(Path(file).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole paired bank input seal required")
        return d

    rb = sealed(args.readback, args.readback_file_hash)
    c, d = sealed(args.bank_root / "complete.json"), sealed(args.bank_root / "declaration.json")
    check_complete(c, d, args.bank_root)
    proof = sealed(args.physics_root / "complete.json")
    pd = sealed(args.physics_root / "declaration.json")
    if (
        rb["schema"] != "soccer.rsi.postcontact_full_bank_readback.v1"
        or rb["complete_hash"] != c["report_hash"]
        or len(rb["contexts"]) != 18
        or proof["schema"] != "soccer.rsi.recorded_postcontact_full_bank_complete.v1"
        or proof["declaration_hash"] != pd["report_hash"]
        or pd["raw_readback_hash"] != rb["report_hash"]
        or pd["producer_complete_hash"] != c["report_hash"]
        or len(proof["rows"]) != 252
        or any(
            type(proof.get(k)) is not int or proof[k] != v
            for k, v in dict(
                comparison_bodies_verified=252,
                new_physical_bodies=238,
                new_physical_steps=714000,
                reused_physical_bodies=14,
                reused_physical_steps=42000,
                new_neural_inferences=0,
                new_optimizer_updates=0,
            ).items()
        )
        or any(
            proof.get(k) is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("same whole independently replayed252-body diagnostic bank required")
    for parent in (rb, pd):
        for name, h in parent["input_file_hashes"].items():
            pin(name, h)
    for ci, context in enumerate(c["contexts"]):
        if (
            context != sealed(args.bank_root / f"context-{ci}-complete.json")
            or rb["contexts"][ci]["source_context_hash"] != context["report_hash"]
        ):
            raise ValueError("all eighteen original raw/physical contexts required")
        for ai, row in enumerate(context["rows"]):
            pr = proof["rows"][ci * 14 + ai]
            if (
                row != sealed(args.bank_root / f"context-{ci}-action-{ai}-complete.json")
                or pr != sealed(args.physics_root / f"context-{ci}-action-{ai}-complete.json")
                or pr["context_index"] != ci
                or pr["action_index"] != ai
                or pr["source_bank_case_hash"] != row["report_hash"]
                or pr["physical_execution_reused"] is not (ci == 1)
            ):
                raise ValueError("all252 complete matched physical body proofs required")
            require_full_physical_review(pr["physical_review"])
            pin(row["arrays"]["path"], row["arrays"]["sha256"])
    if args.preflight:
        print("NINE_HEADING_PAIRS_WHOLE_PHYSICAL_CHAIN_READY_NO_EXECUTION", flush=True)
        return
    args.output.mkdir()
    declaration = dict(
        schema="soccer.rsi.precontact_heading_pairs_declaration.v1",
        source_head=args.source_head,
        input_file_hashes=pins,
        source_pinned_before_import=True,
        producer_complete_hash=c["report_hash"],
        raw_readback_hash=rb["report_hash"],
        physical_complete_hash=proof["report_hash"],
        pair_indices=[list(p) for p in PAIRS],
        distinct_courses=9,
        local_courses=6,
        consumed_retention_courses=3,
        new_physical_step_budget=0,
        new_optimizer_update_budget=0,
        mode="SIM_ONLY",
        teacher_qualified=False,
        motor_policy_activation_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(args.output / "declaration.json", declaration)
    rows = []
    try:
        for index, (oi, ni) in enumerate(PAIRS):
            matrices = []
            for ci in (oi, ni):
                traces = []
                for row in c["contexts"][ci]["rows"]:
                    with np.load(row["arrays"]["path"], allow_pickle=False) as archive:
                        traces.append(dict(archive))
                matrices.append(traces)
            pair = paired_context(
                c["contexts"][oi], c["contexts"][ni], (matrices[0], matrices[1]), index=index
            )
            write_once(args.output / f"pair-{index}-complete.json", pair)
            rows.append(pair)
            print(
                "PAIRED_PRECONTACT_NODE_COMPLETE",
                index,
                pair["observed_safe_passing_actions_by_heading"],
                flush=True,
            )
        if (
            len({tuple(r["course"]) for r in rows}) != 9
            or len({r["current_features_hash"] for r in rows}) != 9
            or {r["source_cluster"] for r in rows} != {0, 1, 2}
        ):
            raise ValueError("nine distinct original courses and observed decision states required")
        if any(digest(n) != h for n, h in pins.items()):
            raise ValueError("whole paired data and pre-import source changed during readback")
        result = dict(
            schema="soccer.rsi.precontact_heading_pairs_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            distinct_predecision_states=9,
            local_training_candidates=6,
            consumed_retention_evaluation_only=3,
            original_source_clusters=3,
            unique_heading_action_labels=234,
            original_comparison_bodies_covered=252,
            new_physical_steps=0,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            training_authorized=False,
            teacher_qualified=False,
            motor_policy_activation_authorized=False,
            hardware_authorized=False,
            runtime_future_outcome_table_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(args.output / "complete.json", result)
    except BaseException as error:
        write_once(
            args.output / "failure.json",
            dict(
                completed_pairs=len(rows),
                new_physical_steps=0,
                new_optimizer_updates=0,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
