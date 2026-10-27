"""Independently re-score twelve new development courses from SIM snapshots.

No physics reintegration, Fresh exam, hardware authority or promotion. Global
substep safety remains a source receipt, not independently certified.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.team_coverage_actor import TeamCoverageActor
from rosclaw_soccer.rsi.team_coverage_admission import clean_reception, recorded_hazard
from rosclaw_soccer.rsi.team_trace_capture import extract_team_trace
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_audit_team_receiving_trace import PHYSICAL_KEYS, ROSTER
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "models", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-hash", "models-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    source_path, model_path = args.source / "complete.json", args.models / "complete.json"
    paths, hashes = (source_path, model_path), (args.source_hash, args.models_hash)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in zip(paths, hashes, strict=True))
    ):
        raise ValueError("exact source and unused external output required")
    report, training = (json.loads(p.read_text()) for p in paths)
    new_guard_dev = report["status"] == "TEAM_RECEIVING_RADIUS_GUARD_NEW_DEVELOPMENT_NOT_PROMOTED"
    guard = new_guard_dev or report["status"] == "TEAM_RECEIVING_NEIGHBOR_RADIUS_GUARD_NOT_PROMOTED"
    if (
        report["status"]
        not in (
            "TEAM_RECEIVING_NEIGHBOR_DEVELOPMENT_NOT_PROMOTED",
            "TEAM_RECEIVING_NEIGHBOR_RADIUS_GUARD_NOT_PROMOTED",
            "TEAM_RECEIVING_RADIUS_GUARD_NEW_DEVELOPMENT_NOT_PROMOTED",
        )
        or report["physical_runs"] != 48
        or report["fresh_exam"] is not False
        or report["promotion_authorized"] is not False
        or report.get("effective_coverage_model_hash", report["learned_coverage_training_hash"])
        != args.models_hash
        or report["course_bank_hash"]
        != (
            "sha256:1919af2c823638018fe0415d0bb7878c003d1f695385e1431ead37aa7c335e3d"
            if new_guard_dev
            else "sha256:60a30fe74e052a141dd295f00b74cec71ef8c708612adfbd6c068f3c34359398"
        )
        or training["status"]
        != (
            "TEAM_COVERAGE_FAILURE_DERIVED_RADIUS_NOT_EVALUATED"
            if guard
            else "TEAM_COVERAGE_SUPERVISED_ACTORS_NOT_EVALUATED"
        )
    ):
        raise ValueError("exact new development and frozen actor lineage required")
    models = {}
    for record in training["records"]:
        path = args.models / record["model_path"]
        if path.is_symlink() or path.parent != args.models or digest(path) != record["model_hash"]:
            raise ValueError("exact confined actor checkpoint required")
        models[f"learned-{record['seed']}"] = TeamCoverageActor.from_dict(
            json.loads(path.read_text())
        )
    if set(models) != {"learned-17", "learned-43", "learned-71"} or [
        g["course_seed"] for g in report["results"]
    ] != (list(range(420101, 420113)) if new_guard_dev else list(range(410001, 410013))):
        raise ValueError("all actors and exact preregistered development seeds required")
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_hash=args.source_hash,
            models_hash=args.models_hash,
            source_head=args.expected_head,
            trace_count=48,
            physical_integrations=0,
            global_safety_not_independently_recomputed=True,
            hardware_authorized=False,
        ),
    )
    rows, statistics = (
        [],
        {arm: {"success": 0, "gains": 0, "losses": 0, "new_recorded_hazards": 0} for arm in models},
    )
    parent_success = 0
    for group in report["results"]:
        seed = group["course_seed"]
        arms = {r["arm"]: r for r in group["rows"]}
        if len(group["rows"]) != 4 or set(arms) != {"parent", *models}:
            raise ValueError("one parent and all three actors required")
        parent = arms["parent"]
        p = clean_reception(parent)
        parent_success += p
        for arm, row in arms.items():
            path = args.source / row["trace_file"]
            if (
                path.is_symlink()
                or path.parent != args.source
                or digest(path) != row["trace_file_hash"]
            ):
                raise ValueError("exact confined physical trace required")
            with np.load(path, allow_pickle=False) as archive:
                trace = {key: archive[key].copy() for key in archive.files}
            if (
                len(trace) != 106
                or hash_json({k: trace[k].tolist() for k in PHYSICAL_KEYS})
                != row["physical_trace_hash"]
            ):
                raise ValueError("complete tactical trace and original physical hash required")
            extracted = extract_team_trace(trace, require_tactical=True)
            _, outcome = receiving_window(
                extracted, agent_ids=ROSTER, agent_id="red.finisher", start=20, frames=100
            )
            explanation = explain_receiving_window(
                extracted, agent_ids=ROSTER, agent_id="red.finisher", start=20, frames=100
            )
            if outcome["controlled_reception"] != row["controlled_reception"]:
                raise ValueError("independent receiving score differs from receipt")
            if arm != "parent":
                action = row["selector_action"]
                if not np.array_equal(row["activation_features"], parent["activation_features"]):
                    raise ValueError("unpaired measured activation states")
                if action == 16:
                    if row["physical_trace_hash"] != parent["physical_trace_hash"]:
                        raise ValueError("fallback must physically reproduce parent")
                elif (
                    parent["activation_enabled"]
                    or parent["protected_episode"]
                    or models[arm].select(tuple(row["activation_features"]))[0] != action
                ):
                    raise ValueError("intervention violates parent gate or actor decision")
                success = clean_reception(row)
                statistics[arm]["success"] += success
                statistics[arm]["gains"] += success and not p
                statistics[arm]["losses"] += p and not success
                statistics[arm]["new_recorded_hazards"] += recorded_hazard(
                    row
                ) and not recorded_hazard(parent)
            item = dict(
                course_seed=seed,
                arm=arm,
                controlled_reception=outcome["controlled_reception"],
                window=outcome,
                criteria=explanation["criteria"],
                raw_trace_hash=row["trace_file_hash"],
                physical_trace_hash=row["physical_trace_hash"],
                source_global_safety=row["safe"],
                source_fault_agents=row["fault_agents"],
                selector_action=row.get("selector_action"),
                global_safety_not_independently_recomputed=True,
            )
            rows.append(item)
            write_once(args.output / f"{seed}-{arm}.json", item)
    if any(digest(p) != h for p, h in zip(paths, hashes, strict=True)):
        raise ValueError("source changed during audit")
    write_once(
        args.output / "complete.json",
        dict(
            status="TEAM_COVERAGE_NEIGHBOR_WINDOW_RECOMPUTED_NOT_REINTEGRATED",
            source_hash=args.source_hash,
            models_hash=args.models_hash,
            source_head=args.expected_head,
            rows=rows,
            parent_success=parent_success,
            candidates=statistics,
            posthoc_failure_guard=guard,
            new_guard_development=new_guard_dev,
            trace_count=len(rows),
            physical_integrations=0,
            global_safety_not_independently_recomputed=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
