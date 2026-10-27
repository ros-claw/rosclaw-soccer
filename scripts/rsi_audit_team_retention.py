"""Read back complete 55-course team selector retention without reintegration.

Checks all source trace hashes, actor selections and paired reported outcomes.
Only nine legacy physical fields were recorded, so original controlled/safety
labels cannot be independently recomputed from this particular experiment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked

from rosclaw_soccer.rsi.team_receiving_selector import TeamReceivingSelector
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_team_selector_smoke import PHYSICAL_KEYS, TRAINING_HASH

PARENT_HASH = "sha256:52f70e48bb39f050350ee103c72bb3000aea47a3f594aa96f35434d9b008c282"


def clean_controlled(row):
    return bool(
        row["safe"]
        and not row["fault_agents"]
        and row["first_foot_frame"] is not None
        and not row["own_nonfoot_frames"]
        and row["controlled_reception"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    source = args.source / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or source.is_symlink()
        or digest(source) != args.source_hash
    ):
        raise ValueError("unused output and exact complete retention source required")
    report = json.loads(source.read_text())
    parent = _checked(args.parent_report)
    training_path = args.training_dir / "complete.json"
    if (
        parent["report_hash"] != PARENT_HASH
        or digest(training_path) != TRAINING_HASH
        or report["status"] != "TEAM_RECEIVING_EIGHT_BODY_FULL_CONSUMED_NOT_PROMOTED"
        or report["physical_runs"] != 220
        or len(report["results"]) != 55
        or report["parent_report_hash"] != PARENT_HASH
        or report["training_hash"] != TRAINING_HASH
        or report["fresh_authorized"] is not False
        or report["promotion_authorized"] is not False
    ):
        raise ValueError("exact unpromoted complete paired retention lineage required")
    train = json.loads(training_path.read_text())
    models = {}
    for item in train["records"]:
        path = args.training_dir / item["model_path"]
        if (
            path.is_symlink()
            or path.parent != args.training_dir
            or digest(path) != item["model_hash"]
        ):
            raise ValueError("exact original trained model required")
        models[str(item["seed"])] = TeamReceivingSelector.from_dict(json.loads(path.read_text()))
    if set(models) != {"17", "43", "71"}:
        raise ValueError("all three actor seeds required")
    historical = {r["course"]["seed"]: r for r in parent["results"][0]["rows"]}
    if len(historical) != 55:
        raise ValueError("complete distinct parent cohort required")
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_hash=args.source_hash,
            source_head=args.expected_head,
            parent_hash=PARENT_HASH,
            training_hash=TRAINING_HASH,
            expected_courses=55,
            expected_runs=220,
            physical_integrations=0,
            safety_and_receiving_labels_not_recomputed=True,
            hardware_authorized=False,
        ),
    )
    rows = []
    for group in report["results"]:
        seed = group["course_seed"]
        if seed not in historical:
            raise ValueError("unrecognized original retention course")
        arms = {row["arm"]: row for row in group["rows"]}
        if set(arms) != {"parent", "17", "43", "71"}:
            raise ValueError("exactly four paired arms per course required")
        for arm, row in arms.items():
            path = args.source / row["trace_file"]
            if (
                path.is_symlink()
                or path.parent != args.source
                or digest(path) != row["trace_file_hash"]
            ):
                raise ValueError("exact retained numeric physical trace required")
            with np.load(path, allow_pickle=False) as f:
                arrays = {key: f[key].copy() for key in PHYSICAL_KEYS}
            if any(
                value.dtype.kind not in "fiu" or not np.isfinite(value).all()
                for value in arrays.values()
            ):
                raise ValueError("finite numeric physical arrays required")
            if (
                hash_json({key: arrays[key].tolist() for key in PHYSICAL_KEYS})
                != row["physical_trace_hash"]
            ):
                raise ValueError("original physical commitment mismatch")
            if arm != "parent":
                action = 0
                if row["activation_enabled"] and not row["protected_episode"]:
                    action, _ = models[arm].select(tuple(row["activation_features"]))
                if action != row["selector_action"]:
                    raise ValueError("same-frame neural selector action mismatch")
                if (
                    action == 0
                    and row["physical_trace_hash"] != arms["parent"]["physical_trace_hash"]
                ):
                    raise ValueError("zero-action parent fallback changed physical trace")
        original_exact = (
            arms["parent"]["physical_trace_hash"] == historical[seed]["physical_trace_hash"]
        )
        if original_exact != group["parent_matches_historical_trace"]:
            raise ValueError("historical parent identity mismatch")
        base = clean_controlled(arms["parent"])
        per = {
            arm: dict(
                action=arms[arm]["selector_action"],
                clean_controlled=clean_controlled(arms[arm]),
                loses_parent=base and not clean_controlled(arms[arm]),
                gains_parent=not base and clean_controlled(arms[arm]),
            )
            for arm in models
        }
        rows.append(
            dict(
                course_seed=seed,
                historical_exact=original_exact,
                parent_clean_controlled=base,
                seeds=per,
            )
        )
    if {r["course_seed"] for r in rows} != set(historical):
        raise ValueError("all 55 source courses exactly once required")
    summary = {
        arm: dict(
            losses=sum(r["seeds"][arm]["loses_parent"] for r in rows),
            gains=sum(r["seeds"][arm]["gains_parent"] for r in rows),
            controlled=sum(r["seeds"][arm]["clean_controlled"] for r in rows),
        )
        for arm in models
    }
    write_once(
        args.output / "complete.json",
        dict(
            status="TEAM_RETENTION_TRACE_SELECTOR_READBACK_NOT_REINTEGRATED",
            source_hash=args.source_hash,
            source_head=args.expected_head,
            rows=rows,
            summary=summary,
            historical_exact_count=sum(r["historical_exact"] for r in rows),
            physical_integrations=0,
            safety_and_receiving_labels_not_recomputed=True,
            fresh_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
