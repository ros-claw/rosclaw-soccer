"""Jointly train a safe six-arm chooser on two sealed physical curricula."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_full_proprio_chooser_train import _physical_features
from scripts.rsi_team_proprio_tree_chooser_train import _select
from scripts.rsi_team_proprio_tree_transfer_v50 import _parent_labels, outcome_counts


def _curriculum(spec: dict[str, Any]) -> dict[str, Any]:
    audit_path = Path(spec["audit_report_path"])
    dataset_path = Path(spec["dataset_path"])
    if (
        hash_bytes(audit_path.read_bytes()) != spec["audit_report_file_hash"]
        or hash_bytes(dataset_path.read_bytes()) != spec["dataset_hash"]
    ):
        raise ValueError("joint curriculum evidence changed")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("report_hash") != spec["audit_report_hash"]
        or audit["report_hash"]
        != hash_json({key: value for key, value in audit.items() if key != "report_hash"})
        or audit.get("dataset_hash") != spec["dataset_hash"]
        or audit.get("physical_scene_count") != 512
    ):
        raise ValueError("unsealed joint curriculum audit")
    with np.load(dataset_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_entry"], dtype=np.float64)
        entry_hashes = data["entry_hashes"].tolist()
        scenarios = data["scenario_hashes"].tolist()
        names = data["arm_names"].tolist()
        labels = np.stack(
            [np.asarray(data[key]) for key in ("safe", "safe_contact", "useful_pass")], axis=2
        ).astype(np.bool_)
    if raw.shape != (512, 24) or labels.shape != (512, 6, 3) or len(scenarios) != 512:
        raise ValueError("invalid six-arm physical data")
    root = Path(spec["physical_root"])
    feature = _physical_features(root=root, report=audit, raw=raw, entry_hashes=entry_hashes)[
        "full_proprio_89"
    ]
    return {
        "audit": audit,
        "names": names,
        "scenarios": scenarios,
        "feature": feature,
        "labels": labels,
        "parent": _parent_labels(root, audit, scenarios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("joint training output already exists")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("schema") != "rsi_team_proprio_joint_train_protocol_v51"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or len(protocol.get("curricula", ())) != 2
        or protocol.get("train_per_curriculum") != 384
        or protocol.get("validation_per_curriculum") != 128
        or protocol.get("max_depths") != [6, 10]
        or protocol.get("min_samples_leaf") != [3, 8]
        or protocol.get("safety_thresholds") != [0.65, 0.75, 0.85]
        or protocol.get("estimators_per_forest") != 120
        or protocol.get("ensemble_seeds") != [11, 19]
        or protocol.get("internal_gate")
        != {
            "minimum_old_useful": 30,
            "maximum_old_unsafe": 12,
            "minimum_new_useful": 23,
            "maximum_new_unsafe": 18,
            "minimum_combined_useful": 58,
        }
    ):
        raise ValueError("invalid frozen joint training protocol")
    old, new = (_curriculum(item) for item in protocol["curricula"])
    if (
        old["names"] != new["names"]
        or old["audit"]["body_hash"] != new["audit"]["body_hash"]
        or set(old["scenarios"]) & set(new["scenarios"])
    ):
        raise ValueError("mixed arm/body identity or overlapping physical scenes")
    stable_sources = (
        "scripts/rsi_team_diverse_curriculum_collect.py",
        "src/rosclaw_soccer/rsi/team_intercept_navigation.py",
        "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        "src/rosclaw_soccer/rsi/team_adaptive_intercept_navigation.py",
    )
    for path in stable_sources:
        if (
            old["audit"]["physical_source_hashes"][path]
            != new["audit"]["physical_source_hashes"][path]
        ):
            raise ValueError(f"navigation semantics changed: {path}")
    train_feature = np.concatenate((old["feature"][:384], new["feature"][:384]))
    train_label = np.concatenate((old["labels"][:384], new["labels"][:384]))
    validation_feature = np.concatenate((old["feature"][384:], new["feature"][384:]))
    rows: list[dict[str, Any]] = []
    for depth in protocol["max_depths"]:
        for leaf in protocol["min_samples_leaf"]:
            for threshold in protocol["safety_thresholds"]:
                selected = _select(
                    train_feature=train_feature,
                    train_label=train_label,
                    validation_feature=validation_feature,
                    depth=depth,
                    leaf=leaf,
                    threshold=threshold,
                    estimators=120,
                    seeds=[11, 19],
                )
                old_metrics = outcome_counts(
                    selected[:128], old["labels"][384:], old["parent"][384:]
                )
                new_metrics = outcome_counts(
                    selected[128:], new["labels"][384:], new["parent"][384:]
                )
                gate = protocol["internal_gate"]
                eligible = bool(
                    old_metrics["useful"] >= gate["minimum_old_useful"]
                    and old_metrics["unsafe"] <= gate["maximum_old_unsafe"]
                    and new_metrics["useful"] >= gate["minimum_new_useful"]
                    and new_metrics["unsafe"] <= gate["maximum_new_unsafe"]
                    and old_metrics["useful"] + new_metrics["useful"]
                    >= gate["minimum_combined_useful"]
                )
                rows.append(
                    {
                        "index": len(rows),
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "safety_threshold": threshold,
                        "old_validation": old_metrics,
                        "new_validation": new_metrics,
                        "eligible_for_fresh_exam": eligible,
                    }
                )
                print(
                    f"depth={depth} leaf={leaf} threshold={threshold} "
                    f"old={old_metrics['useful']}/{old_metrics['unsafe']} "
                    f"new={new_metrics['useful']}/{new_metrics['unsafe']}",
                    flush=True,
                )

    def rank(row: dict[str, Any]) -> tuple[int, int, int]:
        old_score, new_score = row["old_validation"], row["new_validation"]
        return (
            -(old_score["useful"] + new_score["useful"]),
            old_score["unsafe"] + new_score["unsafe"],
            row["index"],
        )

    eligible_rows = sorted((row for row in rows if row["eligible_for_fresh_exam"]), key=rank)
    chosen = (eligible_rows or sorted(rows, key=rank))[0]
    result = {
        "schema": "rsi_team_proprio_joint_train_report_v51",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "dataset_hashes": [item["audit"]["dataset_hash"] for item in (old, new)],
        "body_hash": old["audit"]["body_hash"],
        "sklearn_version": sklearn_version,
        "training_scenes": 768,
        "consumed_internal_validation_scenes": 256,
        "fixed_old_validation": outcome_counts(
            np.full(128, old["names"].index("phase_front04_lat12"), dtype=np.int64),
            old["labels"][384:],
            old["parent"][384:],
        ),
        "fixed_new_validation": outcome_counts(
            np.full(128, new["names"].index("phase_front04_lat12"), dtype=np.int64),
            new["labels"][384:],
            new["parent"][384:],
        ),
        "rows": rows,
        "chosen_index": chosen["index"],
        "internal_gate_passed": bool(eligible_rows),
        "fresh_exam_authorized": bool(eligible_rows),
    }
    result["report_hash"] = hash_json(result)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_V51_JOINT="
        + json.dumps(
            {
                "report_hash": result["report_hash"],
                "chosen": chosen,
                "internal_gate_passed": result["internal_gate_passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
