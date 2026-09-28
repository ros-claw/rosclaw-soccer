"""Train causal navigation choices from 512 full-information physical scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.team_contextual_nav_policy import measured_entry_features
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_contextual_neural_retrain import train_candidate
from scripts.rsi_team_contextual_neural_train import tally


def feature_table(raw: np.ndarray[Any, Any]) -> dict[str, np.ndarray[Any, Any]]:
    if raw.shape != (512, 24) or not np.all(np.isfinite(raw)):
        raise ValueError("finite 512-context measured entry table required")
    relative = []
    for row in raw:
        feet = (
            tuple(float(v) for v in row[12:15]),
            tuple(float(v) for v in row[15:18]),
        )
        relative.append(
            measured_entry_features(
                ball=tuple(float(v) for v in row[6:9]),  # type: ignore[arg-type]
                ball_vx=float(row[9]),
                feet=feet,
            )
        )
    rel = np.asarray(relative, dtype=np.float64)
    return {
        "relative_body_8": np.concatenate((rel, raw[:, [0, 1, 3, 4]]), axis=1),
        "measured_feet_12": raw[:, 12:24],
        "relative_feet_16": np.concatenate((rel, raw[:, 12:24]), axis=1),
        "ball_feet_18": raw[:, 6:24],
        "full_entry_24": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("large-context model output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_large_context_chooser_protocol_v39"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("training_indices") != "batches_00_to_11_384_scenes"
        or protocol.get("internal_validation_indices") != "batches_12_to_15_128_scenes"
        or protocol.get("feature_sets")
        != [
            "relative_body_8",
            "measured_feet_12",
            "relative_feet_16",
            "ball_feet_18",
            "full_entry_24",
        ]
        or protocol.get("epochs") != [25, 50, 100]
        or protocol.get("safety_thresholds") != [0.65, 0.8]
        or protocol.get("network_width") != 24
        or protocol.get("ensemble_seeds") != [11, 19, 29]
        or protocol.get("internal_gate")
        != {"minimum_useful": 23, "maximum_unsafe": 15, "maximum_abstained": 32}
    ):
        raise ValueError("invalid frozen large-context training protocol")
    dataset_path = Path(protocol["dataset_path"])
    if hash_bytes(dataset_path.read_bytes()) != protocol["dataset_hash"]:
        raise ValueError("sealed 512-context physical dataset mismatch")
    with np.load(dataset_path, allow_pickle=False) as data:
        tables = feature_table(np.asarray(data["raw_entry"], dtype=np.float64))
        labels = np.stack(
            [np.asarray(data[key]) for key in ("safe", "safe_contact", "useful_pass")],
            axis=2,
        ).astype(np.float64)
        arms = data["arm_names"].tolist()
        parent_safe = np.asarray(data["parent_safe"])
    if labels.shape != (512, 6, 3) or len(arms) != 6:
        raise ValueError("invalid six-arm physical outcome table")
    torch.set_num_threads(1)
    rows: list[dict[str, Any]] = []
    models = []
    for feature_name in protocol["feature_sets"]:
        table = tables[feature_name]
        for epochs in protocol["epochs"]:
            for threshold in protocol["safety_thresholds"]:
                specification = {
                    "feature_count": int(table.shape[1]),
                    "epochs": epochs,
                    "safety_threshold": threshold,
                }
                model, selected = train_candidate(
                    train_feature=table[:384],
                    train_label=labels[:384],
                    validation_feature=table[384:],
                    candidate=specification,
                    seeds=protocol["ensemble_seeds"],
                    width=protocol["network_width"],
                )
                metrics = tally(selected, labels[384:])
                eligible = bool(
                    metrics["useful"] >= protocol["internal_gate"]["minimum_useful"]
                    and metrics["unsafe"] <= protocol["internal_gate"]["maximum_unsafe"]
                    and metrics["abstained"] <= protocol["internal_gate"]["maximum_abstained"]
                )
                rows.append(
                    {
                        "index": len(rows),
                        "feature_set": feature_name,
                        "specification": specification,
                        "internal_validation": metrics,
                        "eligible_for_fresh_exam": eligible,
                        "arm_counts": {
                            name: int((selected == i).sum()) for i, name in enumerate(arms)
                        },
                    }
                )
                models.append(model)

    def rank(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
        return (
            -row["internal_validation"]["useful"],
            row["internal_validation"]["unsafe"],
            -row["internal_validation"]["safe_contact"],
            row["internal_validation"]["abstained"],
            row["index"],
        )

    eligible_rows = sorted((row for row in rows if row["eligible_for_fresh_exam"]), key=rank)
    chosen = (eligible_rows or sorted(rows, key=rank))[0]
    course_path = Path(__file__).parents[1] / "docs/rsi/protocols/team-diverse-curriculum-v38.json"
    course = json.loads(course_path.read_text(encoding="utf-8"))
    arm_parameters = {arm["name"]: arm for arm in course["arms"] if arm["name"] in arms}
    adaptive = json.loads(Path(course["adaptive_model_path"]).read_text(encoding="utf-8"))
    if (
        set(arm_parameters) != set(arms)
        or adaptive["model_hash"] != course["adaptive_model_hash"]
        or hash_json({key: value for key, value in adaptive.items() if key != "model_hash"})
        != course["adaptive_model_hash"]
    ):
        raise ValueError("unsealed bounded navigation arm family")
    selected_model = {
        "schema": "rsi_team_large_context_chooser_model_v39",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "training_dataset_hash": protocol["dataset_hash"],
        "feature_set": chosen["feature_set"],
        "arm_names": arms,
        "arm_parameters": arm_parameters,
        "adaptive_model_hash": course["adaptive_model_hash"],
        "adaptive_model_parameters": adaptive["parameters"],
        "selected_candidate_index": chosen["index"],
        **models[chosen["index"]],
    }
    selected_model["model_hash"] = hash_json(selected_model)
    report = {
        "schema": "rsi_team_large_context_chooser_train_report_v39",
        "activation_ceiling": "SIM_ONLY",
        "dataset_hash": protocol["dataset_hash"],
        "training_scene_count": 384,
        "internal_validation_scene_count": 128,
        "parent_internal_validation_unsafe": int((~parent_safe[384:]).sum()),
        "fixed_arm_internal_validation": {
            name: tally(np.full(128, index, dtype=np.int64), labels[384:])
            for index, name in enumerate(arms)
        },
        "offline_oracle_internal_validation_diagnostic_only": {
            "safe_contact": int(labels[384:, :, 1].any(axis=1).sum()),
            "useful": int(labels[384:, :, 2].any(axis=1).sum()),
        },
        "candidate_rows": rows,
        "selected_candidate_index": chosen["index"],
        "selected_model_hash": selected_model["model_hash"],
        "fresh_exam_authorized": bool(eligible_rows),
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(selected_model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_LARGE_CHOOSER="
        + json.dumps(
            {
                "report_hash": report["report_hash"],
                "model_hash": selected_model["model_hash"],
                "chosen": chosen,
                "fresh_exam_authorized": bool(eligible_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
