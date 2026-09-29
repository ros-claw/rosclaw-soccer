"""Train a SIM_ONLY six-arm navigation chooser on audited G1 proprioception."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_contextual_neural_retrain import train_candidate
from scripts.rsi_team_contextual_neural_train import tally
from scripts.rsi_team_large_context_chooser_train import feature_table


def _physical_features(
    *,
    root: Path,
    report: dict[str, Any],
    raw: np.ndarray[Any, Any],
    entry_hashes: list[str],
) -> dict[str, np.ndarray[Any, Any]]:
    if (
        raw.ndim != 2
        or raw.shape[1] != 24
        or len(raw) not in (64, 512)
        or len(entry_hashes) != len(raw)
    ):
        raise ValueError("expected audited 64- or 512-scene causal body table")
    batches = len(raw) // 32
    legs: list[np.ndarray[Any, Any]] = []
    full: list[np.ndarray[Any, Any]] = []
    batch_rows: dict[int, list[dict[str, Any]]] = {}
    for batch in range(batches):
        batch_report = json.loads((root / f"b{batch:02d}/parent/report.json").read_text())
        if (
            batch_report.get("report_hash")
            != hash_json(
                {key: value for key, value in batch_report.items() if key != "report_hash"}
            )
            or batch_report["report_hash"]
            != report["arm_report_hashes"].get(f"b{batch:02d}/parent")
            or len(batch_report.get("rows", [])) != 32
        ):
            raise ValueError("unsealed parent physical batch")
        batch_rows[batch] = batch_report["rows"]
    for index in range(len(raw)):
        batch, local = divmod(index, 32)
        episode_path = root / f"b{batch:02d}/parent/t{local:03d}/parent"
        episode = json.loads((episode_path / "report.json").read_text())
        sealed = episode.get("report_hash")
        if (
            sealed
            != hash_json({key: value for key, value in episode.items() if key != "report_hash"})
            or episode.get("trace_hash")
            != hash_bytes((episode_path / "trajectory.npz").read_bytes())
            or episode.get("action_trace_hash")
            != hash_bytes((episode_path / "taskspace_trace.npz").read_bytes())
        ):
            raise ValueError("unsealed parent physical episode")
        action_path = episode_path / "taskspace_trace.npz"
        entry = entry_features(action_path, 30)
        if (
            entry["hash"] != entry_hashes[index]
            or entry["hash"] != batch_rows[batch][local]["entry_hash"]
            or not np.array_equal(entry["values"], raw[index])
        ):
            raise ValueError("parent entry does not match audited physical dataset")
        with np.load(action_path, allow_pickle=False) as trace:
            qpos = np.asarray(trace["pre_step_focal_qpos"], dtype=np.float64)[30, 0]
            qvel = np.asarray(trace["pre_step_focal_qvel"], dtype=np.float64)[30, 0]
        if (
            qpos.shape != (43,)
            or qvel.shape != (41,)
            or not all(np.isfinite(value).all() for value in (qpos, qvel))
        ):
            raise ValueError("invalid measured joint state")
        legs.append(np.concatenate((raw[index], qpos[7:19], qvel[6:18])))
        full.append(np.concatenate((raw[index], qpos[3:7], qvel[3:6], qpos[7:36], qvel[6:35])))
    return {
        "legs_48": np.asarray(legs, dtype=np.float64),
        "full_proprio_89": np.asarray(full, dtype=np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("full-proprio training output exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_full_proprio_chooser_protocol_v46"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("training_scene_count") != 384
        or protocol.get("internal_validation_scene_count") != 128
        or protocol.get("feature_sets") != ["relative_feet_16", "legs_48", "full_proprio_89"]
        or protocol.get("epochs") != [25, 50, 100]
        or protocol.get("safety_thresholds") != [0.65, 0.75]
        or protocol.get("network_width") != 24
        or protocol.get("ensemble_seeds") != [11, 19, 29]
        or protocol.get("internal_gate")
        != {
            "minimum_useful": 28,
            "minimum_safe_contact": 50,
            "maximum_unsafe": 14,
            "maximum_abstained": 32,
        }
    ):
        raise ValueError("invalid frozen full-proprio training protocol")
    audit_path = Path(protocol["audit_report_path"])
    dataset_path = Path(protocol["dataset_path"])
    if (
        hash_bytes(audit_path.read_bytes()) != protocol["audit_report_file_hash"]
        or hash_bytes(dataset_path.read_bytes()) != protocol["dataset_hash"]
    ):
        raise ValueError("physical audit or dataset changed")
    report = json.loads(audit_path.read_text())
    if (
        report.get("report_hash")
        != hash_json({key: value for key, value in report.items() if key != "report_hash"})
        or report.get("report_hash") != protocol["audit_report_hash"]
        or report.get("dataset_hash") != protocol["dataset_hash"]
        or report.get("physical_scene_count") != 512
    ):
        raise ValueError("invalid physical audit commitment")
    with np.load(dataset_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_entry"], dtype=np.float64)
        entry_hashes = data["entry_hashes"].tolist()
        names = data["arm_names"].tolist()
        labels = np.stack(
            [np.asarray(data[key]) for key in ("safe", "safe_contact", "useful_pass")],
            axis=2,
        ).astype(np.float64)
    if raw.shape != (512, 24) or labels.shape != (512, 6, 3) or len(entry_hashes) != 512:
        raise ValueError("invalid audited six-arm physical table")
    features = {
        "relative_feet_16": feature_table(raw)["relative_feet_16"],
        **_physical_features(
            root=Path(protocol["physical_root"]),
            report=report,
            raw=raw,
            entry_hashes=entry_hashes,
        ),
    }
    torch.set_num_threads(1)
    rows: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for name in protocol["feature_sets"]:
        table = features[name]
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
                gate = protocol["internal_gate"]
                eligible = bool(
                    metrics["useful"] >= gate["minimum_useful"]
                    and metrics["safe_contact"] >= gate["minimum_safe_contact"]
                    and metrics["unsafe"] <= gate["maximum_unsafe"]
                    and metrics["abstained"] <= gate["maximum_abstained"]
                )
                rows.append(
                    {
                        "index": len(rows),
                        "feature_set": name,
                        "specification": specification,
                        "internal_validation": metrics,
                        "eligible_for_fresh_exam": eligible,
                        "arm_counts": {
                            arm: int((selected == i).sum()) for i, arm in enumerate(names)
                        },
                    }
                )
                models.append(model)

    def rank(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
        score = row["internal_validation"]
        return (
            -score["useful"],
            score["unsafe"],
            -score["safe_contact"],
            score["abstained"],
            row["index"],
        )

    eligible_rows = sorted((row for row in rows if row["eligible_for_fresh_exam"]), key=rank)
    chosen = (eligible_rows or sorted(rows, key=rank))[0]
    result = {
        "schema": "rsi_team_full_proprio_chooser_train_report_v46",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "dataset_hash": protocol["dataset_hash"],
        "audit_report_hash": protocol["audit_report_hash"],
        "training_scene_count": 384,
        "internal_validation_scene_count": 128,
        "internal_validation_previously_consumed": True,
        "arms": names,
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
    selected_model = {
        "schema": "rsi_team_full_proprio_chooser_model_v46",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": result["protocol_hash"],
        "training_dataset_hash": protocol["dataset_hash"],
        "feature_set": chosen["feature_set"],
        "arm_names": names,
        "selected_candidate_index": chosen["index"],
        **models[chosen["index"]],
    }
    selected_model["model_hash"] = hash_json(selected_model)
    (args.output_dir / "model.json").write_text(
        json.dumps(selected_model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_FULL_PROPRIO="
        + json.dumps(
            {
                "report_hash": result["report_hash"],
                "model_hash": selected_model["model_hash"],
                "chosen": chosen,
                "internal_gate_passed": result["internal_gate_passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
