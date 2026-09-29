"""Audit frozen v47 chooser transfer on the independently collected v49 corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.team_proprio_tree_chooser import TeamProprioTreeChooser
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_full_proprio_chooser_train import _physical_features


def outcome_counts(
    choices: np.ndarray[Any, Any],
    candidate_labels: np.ndarray[Any, Any],
    parent_labels: np.ndarray[Any, Any],
) -> dict[str, int]:
    """An abstention executes Parent; it is never credited as risk-free."""
    if (
        choices.ndim != 1
        or candidate_labels.shape != (len(choices), 6, 3)
        or parent_labels.shape != (len(choices), 3)
        or np.any((choices < -1) | (choices >= 6))
    ):
        raise ValueError("invalid counterfactual choice or outcome table")
    picked = parent_labels.copy()
    active = choices >= 0
    picked[active] = candidate_labels[np.arange(len(choices))[active], choices[active]]
    return {
        "scenes": int(len(choices)),
        "abstained_to_parent": int((~active).sum()),
        "safe": int(picked[:, 0].sum()),
        "safe_contact": int(picked[:, 1].sum()),
        "useful": int(picked[:, 2].sum()),
        "unsafe": int((~picked[:, 0].astype(bool)).sum()),
    }


def _parent_labels(root: Path, audit: dict[str, Any], scenarios: list[str]) -> np.ndarray[Any, Any]:
    labels: list[list[bool]] = []
    for batch in range(16):
        report = json.loads((root / f"b{batch:02d}/parent/report.json").read_text())
        if (
            report.get("report_hash") != audit["arm_report_hashes"][f"b{batch:02d}/parent"]
            or report["report_hash"]
            != hash_json({key: value for key, value in report.items() if key != "report_hash"})
            or len(report.get("rows", ())) != 32
        ):
            raise ValueError("unsealed parent outcome batch")
        for local, row in enumerate(report["rows"]):
            if row["status"] != "COMPLETE" or row["scenario_hash"] != scenarios[batch * 32 + local]:
                raise ValueError("missing or mismatched parent outcome")
            labels.append(
                [
                    bool(row["safe"]),
                    bool(row["safe"] and row["foot_contact_frames"]),
                    bool(row["useful_pass"]),
                ]
            )
    return np.asarray(labels, dtype=np.bool_)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("transfer evidence output already exists")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("schema") != "rsi_team_proprio_tree_transfer_protocol_v50"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("scene_count") != 512
        or protocol.get("frozen_model_hash")
        != "sha256:0d6a2c99834004ea64a8429aee5b90841ccc7acb243d6f49d9cef985e9140e36"
        or protocol.get("audit_report_hash")
        != "sha256:499f5f8082cdfdf14e68c0209a75817fb59c6826ea21882fcafd9b38e8c75a99"
    ):
        raise ValueError("invalid frozen transfer protocol")
    audit_path = Path(protocol["audit_report_path"])
    dataset_path = Path(protocol["dataset_path"])
    model_path = Path(protocol["frozen_model_path"])
    if (
        hash_bytes(audit_path.read_bytes()) != protocol["audit_report_file_hash"]
        or hash_bytes(dataset_path.read_bytes()) != protocol["dataset_hash"]
    ):
        raise ValueError("audited physical evidence changed")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("report_hash") != protocol["audit_report_hash"]
        or audit["report_hash"]
        != hash_json({key: value for key, value in audit.items() if key != "report_hash"})
        or audit.get("dataset_hash") != protocol["dataset_hash"]
        or audit.get("physical_scene_count") != 512
    ):
        raise ValueError("unsealed v49 physical audit")
    model_manifest = json.loads(model_path.read_text())
    if (
        model_manifest.get("model_hash") != protocol["frozen_model_hash"]
        or model_manifest.get("body_hash") != audit["body_hash"]
    ):
        raise ValueError("frozen model or G1 body mismatch")
    with np.load(dataset_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_entry"], dtype=np.float64)
        hashes = data["entry_hashes"].tolist()
        scenarios = data["scenario_hashes"].tolist()
        names = data["arm_names"].tolist()
        labels = np.stack(
            [np.asarray(data[key]) for key in ("safe", "safe_contact", "useful_pass")], axis=2
        ).astype(np.bool_)
    if (
        raw.shape != (512, 24)
        or labels.shape != (512, 6, 3)
        or names != model_manifest["arm_names"]
    ):
        raise ValueError("invalid paired six-arm physical outcomes")
    root = Path(protocol["physical_root"])
    feature = _physical_features(root=root, report=audit, raw=raw, entry_hashes=hashes)[
        "full_proprio_89"
    ]
    parent = _parent_labels(root, audit, scenarios)
    model = TeamProprioTreeChooser(
        agent_id="red.playmaker",
        foundation_hash="offline-frozen-transfer",
        foundation_config_hash="offline-frozen-transfer",
        model_path=model_path,
    )
    selected = np.asarray(
        [
            -1 if (name := model.choose_feature(row)) is None else names.index(name)
            for row in feature
        ],
        dtype=np.int64,
    )
    fixed = np.full(512, names.index("phase_front04_lat12"), dtype=np.int64)
    result = {
        "schema": "rsi_team_proprio_tree_transfer_report_v50",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "audit_report_hash": audit["report_hash"],
        "dataset_hash": audit["dataset_hash"],
        "frozen_model_hash": model.model_hash,
        "body_hash": audit["body_hash"],
        "scene_count": 512,
        "frozen_before_v49_collection": True,
        "parent": outcome_counts(np.full(512, -1, dtype=np.int64), labels, parent),
        "fixed": outcome_counts(fixed, labels, parent),
        "frozen_v47": outcome_counts(selected, labels, parent),
        "frozen_v47_training_half": outcome_counts(selected[:384], labels[:384], parent[:384]),
        "frozen_v47_internal_half": outcome_counts(selected[384:], labels[384:], parent[384:]),
        "chosen_arm_counts": {
            name: int((selected == index).sum()) for index, name in enumerate(names)
        },
        "oracle_useful_diagnostic_only": int(labels[:, :, 2].any(axis=1).sum()),
        "fresh_online_exam": False,
    }
    result["report_hash"] = hash_json(result)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_V50_TRANSFER=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
