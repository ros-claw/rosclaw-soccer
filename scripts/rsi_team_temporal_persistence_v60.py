"""Test sustained short-horizon contact alerts without activating a controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_temporal_nonfoot_risk_v59 import _dataset, _diagnose


def _sustained(
    probability: np.ndarray[Any, Any],
    group: np.ndarray[Any, Any],
    threshold: float,
    consecutive: int,
) -> np.ndarray[Any, Any]:
    if probability.shape != group.shape or consecutive < 1 or not np.isfinite(probability).all():
        raise ValueError("invalid episode-local alert stream")
    raw = probability >= threshold
    alert = np.zeros(len(raw), dtype=np.bool_)
    run = 0
    previous = -1
    for index, episode in enumerate(group):
        run = run + 1 if episode == previous and raw[index] else (1 if raw[index] else 0)
        alert[index] = run >= consecutive
        previous = int(episode)
    return alert


def _episode_metrics(
    alert: np.ndarray[Any, Any], target: np.ndarray[Any, Any], group: np.ndarray[Any, Any]
) -> dict[str, int]:
    # Reuse the v59 tally without interpreting persistence as a probability threshold.
    return _diagnose(alert.astype(np.float64), target, group, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("temporal persistence output exists")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("schema") != "rsi_team_temporal_persistence_protocol_v60"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("thresholds") != [0.3, 0.5, 0.7]
        or protocol.get("consecutive_samples") != [2, 3, 4]
        or protocol.get("internal_gate")
        != {"minimum_event_episode_recall": 10, "maximum_clean_episode_false_alerts": 12}
        or protocol.get("stress_gate")
        != {"minimum_event_episode_recall": 7, "maximum_clean_episode_false_alerts": 6}
    ):
        raise ValueError("invalid frozen v60 persistence protocol")
    root = Path(__file__).parents[1]
    prior_path = root / protocol["prior_protocol_path"]
    prior_report_path = Path(protocol["prior_report_path"])
    if (
        hash_bytes(prior_path.read_bytes()) != protocol["prior_protocol_hash"]
        or hash_bytes(prior_report_path.read_bytes()) != protocol["prior_report_file_hash"]
    ):
        raise ValueError("prior temporal evidence changed")
    prior_report = json.loads(prior_report_path.read_text())
    if (
        prior_report.get("report_hash") != protocol["prior_report_hash"]
        or prior_report["report_hash"]
        != hash_json({k: v for k, v in prior_report.items() if k != "report_hash"})
        or prior_report.get("controller_activation_authorized") is not False
    ):
        raise ValueError("prior temporal diagnostic commitment invalid")
    v59 = json.loads(prior_path.read_text())
    x, y, group, _ = _dataset(v59, "training", 16)
    stress_x, stress_y, stress_group, _ = _dataset(v59, "stress", 2)
    train = group < 384
    positive = np.flatnonzero(train & y)
    negative = np.flatnonzero(train & ~y)
    generator = np.random.default_rng(v59["random_seed"])
    sampled_negative = generator.choice(
        negative,
        size=min(len(negative), len(positive) * v59["negative_to_positive_train_ratio"]),
        replace=False,
    )
    selected = np.concatenate((positive, sampled_negative))
    if (
        len(positive) != prior_report["training_positive_frames"]
        or len(sampled_negative) != prior_report["training_sampled_negative_frames"]
    ):
        raise ValueError("v59 training labels changed")
    model = ExtraTreesClassifier(
        n_estimators=v59["estimators"],
        max_depth=v59["max_depth"],
        min_samples_leaf=v59["min_samples_leaf"],
        max_features=0.8,
        class_weight="balanced",
        random_state=v59["random_seed"],
        n_jobs=1,
    )
    model.fit(x[selected], y[selected].astype(np.int8))
    positive_class = np.flatnonzero(model.classes_ == 1)
    if len(positive_class) != 1:
        raise ValueError("positive event class missing")
    val_probability = model.predict_proba(x[~train])[:, positive_class[0]]
    stress_probability = model.predict_proba(stress_x)[:, positive_class[0]]
    rows: list[dict[str, Any]] = []
    for threshold in protocol["thresholds"]:
        for consecutive in protocol["consecutive_samples"]:
            internal = _episode_metrics(
                _sustained(val_probability, group[~train], threshold, consecutive),
                y[~train],
                group[~train],
            )
            external = _episode_metrics(
                _sustained(stress_probability, stress_group, threshold, consecutive),
                stress_y,
                stress_group,
            )
            internal_gate = protocol["internal_gate"]
            stress_gate = protocol["stress_gate"]
            internal_pass = bool(
                internal["event_episodes_with_alert"]
                >= internal_gate["minimum_event_episode_recall"]
                and internal["clean_episodes_with_false_alert"]
                <= internal_gate["maximum_clean_episode_false_alerts"]
            )
            stress_pass = bool(
                external["event_episodes_with_alert"] >= stress_gate["minimum_event_episode_recall"]
                and external["clean_episodes_with_false_alert"]
                <= stress_gate["maximum_clean_episode_false_alerts"]
            )
            rows.append(
                {
                    "threshold": threshold,
                    "consecutive_samples": consecutive,
                    "internal": internal,
                    "stress": external,
                    "internal_gate_passed": internal_pass,
                    "stress_gate_passed": stress_pass,
                }
            )
    report: dict[str, Any] = {
        "schema": "rsi_team_temporal_persistence_report_v60",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "controller_activation_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "prior_report_hash": prior_report["report_hash"],
        "rows": rows,
        "any_joint_gate_passed": any(
            row["internal_gate_passed"] and row["stress_gate_passed"] for row in rows
        ),
    }
    report["report_hash"] = hash_json(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("RSI_TEAM_TEMPORAL_PERSISTENCE=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
