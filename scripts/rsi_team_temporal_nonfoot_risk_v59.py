"""Diagnose causal short-horizon nonfoot-ball-contact risk from sealed SIM episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesClassifier

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _causal_feature(trace: dict[str, np.ndarray[Any, Any]], frame: int) -> np.ndarray[Any, Any]:
    """Use the current pre-step observation and a past-only ball finite difference."""
    ball = trace["pre_step_ball_position_local_m"][frame, 0]
    previous = trace["pre_step_ball_position_local_m"][frame - 1, 0]
    feet = trace["pre_step_foot_link_position_w"][frame, 0]
    foot_velocity = trace["pre_step_foot_linear_velocity_w"][frame, 0]
    qpos = trace["pre_step_focal_qpos"][frame, 0]
    qvel = trace["pre_step_focal_qvel"][frame, 0]
    side = trace["taskspace_selected_side"][frame, 0]
    values = np.concatenate(
        (
            ball - qpos[:3],
            (ball[None, :] - feet).reshape(-1),
            (ball - previous) / 0.02,
            foot_velocity.reshape(-1),
            qvel[:6],
            qpos[3:7],
            np.asarray([side], dtype=np.float64),
        )
    )
    if values.shape != (29,) or not np.isfinite(values).all():
        raise ValueError("invalid causal 29-feature body observation")
    return np.asarray(values, dtype=np.float64)


def _sealed_episode(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    report = json.loads((path / "report.json").read_text())
    if (
        report.get("report_hash")
        != hash_json({k: v for k, v in report.items() if k != "report_hash"})
        or report.get("trace_hash") != hash_bytes((path / "trajectory.npz").read_bytes())
        or report.get("action_trace_hash")
        != hash_bytes((path / "taskspace_trace.npz").read_bytes())
    ):
        raise ValueError(f"unsealed physical episode: {path}")
    with np.load(path / "taskspace_trace.npz", allow_pickle=False) as data:
        trace = {key: np.asarray(data[key]) for key in data.files}
    return report, trace


def _future_event(first_event: int, frame: int, horizon: int) -> bool:
    """250 is the no-event sentinel, never a positive contact at the episode end."""
    if not 0 <= frame < 250 or not 0 <= first_event <= 250 or horizon < 1:
        raise ValueError("invalid bounded event horizon")
    return first_event < 250 and frame < first_event <= frame + horizon


def _dataset(
    protocol: dict[str, Any], prefix: str, batches: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, int]]:
    audit_path = Path(protocol[f"{prefix}_audit_report_path"])
    if hash_bytes(audit_path.read_bytes()) != protocol[f"{prefix}_audit_report_file_hash"]:
        raise ValueError(f"{prefix} audit file changed")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("report_hash")
        != hash_json({k: v for k, v in audit.items() if k != "report_hash"})
        or audit.get("report_hash") != protocol[f"{prefix}_audit_report_hash"]
        or audit.get("scene_count") != batches * 32
    ):
        raise ValueError(f"{prefix} audit commitment invalid")
    root = Path(protocol[f"{prefix}_physical_root"])
    arm = protocol["arm"]
    features: list[np.ndarray[Any, Any]] = []
    targets: list[bool] = []
    groups: list[int] = []
    incomplete = 0
    event_episodes = 0
    for batch in range(batches):
        arm_report = json.loads((root / f"b{batch:02d}" / arm / "report.json").read_text())
        if (
            arm_report.get("report_hash")
            != hash_json({k: v for k, v in arm_report.items() if k != "report_hash"})
            or arm_report["report_hash"] != audit["arm_report_hashes"].get(f"b{batch:02d}/{arm}")
            or len(arm_report.get("rows", [])) != 32
        ):
            raise ValueError(f"{prefix} unsealed arm batch")
        for local, row in enumerate(arm_report["rows"]):
            if row["status"] != "COMPLETE":
                incomplete += 1
                continue
            path = root / f"b{batch:02d}" / arm / f"t{local:03d}" / "candidate"
            episode, trace = _sealed_episode(path)
            if (
                episode["scenario_hash"] != row["scenario_hash"]
                or episode["report_hash"] != row["report_hash"]
            ):
                raise ValueError("arm row/episode mismatch")
            events = episode["focal_nonfoot_contact_frames"]
            first_event = int(min(events)) if events else 250
            event_episodes += int(bool(events))
            if trace["pre_step_ball_position_local_m"].shape != (250, 1, 3):
                raise ValueError("wrong full physical action trace")
            for frame in range(
                protocol["entry_frame"], min(first_event, 250), protocol["sample_stride_frames"]
            ):
                features.append(_causal_feature(trace, frame))
                targets.append(
                    _future_event(first_event, frame, protocol["prediction_horizon_frames"])
                )
                groups.append(batch * 32 + local)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.bool_),
        np.asarray(groups, dtype=np.int32),
        {
            "complete": batches * 32 - incomplete,
            "incomplete": incomplete,
            "nonfoot_event_episodes": event_episodes,
        },
    )


def _diagnose(
    probability: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    group: np.ndarray[Any, Any],
    threshold: float,
) -> dict[str, int]:
    alert = probability >= threshold
    tp = int((alert & target).sum())
    fp = int((alert & ~target).sum())
    event_groups = set(group[target].tolist())
    clean_groups = set(group.tolist()) - event_groups
    alerted_groups = set(group[alert].tolist())
    return {
        "positive_frames": int(target.sum()),
        "true_alert_frames": tp,
        "false_alert_frames": fp,
        "event_episodes_with_alert": len(event_groups & alerted_groups),
        "event_episodes": len(event_groups),
        "clean_episodes_with_false_alert": len(clean_groups & alerted_groups),
        "clean_episodes": len(clean_groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("temporal risk output exists")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("schema") != "rsi_team_temporal_nonfoot_risk_protocol_v59"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("arm") != "gate22_cap10"
        or protocol.get("entry_frame") != 30
        or protocol.get("sample_stride_frames") != 2
        or protocol.get("prediction_horizon_frames") != 10
        or protocol.get("train_batches") != 12
        or protocol.get("validation_batches") != 4
        or protocol.get("stress_batches") != 2
        or protocol.get("estimators") != 160
        or protocol.get("max_depth") != 12
        or protocol.get("min_samples_leaf") != 3
        or protocol.get("negative_to_positive_train_ratio") != 20
        or protocol.get("random_seed") != 20261259
        or protocol.get("diagnostic_thresholds") != [0.1, 0.2, 0.3]
    ):
        raise ValueError("invalid frozen v59 diagnostic protocol")
    x, y, group, training_counts = _dataset(protocol, "training", 16)
    stress_x, stress_y, stress_group, stress_counts = _dataset(protocol, "stress", 2)
    train_mask = group < 384
    positive = np.flatnonzero(train_mask & y)
    negative = np.flatnonzero(train_mask & ~y)
    if len(positive) < 100:
        raise ValueError("insufficient independently measured event precursors")
    generator = np.random.default_rng(protocol["random_seed"])
    selected_negative = generator.choice(
        negative,
        size=min(len(negative), len(positive) * protocol["negative_to_positive_train_ratio"]),
        replace=False,
    )
    selected = np.concatenate((positive, selected_negative))
    model = ExtraTreesClassifier(
        n_estimators=protocol["estimators"],
        max_depth=protocol["max_depth"],
        min_samples_leaf=protocol["min_samples_leaf"],
        max_features=0.8,
        class_weight="balanced",
        random_state=protocol["random_seed"],
        n_jobs=1,
    )
    model.fit(x[selected], y[selected].astype(np.int8))
    class_one = np.flatnonzero(model.classes_ == 1)
    if len(class_one) != 1:
        raise ValueError("temporal classifier lost positive class")
    validation_probability = model.predict_proba(x[~train_mask])[:, class_one[0]]
    stress_probability = model.predict_proba(stress_x)[:, class_one[0]]
    rows = [
        {
            "threshold": threshold,
            "consumed_internal_validation": _diagnose(
                validation_probability, y[~train_mask], group[~train_mask], threshold
            ),
            "consumed_external_stress": _diagnose(
                stress_probability, stress_y, stress_group, threshold
            ),
        }
        for threshold in protocol["diagnostic_thresholds"]
    ]
    report: dict[str, Any] = {
        "schema": "rsi_team_temporal_nonfoot_risk_report_v59",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "controller_activation_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "sklearn_version": sklearn_version,
        "training_counts": training_counts,
        "stress_counts": stress_counts,
        "training_frames": int(train_mask.sum()),
        "training_positive_frames": len(positive),
        "training_sampled_negative_frames": len(selected_negative),
        "consumed_internal_validation_frames": int((~train_mask).sum()),
        "consumed_external_stress_frames": len(stress_y),
        "rows": rows,
    }
    report["report_hash"] = hash_json(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("RSI_TEAM_TEMPORAL_RISK=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
