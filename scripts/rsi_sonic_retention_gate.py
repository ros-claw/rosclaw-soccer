"""Learn a conservative local-skill gate and test fresh physical courses.

The development failure evidence may choose the gate, never the fresh courses.
No candidate is activated outside this SIM_ONLY isolated MuJoCo proof.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from rsi_sonic_contextual_aim_actor import PARENT, _action, _read_verified_report, _rollout

from rosclaw_soccer.rsi.sonic_contact_selector import load_selector
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

RESERVED = tuple((x, y) for x in (2.185, 2.198) for y in (0.085, 0.125, 0.165))
SCHEMA = "rosclaw_soccer.rsi.sonic_retention_gate.v1"


def _learn_threshold(dev_root: Path) -> tuple[float, list[dict[str, Any]]]:
    evidence = []
    for y in (0.08, 0.12, 0.16):
        reports = {
            name: _read_verified_report(
                dev_root / f"reserved-{name}-x2190-y{round(y * 1000)}-r0" / "report.json"
            )
            for name in ("parent", "candidate")
        }
        if any(report["partition"] != "FRESH" for report in reports.values()):
            raise ValueError("gate development must come from consumed physical holdout")
        parent_crossing = reports["parent"]["goal_crossing_ball_center_xyz_m"]
        candidate_crossing = reports["candidate"]["goal_crossing_ball_center_xyz_m"]
        if parent_crossing is None or candidate_crossing is None:
            raise ValueError("gate development requires both actors to finish each goal")
        delta = abs(float(candidate_crossing[1])) - abs(float(parent_crossing[1]))
        evidence.append(
            {
                "ball_y_m": y,
                "candidate_minus_parent_error_m": delta,
                "parent_report_hash": reports["parent"]["report_hash"],
                "candidate_report_hash": reports["candidate"]["report_hash"],
            }
        )
    return _choose_threshold(evidence), evidence


def _choose_threshold(evidence: list[dict[str, Any]]) -> float:
    harmful = [
        float(row["ball_y_m"]) for row in evidence if row["candidate_minus_parent_error_m"] > 0.10
    ]
    if not harmful:
        return 0.06
    last_harmful = max(harmful)
    next_safe = min(
        (float(row["ball_y_m"]) for row in evidence if row["ball_y_m"] > last_harmful),
        default=None,
    )
    if next_safe is None:
        raise ValueError("no safe candidate region remains")
    return float((last_harmful + next_safe) / 2.0 + 0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contextual-evidence", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    checkout = Path.cwd().resolve()
    output_root = args.output_root.resolve()
    if (
        output_root.exists()
        or output_root == checkout
        or checkout in output_root.parents
        or not args.contextual_evidence.is_dir()
        or not args.model_root.is_dir()
        or not args.stadium_assets.is_dir()
    ):
        raise ValueError("new external output and qualified inputs required")
    selector = load_selector(args.selector)
    source = Path(__file__)
    dependencies = (
        source.with_name("rsi_sonic_contextual_aim_actor.py"),
        source.with_name("rsi_sonic_ball_contact_probe.py"),
    )
    hashes = {str(path): hash_bytes(path.read_bytes()) for path in (source, *dependencies)}
    selector_hash = hash_bytes(args.selector.read_bytes())
    actor_path = args.contextual_evidence / "actor.json"
    actor_file_hash = hash_bytes(actor_path.read_bytes())
    model_hashes = {
        name: hash_bytes((args.model_root / "low_latency" / name).read_bytes())
        for name in ("model_encoder.onnx", "model_decoder.onnx", "config.yaml")
    }
    actor: dict[str, Any] = json.loads(actor_path.read_text(encoding="utf-8"))
    actor_hash = actor.pop("actor_hash")
    if hash_json(actor) != actor_hash:
        raise ValueError("contextual actor integrity mismatch")
    actor["actor_hash"] = actor_hash
    protocol = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "source_hashes": hashes,
        "selector_hash": selector_hash,
        "actor_hash": actor_hash,
        "actor_file_hash": actor_file_hash,
        "model_hashes": model_hashes,
        "development_evidence_root": str(args.contextual_evidence.resolve()),
        "reserved_courses": RESERVED,
        "gate_rule": "midpoint_after_last_development_regression_gt_0.10m_plus_0.01m_margin",
        "acceptance": (
            "all_6_safe_foot_goals; max_error<=0.40m; mean_gain>=0.10m; "
            "no_course_regression>0.10m; speed_noninferiority>=-0.40mps; strict_replay"
        ),
    }
    output_root.mkdir(parents=True)
    protocol["protocol_hash"] = hash_json(protocol)
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    threshold, dev_evidence = _learn_threshold(args.contextual_evidence)
    gate = {
        "schema": SCHEMA,
        "protocol_hash": protocol["protocol_hash"],
        "actor_hash": actor_hash,
        "candidate_if_ball_y_at_least_m": threshold,
        "development_evidence": dev_evidence,
    }
    gate["gate_hash"] = hash_json(gate)
    (output_root / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    futures = []
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for x, y in RESERVED:
            action = _action(actor["anchors"], y) if y >= threshold else PARENT
            for name, applied in (("parent", PARENT), ("gated", action)):
                for repeat in (0, 1):
                    futures.append(
                        pool.submit(
                            _rollout,
                            actor=name,
                            action=applied,
                            x=x,
                            y=y,
                            repeat=repeat,
                            output_root=output_root,
                            model_root=args.model_root,
                            stadium_assets=args.stadium_assets,
                            selector=selector,
                        )
                    )
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda row: (row["actor"], row["course"], row["repeat"]))

    def course(name: str, x: float, y: float, repeat: int = 0) -> dict[str, Any]:
        return next(
            row
            for row in rows
            if row["actor"] == name and row["course"] == [x, y] and row["repeat"] == repeat
        )

    strict = all(
        course(name, x, y)["trajectory_hash"] == course(name, x, y, 1)["trajectory_hash"]
        for name in ("parent", "gated")
        for x, y in RESERVED
    )
    parent = [course("parent", x, y) for x, y in RESERVED]
    gated = [course("gated", x, y) for x, y in RESERVED]
    valid_goals = all(row["safe"] and row["foot_first"] and row["goal"] for row in gated)
    complete_errors = all(row["goal_error_m"] is not None for row in (*parent, *gated))
    parent_errors = [float(row["goal_error_m"]) for row in parent] if complete_errors else []
    gated_errors = [float(row["goal_error_m"]) for row in gated] if complete_errors else []
    parent_speed = float(np.mean([row["peak_ball_speed_mps"] for row in parent]))
    gated_speed = float(np.mean([row["peak_ball_speed_mps"] for row in gated]))
    source_stable = bool(
        all(hash_bytes(Path(name).read_bytes()) == digest for name, digest in hashes.items())
        and hash_bytes(args.selector.read_bytes()) == selector_hash
        and hash_bytes(actor_path.read_bytes()) == actor_file_hash
        and all(
            hash_bytes((args.model_root / "low_latency" / name).read_bytes()) == digest
            for name, digest in model_hashes.items()
        )
    )
    passed = bool(
        valid_goals
        and complete_errors
        and max(gated_errors) <= 0.40
        and np.mean(parent_errors) - np.mean(gated_errors) >= 0.10
        and max(
            gated_error - parent_error
            for gated_error, parent_error in zip(gated_errors, parent_errors, strict=True)
        )
        <= 0.10
        and gated_speed - parent_speed >= -0.40
        and strict
        and source_stable
    )
    result = {
        "schema": SCHEMA,
        "protocol_hash": protocol["protocol_hash"],
        "gate_hash": gate["gate_hash"],
        "reserved_physical_episodes": len(rows),
        "parent_foot_goals": sum(
            row["safe"] and row["foot_first"] and row["goal"] for row in parent
        ),
        "gated_foot_goals": sum(row["safe"] and row["foot_first"] and row["goal"] for row in gated),
        "parent_mean_error_m": float(np.mean(parent_errors)) if complete_errors else None,
        "gated_mean_error_m": float(np.mean(gated_errors)) if complete_errors else None,
        "gated_max_error_m": max(gated_errors) if complete_errors else None,
        "max_per_course_regression_m": (
            max(
                gated_error - parent_error
                for gated_error, parent_error in zip(gated_errors, parent_errors, strict=True)
            )
            if complete_errors
            else None
        ),
        "parent_speed_mps": parent_speed,
        "gated_speed_mps": gated_speed,
        "strict_replay": strict,
        "source_stable_during_run": source_stable,
        "passed": passed,
        "promotion_status": "FROZEN_LOCAL_RETENTION_GATE" if passed else "REJECTED_DEVELOPMENT",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "no_team_claim": True,
    }
    result["report_hash"] = hash_json(result)
    (output_root / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
