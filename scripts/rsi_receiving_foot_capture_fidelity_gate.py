"""Seal paired full-world fidelity for the SIM_ONLY foot-capture training proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _sealed(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    commitment = value.pop("report_hash")
    if (
        commitment != hash_json(value)
        or value.get("schema") != schema
        or value.get("activation_ceiling") != "SIM_ONLY"
        or value.get("promotion_authorized") is not False
    ):
        raise ValueError(f"sealed unpromoted SIM_ONLY {schema} required")
    return value, str(commitment)


def assess(
    *,
    baseline_proxy: Path,
    baseline_full: Path,
    anchor_proxy: Path,
    anchor_full: Path,
    challenger_proxy: Path,
    challenger_full: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.parents[1]):
        raise ValueError("new external SIM_ONLY fidelity gate directory required")
    base_proxy, base_proxy_hash = _sealed(
        baseline_proxy / "report.json", "rosclaw_soccer.rsi.receiving_single_live_sonic_proxy.v1"
    )
    base_full, base_full_hash = _sealed(
        baseline_full / "report.json", "rosclaw_soccer.rsi.receiving_live_short_fidelity.v1"
    )
    pairs = []
    for name, proxy_path, full_path in (
        ("anchor", anchor_proxy, anchor_full),
        ("challenger", challenger_proxy, challenger_full),
    ):
        proxy, proxy_hash = _sealed(
            proxy_path / "report.json",
            "rosclaw_soccer.rsi.receiving_single_live_precontact_foot.v1",
        )
        full, full_hash = _sealed(
            full_path / "report.json",
            "rosclaw_soccer.rsi.receiving_full_prepost_capture_probe.v1",
        )
        if (
            proxy["trajectory_hash"] != hash_bytes((proxy_path / "single-live.npz").read_bytes())
            or full["trace_hash"] != hash_bytes((full_path / "full-prepost.npz").read_bytes())
            or full["proxy_candidate_report_hash"] != proxy_hash
            or full["proxy_frame86_ball_speed_mps"] != proxy["proxy_frame86_ball_speed_mps"]
            or proxy["student_model_hash"] != base_proxy["student_model_hash"]
            or full["student_model_hash"] != base_proxy["student_model_hash"]
            or proxy["source_hashes"] != base_proxy["source_hashes"]
        ):
            raise ValueError("matched sealed proxy/full-world pair required")
        proxy_speed = float(proxy["tail_maximum_ball_speed_mps"])
        full_explanation = full["measurement"]["authoritative_explanation"]
        full_speed = float(full_explanation["tail_maximum_ball_speed_mps"])
        proxy_foot = float(proxy["tail_maximum_foot_distance_m"])
        full_foot = float(full_explanation["tail_maximum_foot_distance_m"])
        pairs.append(
            {
                "name": name,
                "proxy_report_hash": proxy_hash,
                "full_report_hash": full_hash,
                "proxy_speed_mps": proxy_speed,
                "full_speed_mps": full_speed,
                "speed_error_mps": abs(proxy_speed - full_speed),
                "proxy_foot_distance_m": proxy_foot,
                "full_foot_distance_m": full_foot,
                "foot_distance_error_m": abs(proxy_foot - full_foot),
                "full_safe": full["result"]["safe"],
                "full_nonfoot_frames": full["own_nonfoot_contact_frames"],
                "controlled_reception": full_explanation["controlled_reception"],
            }
        )
    if (
        base_proxy["trajectory_hash"]
        != hash_bytes((baseline_proxy / "single-live.npz").read_bytes())
        or base_full["trace_hash"] != hash_bytes((baseline_full / "live-short.npz").read_bytes())
        or base_full["physical_prefix_exact"] is not True
    ):
        raise ValueError("matched sealed zero parent required")
    baseline_speed = float(
        base_full["measurement"]["authoritative_explanation"]["tail_maximum_ball_speed_mps"]
    )
    speed_effect_direction_matches = all(
        (pair["proxy_speed_mps"] - baseline_speed) * (pair["full_speed_mps"] - baseline_speed) > 0
        for pair in pairs
    )
    ball_effect_observed = bool(
        speed_effect_direction_matches and all(pair["speed_error_mps"] <= 0.01 for pair in pairs)
    )
    joint_proxy_authorized = bool(
        ball_effect_observed
        and all(
            pair["foot_distance_error_m"] <= 0.02
            and pair["full_safe"] is True
            and not pair["full_nonfoot_frames"]
            for pair in pairs
        )
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("fidelity source changed during assessment")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_foot_capture_fidelity_gate.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "baseline_proxy_report_hash": base_proxy_hash,
        "baseline_full_report_hash": base_full_hash,
        "baseline_full_tail_speed_mps": baseline_speed,
        "paired_probes": pairs,
        "paired_speed_effect_direction_matches": speed_effect_direction_matches,
        "paired_ball_effect_observed": ball_effect_observed,
        "joint_task_training_proxy_authorized": joint_proxy_authorized,
        "foot_distance_ranking_authorized": joint_proxy_authorized,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline-proxy",
        "baseline-full",
        "anchor-proxy",
        "anchor-full",
        "challenger-proxy",
        "challenger-full",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    report = assess(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "paired_ball_effect_observed": report["paired_ball_effect_observed"],
                "joint_task_training_proxy_authorized": report[
                    "joint_task_training_proxy_authorized"
                ],
                "paired_probes": report["paired_probes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
