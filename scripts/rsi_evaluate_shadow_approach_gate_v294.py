"""Retrospective, non-promoting shadow-contact option analysis of the v292 bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.parent_contact_approach_gate import choose_shadow_approach_gain
from rosclaw_soccer.sim.contracts import hash_json

V292_HASH = "sha256:8172d79326422997eeb5706e7ea554a080441d292b08dbfa48180334d9f086ae"


def evaluate(bank_root: Path) -> dict[str, Any]:
    bank = json.loads((bank_root / "bank_summary.json").read_text(encoding="utf-8"))
    if (
        bank.get("report_hash") != V292_HASH
        or bank.get("report_hash")
        != hash_json({k: v for k, v in bank.items() if k != "report_hash"})
        or bank.get("complete") is not True
        or len(bank.get("courses", [])) != 9
    ):
        raise ValueError("sealed complete v292 bank required")
    choices = []
    for row in bank["courses"]:
        seed, lane = row["seed"], row["lane"]
        parent_folder = bank_root / f"seed{seed}-lane{lane}-gain_12-parent"
        parent = json.loads((parent_folder / "report.json").read_text(encoding="utf-8"))
        receipt = audit_lateral_approach(parent_folder)
        if (
            parent["report_hash"] != row["arms"]["gain_12"]["parent_report_hash"]
            or receipt["report_hash"] != row["arms"]["gain_12"]["parent_command_audit_hash"]
            or parent["environments"][0]["course"] != row["course"]
        ):
            raise ValueError("candidate parent provenance mismatch")
        gain, reason = choose_shadow_approach_gain(parent)
        baseline = row["arms"]["gain_08"]
        selected = row["arms"]["gain_12" if gain == 1.2 else "gain_08"]
        choices.append(
            {
                "seed": seed,
                "lane": lane,
                "parent_report_hash": parent["report_hash"],
                "parent_command_audit_hash": receipt["report_hash"],
                "selected_gain": gain,
                "reason": reason,
                "baseline_high_quality": baseline["high_quality"],
                "selected_high_quality": selected["high_quality"],
                "baseline_clean_foot_only": baseline["clean_foot_only"],
                "selected_clean_foot_only": selected["clean_foot_only"],
                "baseline_out_of_play": baseline["maximum_lateral_excursion_m"] > 4,
                "selected_out_of_play": selected["maximum_lateral_excursion_m"] > 4,
                "baseline_reward": baseline["reward"],
                "selected_reward": selected["reward"],
            }
        )
    result: dict[str, Any] = {
        "schema": "rsi_retrospective_shadow_approach_gate_v1",
        "activation_ceiling": "SIM_ONLY",
        "data_status": "CONSUMED_DEVELOPMENT_NOT_GENERALIZATION",
        "v292_report_hash": V292_HASH,
        "choices": choices,
        "active": sum(row["selected_gain"] == 1.2 for row in choices),
        "vetoed": sum(row["selected_gain"] == 0.8 for row in choices),
        "high_quality_gain": sum(
            row["selected_high_quality"] - row["baseline_high_quality"] for row in choices
        ),
        "clean_foot_loss": sum(
            row["baseline_clean_foot_only"] and not row["selected_clean_foot_only"]
            for row in choices
        ),
        "new_out_of_play": sum(
            not row["baseline_out_of_play"] and row["selected_out_of_play"] for row in choices
        ),
        "mean_reward_gain": sum(row["selected_reward"] - row["baseline_reward"] for row in choices)
        / len(choices),
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    result = evaluate(args.bank_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["report_hash"])
    print(
        f"ACTIVE={result['active']} VETOED={result['vetoed']} "
        f"HIGH_GAIN={result['high_quality_gain']} CLEAN_LOSS={result['clean_foot_loss']}"
    )


if __name__ == "__main__":
    main()
