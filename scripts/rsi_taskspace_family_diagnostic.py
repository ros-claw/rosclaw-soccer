"""Separate action-support failure from selection failure on consumed G1 holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_contextual_phase_train import lane_outcomes

PROTOCOL = Path("docs/rsi/protocols/first-touch-family-failure-diagnostic-v9c.json")
SELECTOR_PRIORITY_EXTRA_RESCUES = 3  # Matches the frozen protocol's written decision.


def oracle_support(clean: np.ndarray[Any, Any]) -> dict[str, Any]:
    if clean.ndim != 2 or clean.shape[1] != 4 or not np.isin(clean, (0, 1)).all():
        raise ValueError("parent/family/up/wide paired clean labels required")
    parent, family, up, wide = clean.T.astype(np.bool_)
    possible = (~parent) & (up | wide)
    selected = (~parent) & family
    return {
        "parent_clean": int(np.count_nonzero(parent)),
        "family_clean": int(np.count_nonzero(family)),
        "oracle_new_rescues": np.flatnonzero(possible).tolist(),
        "family_new_rescues": np.flatnonzero(selected).tolist(),
        "oracle_unselected_rescues": np.flatnonzero(possible & ~selected).tolist(),
        "up_regressed_parent": np.flatnonzero(parent & ~up).tolist(),
        "wide_regressed_parent": np.flatnonzero(parent & ~wide).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-actor", required=True, type=Path)
    parser.add_argument("--family-actor", required=True, type=Path)
    parser.add_argument("--holdout-verdict", required=True, type=Path)
    parser.add_argument("--set", dest="sets", nargs=5, action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(args.sets) != 2:
        parser.error("two consumed paired groups and fresh output required")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    verdict = json.loads(args.holdout_verdict.read_text(encoding="utf-8"))
    if (
        verdict.get("verdict") != "REJECTED"
        or verdict.get("report_hash") != protocol["frozen_holdout_verdict_hash"]
        or verdict.get("report_hash")
        != hash_json({key: value for key, value in verdict.items() if key != "report_hash"})
    ):
        raise ValueError("missing rejected frozen holdout commitment")
    rows = []
    seeds = []
    for bank_raw, parent_raw, family_raw, up_raw, wide_raw in args.sets:
        bank = Path(bank_raw)
        paths = list(map(Path, (parent_raw, family_raw, up_raw, wide_raw)))
        bank_audit = audit_snapshot_bank(bank)
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        source = Path(manifest["snapshots"][0]["source_folder"])
        seed = json.loads((source / "report.json").read_text(encoding="utf-8")).get(
            "training_course_seed"
        )
        count = manifest["snapshot_count"]
        if (
            seed in seeds
            or seed not in protocol["consumed_course_seeds"]
            or manifest.get("partition") != "SEALED_HOLDOUT"
        ):
            raise ValueError("unregistered consumed diagnostic group")
        seeds.append(seed)
        audits = []
        labels = []
        for action, path in zip(("parent", "family", "up", "wide"), paths, strict=True):
            audit = audit_snapshot_replay(
                path,
                snapshot_bank=bank,
                local_phase_policy_path=args.phase_actor,
                taskspace_family_policy_path=(args.family_actor if action == "family" else None),
            )
            report = json.loads((path / "report.json").read_text(encoding="utf-8"))
            if (
                audit["sample_count"] != count
                or (action == "parent" and report.get("taskspace_forward_m") is not None)
                or (action == "family" and not audit.get("taskspace_action_audited"))
                or (
                    action in ("up", "wide")
                    and (
                        not audit.get("taskspace_action_audited")
                        or report.get("taskspace_forward_m") != 0.08
                        or report.get("taskspace_vertical_offset_m")
                        != (0.04 if action == "up" else 0.0)
                        or report.get("taskspace_lateral_cap_m")
                        != (0.10 if action == "wide" else 0.05)
                    )
                )
            ):
                raise ValueError("unverified same-state physical diagnostic")
            labels.append(lane_outcomes(path, count)[1])
            audits.append(audit["report_hash"])
        row = oracle_support(np.column_stack(labels))
        row.update(
            seed=seed, sample_count=count, bank_hash=bank_audit["manifest_hash"], audits=audits
        )
        rows.append(row)
    if set(seeds) != set(protocol["consumed_course_seeds"]):
        raise ValueError("missing preregistered consumed group")
    oracle = sum(len(row["oracle_new_rescues"]) for row in rows)
    selected = sum(len(row["family_new_rescues"]) for row in rows)
    result = {
        "schema": "rsi_taskspace_family_failure_diagnostic_v9c",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_json(protocol),
        "frozen_holdout_verdict_hash": verdict["report_hash"],
        "rows": rows,
        "oracle_new_rescue_count": oracle,
        "family_new_rescue_count": selected,
        "oracle_minus_family_rescue_count": oracle - selected,
        "primary_bottleneck": (
            "selector_timing_or_features"
            if oracle - selected >= SELECTOR_PRIORITY_EXTRA_RESCUES
            else "action_support"
        ),
        "oracle_is_deployable": False,
        "fresh_holdout_reuse_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"bottleneck": result["primary_bottleneck"], "report_hash": result["report_hash"]}
        )
    )


if __name__ == "__main__":
    main()
