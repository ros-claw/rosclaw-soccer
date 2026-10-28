"""Compare each actual SIM run to its separately executed reference arm."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.oof_reference_identity import exact_reference_arrays
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def study_predictions(actual: dict, risk: dict) -> list:
    """Bind the actual selector to its correctly labelled development study."""
    variant = actual.get("risk_variant")
    expected_status = {
        "ball_augmented": "RELAY_OOF_RISK_PHYSICAL_DEVELOPMENT_NOT_SUCCESS",
        "single": "RELAY_SINGLE_RISK_PHYSICAL_DEVELOPMENT_NOT_SUCCESS",
    }
    if variant not in expected_status or actual.get("status") != expected_status[variant]:
        raise ValueError("unsupported or mismatched actual risk variant")
    section = "ball_augmented" if variant == "ball_augmented" else "in_sample_rehearsal"
    return risk[section]["predictions"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "candidate", "actual", "risk", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("baseline-hash", "candidate-hash", "actual-hash", "risk-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    pins = (
        (args.baseline, args.baseline_hash),
        (args.candidate, args.candidate_hash),
        (args.actual, args.actual_hash),
        (args.risk, args.risk_hash),
    )
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in pins)
    ):
        raise ValueError("exact immutable receipts and new external output required")
    baseline, candidate, actual, risk = (json.loads(p.read_text()) for p, _ in pins)
    if (
        baseline.get("status") != "RELAY_NONDEFAULT_SMOOTH_RETURN_PHYSICAL_CANDIDATE_NOT_SUCCESS"
        or candidate.get("status")
        != "RELAY_LONG_PROBE_CAUSAL_RETURN_PHYSICAL_CANDIDATE_NOT_SUCCESS"
        or actual.get("risk_receipt_hash") != args.risk_hash
        or any(
            baseline.get(k) != candidate.get(k) or baseline.get(k) != actual.get(k)
            for k in (
                "source_hash",
                "audit_hash",
                "snapshot_hash",
                "frontier_hash",
                "role_proof_hash",
                "expanded_velocity_proof_hash",
            )
        )
        or any(
            d.get(k) is not False
            for d in (baseline, candidate, actual, risk)
            for k in (
                "independent_fresh",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
    ):
        raise ValueError("exact common-source unpromoted actual and reference arms required")
    predictions = study_predictions(actual, risk)
    banks = [{r["index"]: r for r in d["rows"]} for d in (baseline, candidate, actual)]
    if any(len(d["rows"]) != 24 for d in (baseline, candidate, actual)) or any(
        len(bank) != 24 or set(bank) != set(banks[0]) for bank in banks
    ):
        raise ValueError("all 24 unique matched cases required")
    ids = [(r["snapshot_hash"], r["index"], r["context"]) for r in risk["identities"]]
    if len(ids) != 72 or len(set(ids)) != 72 or len(predictions) != 72:
        raise ValueError("exact development identities and predictions required")
    rows = []
    for index in sorted(banks[0]):
        br, cr, ar = (bank[index] for bank in banks)
        if any(
            br[k] != cr[k] or br[k] != ar[k]
            for k in ("context", "side", "receiver_lateral_offset_m")
        ) or ar["selector_class"] not in (0, 1):
            raise ValueError("same source case and explicit actual model choice required")
        identity = (actual["snapshot_hash"], index, ar["context"])
        if identity not in ids or predictions[ids.index(identity)] != ar["selector_class"]:
            raise ValueError("actual model choice differs from bound development study")
        reference_path, reference_row = (
            (args.candidate, cr) if ar["selector_class"] else (args.baseline, br)
        )
        histories = []
        for receipt, row in ((args.actual, ar), (reference_path, reference_row)):
            path = receipt.parent / row["physical_trajectory"]
            if (
                path.is_symlink()
                or path.resolve().parent != receipt.parent.resolve()
                or digest(path) != row["physical_trajectory_hash"]
            ):
                raise ValueError("confined exact physical trajectory required")
            with np.load(path, allow_pickle=False) as archive:
                histories.append({k: archive[k].copy() for k in archive.files})
            if digest(path) != row["physical_trajectory_hash"]:
                raise ValueError("physical trajectory changed during audit")
        legacy = (
            not ar["selector_class"]
            and baseline.get("source_head") == "ee82b7079a7646408806d5abc5fd4791154dca5d"
        )
        match = exact_reference_arrays(*histories, legacy_empty_follow_unavailable=legacy)
        outcomes = all(
            ar[k] == reference_row[k]
            for k in (
                "body_failure",
                "receiving_ready",
                "consecutive_controls",
                "first_contact_frame_and_substep",
                "first_contact_groups",
                "return_controls",
            )
        )
        rows.append(
            dict(
                index=index,
                context=ar["context"],
                selected_arm="candidate" if ar["selector_class"] else "baseline",
                arrays_exact=match,
                legacy_follow_arrays_unavailable=legacy,
                outcomes_exact=outcomes,
                actual_body_failure=ar["body_failure"],
                receiving_ready=ar["receiving_ready"],
            )
        )
    if any(digest(p) != h for p, h in pins):
        raise ValueError("input changed during audit")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="RISK_REFERENCE_REPLAY_DEVELOPMENT_NOT_FRESH",
            risk_variant=actual["risk_variant"],
            source_head=args.expected_head,
            baseline_hash=args.baseline_hash,
            candidate_hash=args.candidate_hash,
            actual_hash=args.actual_hash,
            risk_hash=args.risk_hash,
            rows=rows,
            all_available_arrays_exact=all(r["arrays_exact"] and r["outcomes_exact"] for r in rows),
            all_arrays_exact=all(
                r["arrays_exact"]
                and r["outcomes_exact"]
                and not r["legacy_follow_arrays_unavailable"]
                for r in rows
            ),
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
