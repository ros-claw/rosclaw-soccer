"""Read-only reconstruction of progressive replay weights and all 156 executions."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.progressive_motor_actor import make_model, update_from_physics
from rosclaw_soccer.rsi.progressive_motor_bank import build_bank
from rosclaw_soccer.sim.contracts import hash_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root",
        "bank-root",
        "stable-root",
        "online-root",
        "bank-path",
        "preview-root",
        "validation-root",
        "fresh-root",
        "old-quarantine",
        "pool-ledger",
        "new-quarantine",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    bank = build_bank(
        args.stable_root,
        args.online_root,
        args.bank_path,
        args.preview_root,
        args.validation_root,
        args.fresh_root,
        args.old_quarantine,
        args.pool_ledger,
    )
    if bank != _sealed(args.bank_root / "learning_bank.json"):
        raise ValueError("broader replay no longer reconstructs from physical sources")
    parent = json.loads((args.stable_root / "stable_model.json").read_text())
    protocol = json.loads(args.new_quarantine.read_text())
    initial = make_model(
        parent,
        [r["observation"] for r in bank["courses"]],
        bank["protected_features"],
        learning_report_hash=bank["report_hash"],
        fresh_protocol_hash=hash_json(protocol),
    )
    candidate = update_from_physics(initial, bank["samples"], bank["report_hash"])
    if initial != json.loads(
        (args.bank_root / "initial_model.json").read_text()
    ) or candidate != json.loads((args.bank_root / "replay_model.json").read_text()):
        raise ValueError("expanded encoder, critic or actor weights do not reconstruct")
    commitment = json.loads((args.root / "commitment.json").read_text())
    summary = _sealed(args.root / "preview_summary.json")
    reproduction = json.loads((args.root / "reproduction.json").read_text())
    if (
        summary["commitment"] != commitment
        or summary["physical_episode_count"] != 156
        or commitment["initial_model_hash"] != initial["model_hash"]
        or commitment["candidate_model_hash"] != candidate["model_hash"]
        or commitment["learning_bank_hash"] != bank["report_hash"]
        or summary["promotion_authorized"] is not False
        or summary["hardware_authorized"] is not False
        or reproduction["matched_course_count"] != 52
    ):
        raise ValueError("progressive physical provenance or qualification drift")
    rows = []
    for index, course in enumerate(bank["courses"]):
        predecessor = _sealed(Path(course["predecessor_folder"]) / "report.json")
        old_parent = _sealed(Path(course["parent_report"]))
        parent_folder = args.root / f"seed{course['seed']}-lane{course['lane']}-reproduction-parent"
        physical_parent = _sealed(parent_folder / "report.json")
        from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach

        audit_lateral_approach(parent_folder)
        if physical_parent["source_hash"] != commitment["runner_hash"] or any(
            physical_parent[k] != old_parent[k]
            for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
        ):
            raise ValueError("physical parent is not an unchanged current-source reproduction")
        for arm, model in (
            ("progressive-reproduction", initial),
            ("progressive-replay", candidate),
        ):
            folder = args.root / f"seed{course['seed']}-lane{course['lane']}-{arm}-actor"
            report = _sealed(folder / "report.json")
            if (
                report["contact_motor_policy"]["progressive_motor_proof"]["model"] != model
                or report["parent_report_hash"] != physical_parent["report_hash"]
            ):
                raise ValueError("physical progressive action not bound to model and parent")
            measured = _outcome(folder, report["contact_motor_policy_hash"], commitment)["outcome"]
            if (
                arm == "progressive-reproduction" or course["predecessor_outcome"]["high_quality"]
            ) and any(report[k] != predecessor[k] for k in ("body_trace_hash", "trace_hash")):
                raise ValueError("predecessor or protected physical execution changed")
            if arm == "progressive-reproduction":
                cached = reproduction["rows"][index]
                if (
                    cached["index"] != index
                    or cached["report_hash"] != report["report_hash"]
                    or any(v != measured[k] for k, v in cached["outcome"].items())
                ):
                    raise ValueError("reproduction outcome drift")
            else:
                row = dict(
                    index=index,
                    seed=course["seed"],
                    lane=course["lane"],
                    report_hash=report["report_hash"],
                    **measured,
                )
                if row != summary["rows"][index]:
                    raise ValueError("broader learned outcome differs from physical evidence")
                rows.append(row)
    quality = sum(r["high_quality"] for r in rows)
    loss = sum(
        c["predecessor_outcome"]["high_quality"] and not r["high_quality"]
        for c, r in zip(bank["courses"], rows, strict=True)
    )
    outs = sum(r["maximum_lateral_excursion_m"] > 4 for r in rows)
    pelvis = all(r["minimum_pelvis_z_m"] >= 0.65 for r in rows)
    if (
        quality != summary["high_quality_count"]
        or loss != summary["protected_quality_loss"]
        or outs != summary["out_of_play_count"]
        or pelvis != summary["safe_pelvis_guardrail"]
    ):
        raise ValueError("broader physical score does not reconstruct")
    result = dict(
        schema="soccer.rsi.independent_progressive_review.v312",
        audited_physical_executions=156,
        independently_rebuilt_replay_hash=bank["report_hash"],
        source_summary_hash=summary["report_hash"],
        model_hash=candidate["model_hash"],
        high_quality_count=quality,
        protected_quality_loss=loss,
        out_of_play_count=outs,
        safe_pelvis_guardrail=pelvis,
        qualification="CONSUMED_PHYSICS_ONLY",
        fresh_evaluation_status="NOT_OPENED",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
