"""After terminal audit, summarize all four original/new measured endpoints.

No physics, retry, training, policy admission or private holdout access.
"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reader-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    from rosclaw_soccer.rsi.contact_exit_closed_loop_metrics import closed_loop_metrics
    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    cfg = json.loads(args.config.read_text())
    directional = cfg.get("use_directional_contact_compliance", False)
    torque_mpc = cfg.get("use_contact_exit_torque_mpc", False)
    surface = cfg.get("use_mesh_surface_swing", False)
    if (
        any(type(x) is not bool for x in (directional, torque_mpc, surface))
        or sum((directional, torque_mpc, surface)) != 1
    ):
        raise ValueError("one explicit unmixed postcontact intervention required")
    kind = (
        "mesh_surface_swing"
        if surface
        else "directional_contact_compliance"
        if directional
        else "contact_exit_torque_mpc"
    )
    cases, bodies = (6, 7) if surface else (4, 4)
    audit = _sealed(args.audit)
    root = Path(cfg["output_root"])
    complete, declaration = (_sealed(root / name) for name in ("complete.json", "declaration.json"))
    if (
        args.output.exists()
        or cfg.get("suppress_postcontact_swing_release") is not False
        or audit.get("schema") != f"soccer.rsi.{kind}_terminal_readback.v1"
        or audit.get("reader_source_hash") != digest(args.reader_source)
        or audit.get("complete_hash") != complete["report_hash"]
        or complete.get("declaration_hash") != declaration["report_hash"]
        or complete.get("schema") != f"soccer.rsi.{kind}_pilot_complete.v1"
        or any(declaration.get(k) != v for k, v in cfg.items())
        or len(complete.get("results", [])) != bodies
        or len(audit.get("rows", [])) != bodies
        or len(cfg.get("baseline_receipts", [])) != cases
        or any(
            audit.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
                "all_original_pre_activation_prefixes_bit_equal"
                if surface
                else "all_original_first_contact_prefixes_bit_equal",
                "reused_endpoints_authenticated_via_prior_complete_terminal_readbacks",
            )
        )
    ):
        raise ValueError("whole audited four-pair evidence and unused summary output required")
    if surface and (
        audit.get("disabled_complete_300_frame_noninterference_verified") is not True
        or audit.get("original_none_raw_retention") != complete.get("original_none_raw_retention")
    ):
        raise ValueError("disabled source parity and original None retention must be audited")
    pins = {
        str(path): digest(path)
        for path in (
            args.config,
            args.audit,
            args.reader_source,
            root / "complete.json",
            root / "declaration.json",
            Path(__file__),
        )
    }
    controls, evaluated, rows = {}, {}, []
    for item, receipt, audited in zip(
        cfg["baseline_receipts"],
        complete["results"][1:] if surface else complete["results"],
        audit["rows"][1:] if surface else audit["rows"],
        strict=True,
    ):
        baseline = json.loads(Path(item["path"]).read_text())
        if (
            item["pair_id"] != receipt["pair_id"]
            or baseline.get("receipt_hash")
            != hash_json({k: v for k, v in baseline.items() if k != "receipt_hash"})
            or item["receipt_hash"] != baseline["receipt_hash"]
            or item["file_hash"] != digest(Path(item["path"]))
            or receipt["receipt_hash"]
            != hash_json({k: v for k, v in receipt.items() if k != "receipt_hash"})
            or audited["receipt_hash"] != receipt["receipt_hash"]
            or audited["outcome"] != receipt["outcome"]
            or audited["job"] != receipt["job"]
        ):
            raise ValueError("all original baseline and new receipt identities required")
        metrics = []
        pins[item["path"]] = item["file_hash"]
        for endpoint in (baseline, receipt):
            witness = endpoint["witnesses"]["trace"]
            path = Path(witness["path"])
            if digest(path) != witness["sha256"]:
                raise ValueError("actual endpoint trace changed after terminal audit")
            pins[str(path)] = witness["sha256"]
            with np.load(path, allow_pickle=False) as archive:
                measured = closed_loop_metrics(dict(archive))
            if measured["outcome"] != endpoint["outcome"]:
                raise ValueError("closed-loop primitive labels differ from audited endpoint")
            metrics.append(measured)
        controls[item["pair_id"]] = metrics[0]["outcome"]
        evaluated[item["pair_id"]] = metrics[1]["outcome"]
        rows.append(dict(pair_id=item["pair_id"], baseline=metrics[0], evaluated=metrics[1]))
    paired = assess_pairs(controls, evaluated)
    if paired != complete["paired_development"] or paired != audit["paired_development"]:
        raise ValueError("independently summarized complete pair outcomes differ")
    passed = (
        paired["rescued_pairs"] >= 1
        and paired["regressed_pairs"] == 0
        and paired["newly_unsafe_pairs"] == 0
    )
    if surface:
        passed = passed and all(
            complete["original_none_raw_retention"][k] == 0
            for k in ("regressed_pairs", "newly_unsafe_pairs")
        )
    result = dict(
        schema="soccer.rsi.mesh_surface_measured_summary.v1"
        if surface
        else "soccer.rsi.directional_compliance_measured_summary.v1"
        if directional
        else "soccer.rsi.contact_exit_mpc_measured_summary.v1",
        complete_hash=complete["report_hash"],
        terminal_readback_hash=audit["report_hash"],
        paired_development=paired,
        rows=rows,
        decision="LOCAL_PILOT_PASS_EXPANSION_REQUIRED" if passed else "REJECT_LOCAL_TEACHER",
        next_required_work=(
            "expanded original success/failure retention and perturbations before student training"
            if passed
            else (
                "inspect actual activation/clipping, peak-bin persistence, reversal "
                "and body outcomes; "
                "no same-setting retry or student training from rejected teacher"
            )
        ),
        input_file_hashes=pins,
        source_body_integrations=6000 * bodies,
        new_body_integrations=0,
        new_optimizer_updates=0,
        successor_ready=None,
        additional_all_robot_contact_safety_review_required=True,
        all_robot_body_contact_safety_confirmed=False,
        teacher_qualified=False,
        private_fresh_accessed=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if surface:
        result["original_none_raw_retention"] = complete["original_none_raw_retention"]
    if any(digest(Path(name)) != expected for name, expected in pins.items()):
        raise ValueError("summary input bytes changed during readback")
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["decision"], paired, flush=True)


if __name__ == "__main__":
    main()
