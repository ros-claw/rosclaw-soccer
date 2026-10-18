"""Recover a previously audited public C7 success, not a new teacher or speed ablation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def require_actual_zero_control(receipt, trace):
    if (
        receipt.get("job") != "zero_control-2026101801-context-7"
        or receipt.get("group") != 77
        or type(receipt.get("cerebellar_model_optimizer_steps")) is not int
        or receipt["cerebellar_model_optimizer_steps"] != 0
        or type(receipt.get("cerebellar_sampling_draws")) is not int
        or receipt["cerebellar_sampling_draws"] != 0
        or receipt.get("cerebellar_inference_calls") != 3000
    ):
        raise ValueError("exact original zero-control context7 receipt required")
    for key in (
        "cerebellar_sampling_mean",
        "cerebellar_sampling_latent_action",
        "requested_cerebellar_torque_offset_nm",
        "applied_cerebellar_torque_offset_nm",
    ):
        a = np.asarray(trace.get(key))
        if a.shape != (300, 1, 10, 29) or not np.isfinite(a).all() or np.any(a != 0):
            raise ValueError("all3000 actual cerebellar means, requests and offsets must be zero")
    sampled = np.asarray(trace.get("cerebellar_sampling_performed"))
    if sampled.shape != (300, 1, 10, 1) or np.any(sampled != 0):
        raise ValueError("zero-control cannot contain random exploration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cerebellar-complete",
        "cerebellar-audit",
        "cerebellar-reader",
        "screen-complete",
        "screen-audit",
        "screen-reader",
        "catalog-fit",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("unused research readback required")
    pins = {}

    def digest(path):
        with Path(path).open("rb") as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()

    def sealed(path):
        d = json.loads(Path(path).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole source seal differs")
        pins[str(path)] = digest(path)
        return d

    complete, audit = sealed(args.cerebellar_complete), sealed(args.cerebellar_audit)
    screen, review = sealed(args.screen_complete), sealed(args.screen_audit)
    fit = sealed(args.catalog_fit)
    if (
        complete.get("native_bodies") != 44
        or len(complete.get("results", [])) != 44
        or len(audit.get("rows", [])) != 44
        or audit.get("schema") != "soccer.rsi.mesh_cerebellar_terminal_readback.v1"
        or audit.get("complete_hash") != complete["report_hash"]
        or audit.get("audit_source_hash") != digest(args.cerebellar_reader)
        or audit.get("full_source_and_witness_bytes_verified") is not True
        or audit.get("actual_persistent_actor_critic_adam_reconstructed") is not True
        or len(screen.get("results", [])) != 18
        or review.get("schema") != "soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1"
        or review.get("complete_hash") != screen["report_hash"]
        or review.get("reader_source_hash") != digest(args.screen_reader)
        or any(
            review.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
            )
        )
    ):
        raise ValueError("both whole original audited studies required")
    zero, baseline = complete["results"][7], screen["results"][15]
    traces, commitments = [], []
    for receipt, source_rows, cerebellar in (
        (zero, audit["rows"], True),
        (baseline, review["rows"], False),
    ):
        if receipt["receipt_hash"] != hash_json(
            {k: v for k, v in receipt.items() if k != "receipt_hash"}
        ) or not any(
            r["receipt_hash"] == receipt["receipt_hash"]
            and r["job"] == receipt["job"]
            and r["outcome"] == receipt["outcome"]
            for r in source_rows
        ):
            raise ValueError("exact endpoint must belong to its whole original audit")
        for witness in receipt["witnesses"].values():
            if digest(witness["path"]) != witness["sha256"]:
                raise ValueError("source witness bytes changed")
            pins[witness["path"]] = witness["sha256"]
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as a:
            trace = dict(a)
        outcome = label_first_touch(trace)
        kwargs = (
            dict(cerebellar_model_hash=zero["cerebellar_model_hash"], cerebellar_sampling_draws=0)
            if cerebellar
            else {}
        )
        alignment = require_first_touch_qualification(
            sealed(receipt["witnesses"]["independent_review"]["path"]),
            sealed(receipt["witnesses"]["foundation_review"]["path"]),
            outcome,
            **kwargs,
        )
        if (
            outcome.to_dict() != receipt["outcome"]
            or alignment != receipt["qualification_alignment"]
        ):
            raise ValueError("primitive and original independent physical review differ")
        # Native commitments have no report_hash field; their exact bytes are
        # authenticated by the receipt witness above, not a fabricated seal.
        commitments.append(json.loads(Path(receipt["witnesses"]["commitment"]["path"]).read_text()))
        traces.append(trace)
    require_actual_zero_control(zero, traces[0])
    model = zero["model_witness"]
    if digest(model["path"]) != model["sha256"]:
        raise ValueError("original zero actor file changed")
    pins[model["path"]] = model["sha256"]
    same_keys = (
        "course",
        "seed",
        "lane",
        "foundation_hash",
        "late_actor_hash",
        "step_model_hash",
        "motor_policy_hash",
        "compiled_model_hash",
        "swing_mode",
        "approach_mode",
    )
    if (
        zero["candidate"] != fit["options"][2]["candidate"]
        or baseline["candidate"] != fit["options"][4]["candidate"]
        or zero["planning_context_hash"] != baseline["planning_context_hash"]
        or baseline["job"] != "teacher-context-7"
        or any(commitments[0][k] != commitments[1][k] for k in same_keys)
        or any(
            traces[0][k][0].tobytes() != traces[1][k][0].tobytes()
            for k in ("canonical_qpos", "canonical_qvel")
        )
    ):
        raise ValueError(
            "same initial course/body/parent but explicit different catalog options required"
        )
    for p in (args.cerebellar_reader, args.screen_reader, Path(__file__)):
        pins[str(p)] = digest(p)
    result = dict(
        schema="soccer.rsi.archived_receding_expert_readback.v1",
        recovered_receipt_hash=zero["receipt_hash"],
        comparison_receipt_hash=baseline["receipt_hash"],
        recovered_outcome=zero["outcome"],
        comparison_outcome=baseline["outcome"],
        recovered_catalog_index=2,
        comparison_catalog_index=4,
        matched_commitment_keys=list(same_keys),
        exact_initial_canonical_qpos_qvel_equal=True,
        actual_all3000_cerebellar_outputs_exactly_zero=True,
        native_kernel_source_hashes=[c["source_hash"] for c in commitments],
        same_native_kernel_source=commitments[0]["source_hash"] == commitments[1]["source_hash"],
        single_parameter_speed_ablation=False,
        historical_evidence_reuse=True,
        all_robot_body_contact_review_completed_here=False,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        independent_courses=1,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        input_file_hashes=pins,
        next_required_work="full-body review and nearby public contexts before admission",
    )
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError("source bytes changed during readback")
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
