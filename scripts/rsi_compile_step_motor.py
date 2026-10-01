"""Seal an equivalent inference representation, optionally audit a completed trace."""

import argparse
import json
import time
from pathlib import Path

from rosclaw_soccer.rsi.compiled_step_inference import make_model, make_preview
from rosclaw_soccer.rsi.contact_motor_evidence import audit_motor_execution
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-folder", type=Path)
    parser.add_argument("--review-output", type=Path)
    args = parser.parse_args()
    if (args.evidence_folder is None) != (args.review_output is None):
        parser.error("evidence folder and review output must be supplied together")
    base = json.loads(args.model.read_text())
    wrapped = make_model(base)
    make_preview(wrapped)
    if args.evidence_folder is not None:
        report = _sealed(args.evidence_folder / "report.json")
        if report["contact_motor_policy"]["step_motor_proof"]["model"] != base:
            raise ValueError("completed trace used a different inference model")
        start = time.perf_counter()
        audited = audit_motor_execution(args.evidence_folder, report)
        elapsed = time.perf_counter() - start
        review = dict(
            schema="soccer.rsi.compiled_step_decoder_review.v1",
            original_report_hash=report["report_hash"],
            base_model_hash=base["model_hash"],
            compiled_model_hash=wrapped["model_hash"],
            audited_frames=300,
            compilation_changes_weights=False,
            original_run_used_compilation=False,
            actual_actions_reconstructed=audited,
            audit_elapsed_sec=elapsed,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        review["report_hash"] = hash_json(review)
        write_once(args.review_output, review)
        print(json.dumps(review), flush=True)
    write_once(args.output, wrapped)
    print(json.dumps(dict(compiled_model_hash=wrapped["model_hash"])), flush=True)


if __name__ == "__main__":
    main()
