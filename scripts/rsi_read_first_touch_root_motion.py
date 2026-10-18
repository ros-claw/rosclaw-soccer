"""Read sampled root recovery from a fully audited catalog, zero new physics."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.physics.root_motion_window import summarize_root_motion_window
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--readback-hash", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if (
        args.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("clean frozen diagnostic and unused output required")
    pins = {str(args.config): digest(args.config)}

    def sealed(path, key="report_hash"):
        value = json.loads(path.read_text())
        pins[str(path)] = digest(path)
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("source seal differs")
        return value

    reviewed = sealed(args.readback)
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["output_root"])
    complete = sealed(root / "complete.json")
    declaration = sealed(root / "declaration.json")
    if (
        reviewed["report_hash"] != args.readback_hash
        or reviewed["schema"] != "soccer.rsi.mesh_contact_catalog_terminal_readback.v1"
        or reviewed["complete_hash"] != complete["report_hash"]
        or complete["declaration_hash"] != declaration["report_hash"]
        or len(reviewed["rows"]) != 16
        or len(complete["results"]) != 16
        or any(
            reviewed.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
            )
        )
        or any(declaration.get(k) != v for k, v in cfg.items())
        or declaration["input_file_hashes"].get(str(args.config)) != pins[str(args.config)]
    ):
        raise ValueError("exact complete independently audited contact catalog required")
    for relative in (
        "scripts/rsi_read_first_touch_root_motion.py",
        "src/rosclaw_soccer/physics/root_motion_window.py",
        "src/rosclaw_soccer/rsi/first_touch_outcome.py",
    ):
        pins[str(source / relative)] = digest(source / relative)
    rows = []
    for index, review_row in enumerate(reviewed["rows"]):
        receipt = sealed(root / f"receipt-{index:04d}.json", "receipt_hash")
        if (
            receipt != complete["results"][index]
            or receipt["job"] != review_row["job"]
            or receipt["receipt_hash"] != review_row["receipt_hash"]
        ):
            raise ValueError("same exact original ordered receipt required")
        trace_path = Path(receipt["witnesses"]["trace"]["path"])
        if digest(trace_path) != receipt["witnesses"]["trace"]["sha256"]:
            raise ValueError("original trace bytes changed")
        pins[str(trace_path)] = receipt["witnesses"]["trace"]["sha256"]
        with np.load(trace_path, allow_pickle=False) as archive:
            trace = dict(archive)
        outcome = label_first_touch(trace)
        if outcome.to_dict() != receipt["outcome"] or outcome.to_dict() != review_row["outcome"]:
            raise ValueError("original task labels differ")
        q, v = trace["canonical_qpos"], trace["canonical_qvel"]
        if any(
            a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all()
            for a, shape in ((q, (300, 1, 43)), (v, (300, 1, 41)))
        ):
            raise ValueError("whole canonical frame-start root capture required")
        recovery = None
        if outcome.quality_observed:
            first = outcome.first_foot_contact_frame
            section = slice(first + 1, first + 61)
            recovery = summarize_root_motion_window(
                q[section, 0, :7],
                v[section, 0, :6],
                sample_dt_s=0.02,
                forward_axis_world=np.array([1.0, 0.0, 0.0]),
            )
        terminal = summarize_root_motion_window(
            q[-30:, 0, :7],
            v[-30:, 0, :6],
            sample_dt_s=0.02,
            forward_axis_world=np.array([1.0, 0.0, 0.0]),
        )
        rows.append(
            dict(
                job=receipt["job"],
                receipt_hash=receipt["receipt_hash"],
                outcome=outcome.to_dict(),
                post_touch_60_sample_root_motion=recovery,
                final_30_sample_root_motion=terminal,
                successor_ready=None,
            )
        )
    if any(digest(path) != h for path, h in pins.items()):
        raise ValueError("frozen inputs changed during readback")
    result = dict(
        schema="soccer.rsi.first_touch_sampled_root_motion_readback.v1",
        source_head=args.source_head,
        upstream_readback_hash=reviewed["report_hash"],
        input_file_hashes=pins,
        rows=rows,
        measurement_clock="canonical before each 20ms control frame; not 2ms or cached body pose",
        complete_observed_window_samples=60,
        observed_position_endpoint_span_s=1.18,
        source_shared_payload_logical_seals_recomputed_here=False,
        original_task_labels_unchanged=True,
        successor_skill_success_measured=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
