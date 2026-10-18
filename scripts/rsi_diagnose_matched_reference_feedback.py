"""Read a whole closed reference comparison and diagnose all eight paired traces."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.matched_reference_comparison import compare_matched_references
from rosclaw_soccer.rsi.reference_feedback_divergence import reference_feedback_divergence
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_compare_matched_references import digest, load_cohort, require_terminal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    source, output = Path(cfg["source_root"]), Path(cfg["output"])
    if (
        cfg.get("profile") != "whole-reference-feedback-diagnosis-v1"
        or cfg.get("mode") != "SIM_ONLY"
        or output.exists()
        or cfg.get("hardware_authorized") is not False
        or cfg.get("promotion_authorized") is not False
        or Path(__file__).resolve() != source / "scripts/rsi_diagnose_matched_reference_feedback.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused frozen simulation-only readout required")
    require_terminal(cfg["comparison_unit"])
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        path = Path(path)
        v = json.loads(path.read_text())
        if v.get(key) != hash_json({k: x for k, x in v.items() if k != key}):
            raise ValueError("whole sealed source evidence required")
        pins[str(path)] = digest(path)
        return v

    comparison = sealed(cfg["comparison_complete"])
    cc = comparison["configuration"]
    if (
        comparison["report_hash"] != cfg["comparison_complete_hash"]
        or comparison["schema"] != "soccer.rsi.matched_reference_comparison_complete.v1"
        or cc["profile"]
        not in (
            "matched-positive-zero-teacher-student-comparison-v1",
            "actual-winner-versus-recorded-zero-comparison-v1",
        )
        or cc["mode"] != "SIM_ONLY"
        or set(cc["cohorts"]) != {"zero", "student"}
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cc["source_root"], text=True
        ).strip()
        != cc["source_head"]
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cc["source_root"], text=True
        ).strip()
    ):
        raise ValueError("original whole comparison and frozen producer required")
    cohorts = {
        role: load_cohort(spec, sealed=sealed, pins=pins) for role, spec in cc["cohorts"].items()
    }
    rebuilt = compare_matched_references(
        cohorts["zero"],
        cohorts["student"],
        allow_zero_normalization_reuse=cc["profile"]
        == "actual-winner-versus-recorded-zero-comparison-v1",
    )
    if rebuilt != comparison["comparison"]:
        raise ValueError("whole original scientific comparison must reconstruct")
    for name, h in comparison["input_file_hashes"].items():
        if digest(name) != h or (name in pins and pins[name] != h):
            raise ValueError("original whole source/witness bytes changed")
        pins[name] = h
    receipts = {
        role: {
            r["group"]: r
            for r in c["complete"]["results"]
            if r["plan"] == "positive19" and r["role"] == role
        }
        for role, c in cohorts.items()
    }
    groups = [12, 13, 28, 29, 76, 77, 92, 93]
    if any(list(rs) != groups for rs in receipts.values()):
        raise ValueError("all original ordered eight courses including failures required")
    rows = []
    for group in groups:
        traces = {}
        for role in ("zero", "student"):
            w = receipts[role][group]["witnesses"]["trace"]
            if pins.get(w["path"]) != w["sha256"] or digest(w["path"]) != w["sha256"]:
                raise ValueError("owned original trace bytes required")
            with np.load(w["path"], allow_pickle=False) as a:
                traces[role] = dict(a)
        rows.append(
            dict(
                group=group,
                source_receipt_hashes={
                    role: receipts[role][group]["receipt_hash"] for role in receipts
                },
                original_outcomes={role: receipts[role][group]["outcome"] for role in receipts},
                measured=reference_feedback_divergence(traces["zero"], traces["student"]),
            )
        )
    if any(digest(name) != h for name, h in pins.items()):
        raise ValueError("source or physical evidence changed during diagnosis")
    result = dict(
        schema="soccer.rsi.whole_reference_feedback_diagnosis.v1",
        configuration=cfg,
        comparison_hash=comparison["report_hash"],
        input_file_hashes=pins,
        rows=rows,
        all_failures_retained=True,
        new_physical_steps=0,
        optimizer_updates=0,
        task_gates_changed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(output, result)
    print("WHOLE_REFERENCE_FEEDBACK_DIAGNOSED_NOT_NEW_PHYSICS", result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
