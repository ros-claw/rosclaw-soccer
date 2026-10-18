"""Once-only zero/teacher/student whole physical evidence comparison; no fit.

Wait for both declared original native/audit/dense chains. Rebuild both dense
readouts, authenticate sources and all witness bytes, then choose a research
branch using training sources only. Never executes physics or optimization.
"""

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.rsi.matched_reference_comparison import compare_matched_references
from rosclaw_soccer.rsi.reference_student_exam_diagnostics import (
    measured_readout as _measured_readout,
)
from rosclaw_soccer.rsi.reference_student_exam_diagnostics import require_cohort as _require_cohort
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest as _digest
from scripts.rsi_stage_release_contact_replay import require_terminal as _require_terminal

digest = cast(Callable[[str | Path], str], _digest)
require_terminal = cast(Callable[[str], None], _require_terminal)
require_cohort = cast(Callable[..., dict[str, Any]], _require_cohort)
measured_readout = cast(Callable[..., dict[str, Any]], _measured_readout)


def load_cohort(
    spec: dict[str, Any], *, sealed: Callable[..., dict[str, Any]], pins: dict[str, str]
) -> dict[str, Any]:
    """Authenticate the complete original source chain, including every failure."""
    cp = Path(spec["contact_configuration"])
    if digest(cp) != spec["contact_configuration_file_hash"]:
        raise ValueError("unchanged original contact-stage configuration required")
    pins[str(cp)] = digest(cp)
    cc = json.loads(cp.read_text())
    for unit in (cc["producer_unit"], cc["audit_unit"], spec["contact_unit"]):
        require_terminal(unit)
    pp = Path(cc["producer_config"])
    if digest(pp) != cc["producer_config_file_hash"]:
        raise ValueError("unchanged original physical producer configuration required")
    pins[str(pp)] = digest(pp)
    cfg = json.loads(pp.read_text())
    original_source = Path(cfg["source_root"])
    if (
        cc["source_head"] != cfg["source_head"]
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=original_source, text=True
        ).strip()
        != cfg["source_head"]
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=original_source, text=True
        ).strip()
    ):
        raise ValueError("unchanged same frozen original native/audit/contact source required")
    reader = Path(cfg["source_root"]) / "scripts/rsi_read_reference_student_exam.py"
    if Path(cc["audit_source"]).resolve() != reader.resolve():
        raise ValueError("original producer-owned audit source required")
    pins[str(reader)] = digest(reader)
    root = Path(cfg["output_root"])
    declaration, complete = (sealed(root / n) for n in ("declaration.json", "complete.json"))
    audit, dense = sealed(cc["readback"]), sealed(cc["complete_path"])
    pairs = require_cohort(cfg, declaration, complete, audit)
    replay = sealed(Path(cc["replay_output_root"]) / "complete.json")
    metrics = sealed(cc["metrics_output"])
    measured = measured_readout(declaration, complete, replay, metrics)
    if (
        audit["reader_source_hash"] != digest(reader)
        or dense.get("schema") != "soccer.rsi.reference_exam_contact_complete.v1"
        or dense["upstream_readback_hash"] != audit["report_hash"]
        or dense["replay_complete_hash"] != replay["report_hash"]
        or dense["metrics_hash"] != metrics["report_hash"]
        or dense["paired_results"] != pairs
        or dense["measured_contact_readout"] != measured
        or dense["new_fixed_control_replay_substeps"] != cc["new_substep_budget"]
        or dense["total_metrics_endpoints"] != len(complete["results"])
        or any(
            type(dense.get(k)) is not int or dense[k] != 0
            for k in (
                "new_closed_loop_policy_trials",
                "new_neural_inferences",
                "new_optimizer_updates",
            )
        )
        or any(
            dense.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "dataset_admission_authorized",
                "neural_training_authorized",
                "teacher_qualified",
                "fresh_exam",
            )
        )
    ):
        raise ValueError("whole original audit and dense readout must rebuild exactly")
    for i, r in enumerate(complete["results"]):
        if sealed(root / f"receipt-{i:04d}.json", "receipt_hash") != r:
            raise ValueError("all original owned physical receipt files required")
    for owner in (declaration, audit):
        for name, h in {**owner["input_file_hashes"], **owner.get("prepared_pins", {})}.items():
            if digest(name) != h:
                raise ValueError("original physical source or witness bytes changed")
            if name in pins and pins[name] != h:
                raise ValueError("cross-cohort shared input bytes must agree")
            pins[name] = h
    return dict(
        config=cfg, declaration=declaration, complete=complete, audit=audit, measured=measured
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        cfg.get("profile")
        not in (
            "matched-positive-zero-teacher-student-comparison-v1",
            "actual-winner-versus-recorded-zero-comparison-v1",
        )
        or cfg.get("mode") != "SIM_ONLY"
        or root.exists()
        or time.time() >= cfg["deadline_unix"]
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "neural_training_authorized",
                "dataset_admission_authorized",
                "private_fresh_accessed",
            )
        )
        or set(cfg["cohorts"]) != {"zero", "student"}
        or Path(__file__).resolve() != source / "scripts/rsi_compare_matched_references.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused frozen once-only SIM whole-comparison report required")
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path: str | Path, key: str = "report_hash") -> dict[str, Any]:
        path = Path(path)
        value = json.loads(path.read_text())
        if value.get(key) != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("unchanged whole sealed original evidence required")
        pins[str(path)] = digest(path)
        return cast(dict[str, Any], value)

    cohorts = {
        role: load_cohort(spec, sealed=sealed, pins=pins) for role, spec in cfg["cohorts"].items()
    }
    comparison = compare_matched_references(
        cohorts["zero"],
        cohorts["student"],
        allow_zero_normalization_reuse=cfg["profile"]
        == "actual-winner-versus-recorded-zero-comparison-v1",
    )
    if args.preflight:
        print(
            "MATCHED_ZERO_TEACHER_STUDENT_WHOLE_EVIDENCE_READY_ZERO_PHYSICS_OR_OPTIMIZER_ALLOCATIONS",
            flush=True,
        )
        return
    root.mkdir()
    write_once(root / "started.json", dict(configuration=cfg, automatic_retry_authorized=False))
    try:
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("source or original evidence changed during comparison")
        report = dict(
            schema="soccer.rsi.matched_reference_comparison_complete.v1",
            configuration=cfg,
            input_file_hashes=pins,
            cohort_complete_hashes={
                role: c["complete"]["report_hash"] for role, c in cohorts.items()
            },
            cohort_audit_hashes={role: c["audit"]["report_hash"] for role, c in cohorts.items()},
            comparison=comparison,
            original_task_labels_unchanged=True,
            scientific_recommendation_not_automatic_successor=True,
            new_optimizer_updates=0,
            new_physical_steps=0,
            new_foundation_inferences=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        report["report_hash"] = hash_json(report)
        write_once(root / "complete.json", report)
        print(
            "MATCHED_REFERENCE_COMPARISON_COMPLETE_NOT_PROMOTION",
            comparison["next_research"],
            report["report_hash"],
            flush=True,
        )
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(type=type(error).__name__, message=str(error), automatic_retry_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
