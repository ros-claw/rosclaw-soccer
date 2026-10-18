"""Explicit successor evidence load after the previous whole physical exam.

Default exam path never calls this module. No physical allocation or fitting.
"""

import json
from pathlib import Path

from rosclaw_soccer.rsi.reference_ablation_evidence import require_matched_fit
from scripts.rsi_stage_release_contact_replay import require_terminal


def load_ablation(cfg, *, sealed, digest, pins, original_stage, original_fit):
    spec = cfg["matched_ablation"]
    if type(spec) is not dict or set(spec) != {
        "fit_configuration",
        "fit_configuration_file_hash",
        "fit_unit",
        "predecessor_configuration",
        "predecessor_configuration_file_hash",
        "predecessor_native_unit",
        "predecessor_audit_unit",
        "predecessor_contact_unit",
        "predecessor_readback",
        "predecessor_dense_complete",
        "predecessor_contact_configuration",
        "predecessor_contact_configuration_file_hash",
    }:
        raise ValueError("explicit complete matched-ablation predecessor binding required")
    for key in (
        "fit_unit",
        "predecessor_native_unit",
        "predecessor_audit_unit",
        "predecessor_contact_unit",
    ):
        require_terminal(spec[key])

    def config(key):
        p = Path(spec[key])
        if digest(p) != spec[key + "_file_hash"]:
            raise ValueError("unchanged original ablation/predecessor configuration required")
        pins[str(p)] = digest(p)
        return json.loads(p.read_text())

    fc = config("fit_configuration")
    fr = Path(fc["output_root"])
    stage, fit = (sealed(fr / n) for n in ("complete.json", "numeric-fit.json"))
    model = require_matched_fit(
        stage, fit, configuration=fc, original_stage=original_stage, original_fit=original_fit
    )
    pc = config("predecessor_configuration")
    if "matched_ablation" in pc:
        raise ValueError("original unablated predecessor, not a recursive successor, required")
    root = Path(pc["output_root"])
    declaration, complete = (sealed(root / n) for n in ("declaration.json", "complete.json"))
    audit = sealed(spec["predecessor_readback"])
    # Delayed import avoids the existing exam/diagnostics metadata import edge.
    from rosclaw_soccer.rsi.reference_student_exam_diagnostics import (
        measured_readout,
        require_cohort,
    )

    paired = require_cohort(pc, declaration, complete, audit)
    cc = config("predecessor_contact_configuration")
    if (
        cc["producer_config"] != spec["predecessor_configuration"]
        or cc["readback"] != spec["predecessor_readback"]
        or cc["complete_path"] != spec["predecessor_dense_complete"]
        or audit["reader_source_hash"] != digest(cc["audit_source"])
    ):
        raise ValueError("original whole audit and dense stage source/configuration required")
    dense = sealed(spec["predecessor_dense_complete"])
    replay = sealed(Path(cc["replay_output_root"]) / "complete.json")
    metrics = sealed(cc["metrics_output"])
    measured = measured_readout(declaration, complete, replay, metrics)
    if (
        declaration["models"]["student"] != original_fit["numeric_model"]
        or declaration["fit_stage_hash"] != original_stage["report_hash"]
        or dense.get("upstream_readback_hash") != audit["report_hash"]
        or dense.get("paired_results") != paired
        or dense.get("replay_complete_hash") != replay["report_hash"]
        or dense.get("metrics_hash") != metrics["report_hash"]
        or dense.get("measured_contact_readout") != measured
        or dense.get("new_fixed_control_replay_substeps") != 39000
        or dense.get("new_closed_loop_policy_trials") != 0
        or any(
            dense.get(k) is not False
            for k in (
                "promotion_authorized",
                "hardware_authorized",
                "teacher_qualified",
            )
        )
        or any(
            pc[k] != cfg[k]
            for k in (
                "behavior_model_hash",
                "mesh_task_contact_world_hash",
                "fit_configuration",
                "fit_configuration_file_hash",
            )
        )
    ):
        raise ValueError("whole original same-course unablated exam and dense diagnostics required")
    for owner in (stage, declaration, audit):
        for name, h in owner["input_file_hashes"].items():
            if digest(name) != h:
                raise ValueError("original ablation or predecessor evidence changed")
            pins[name] = h
    return model, stage, fit
