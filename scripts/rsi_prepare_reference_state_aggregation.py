"""Proposed own-state aggregation only; no fitting, physics or data admission."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.pre_parent_reference_features import FEATURE_NAMES, trace_observations
from rosclaw_soccer.rsi.reference_state_aggregation import (
    LABEL_CEILING,
    prepare_proposed_aggregation,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_behavior_fit(
    profile, producer, declaration, *, original_stage, original_fit, sealed, pins
):
    """New-generation inputs require their own authenticated fitted behaviour.

    Original teacher rows/splits still come from original_fit. This changes
    only the actually visited student-body distribution, not teacher labels.
    """
    if profile == "proposed-original16-own13-aggregation-v1":
        expected = original_fit
        model, stage = original_fit["numeric_model"], original_stage
    elif profile == "proposed-original16-ablation-own13-aggregation-v1":
        from scripts.rsi_load_reference_ablation_exam import load_ablation

        model, stage, expected = load_ablation(
            producer,
            sealed=sealed,
            digest=digest,
            pins=pins,
            original_stage=original_stage,
            original_fit=original_fit,
        )
    else:
        raise ValueError("explicit original or input-ablation own-state generation required")
    if (
        declaration["fit_report_hash"] != expected["report_hash"]
        or declaration["models"]["student"] != model
        or declaration["fit_stage_hash"] != stage["report_hash"]
    ):
        raise ValueError("own-body behaviour fit identity must match its actual generation")


def original_actor_rows(trace, *, captured_required):
    """Never substitute an expert reference for the behaviour policy's input."""
    value = trace_observations(trace)
    if captured_required:
        # The runtime stores projected increment before floating target
        # addition. Recovering it by target-NAV can differ by roundoff.
        captured = trace.get("reference_student_input_features")
        other = [
            i
            for i, name in enumerate(FEATURE_NAMES)
            if not name.startswith("previous_applied_reference_increment:")
        ]
        if (
            type(captured) is not np.ndarray
            or captured.shape != (300, 1, 135)
            or captured.dtype != np.float64
            or not np.isfinite(captured).all()
            or captured[:, 0, other].tobytes() != value["observation"][:, other].tobytes()
        ):
            raise ValueError(
                "actual captured student actor inputs and unchanged measured state required"
            )
        value["observation"] = captured[:, 0].copy()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        cfg.get("profile")
        not in (
            "proposed-original16-own13-aggregation-v1",
            "proposed-original16-ablation-own13-aggregation-v1",
        )
        or cfg.get("mode") != "SIM_ONLY"
        or root.exists()
        or time.time() >= cfg["deadline_unix"]
        or cfg.get("acknowledged_label_ceiling") != LABEL_CEILING
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "dataset_admission_authorized",
                "neural_training_authorized",
                "private_fresh_accessed",
            )
        )
        or any(
            type(cfg.get(k)) is not int or cfg[k] != 0
            for k in (
                "new_physical_steps",
                "optimizer_updates",
            )
        )
        or Path(__file__).resolve() != source / "scripts/rsi_prepare_reference_state_aggregation.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError(
            "fixed unused proposed aggregation, explicit ceiling and zero fitting required"
        )
    for unit in (cfg["own_query_unit"], cfg["original_fit_unit"]):
        require_terminal(unit)
    pins = {str(args.config): digest(args.config)}
    for project in (source, Path(cfg["core_root"])):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path):
        path = Path(path)
        v = json.loads(path.read_text())
        if v.get("report_hash") != hash_json({k: x for k, x in v.items() if k != "report_hash"}):
            raise ValueError("original sealed complete evidence required")
        pins[str(path)] = digest(path)
        return v

    own = sealed(cfg["own_query_complete"])
    stage = sealed(cfg["original_fit_complete"])
    fit = sealed(Path(cfg["original_fit_complete"]).parent / "numeric-fit.json")
    if (
        own["report_hash"] != cfg["own_query_complete_hash"]
        or stage["report_hash"] != cfg["original_fit_complete_hash"]
        or fit["report_hash"] != stage["numeric_fit_hash"]
        or own["source_body_episodes"] != 13
        or len(own["rows"]) != 13
        or own["same_actual_body_and_navigation_only"] is not True
        or any(
            own[k] is not False
            for k in (
                "dagger_training_performed",
                "teacher_qualified",
                "dataset_admission_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(r.get("eligible_for_training") is not False for r in own["rows"])
    ):
        raise ValueError(
            "original explicitly unadmitted own-body proposals and original fit required"
        )
    for owner in (stage, own):
        for name, h in owner["input_file_hashes"].items():
            if digest(name) != h:
                raise ValueError("original evidence or source changed")
            pins[name] = h
    pc = json.loads(Path(stage["configuration"]["positive_configuration"]).read_text())
    original = sealed(Path(pc["output_root"]) / "complete.json")
    cc = json.loads(Path(own["configuration"]["producer_contact_configuration"]).read_text())
    producer = json.loads(Path(cc["producer_config"]).read_text())
    sroot = Path(producer["output_root"])
    declaration, student = (sealed(sroot / n) for n in ("declaration.json", "complete.json"))
    require_behavior_fit(
        cfg["profile"],
        producer,
        declaration,
        original_stage=stage,
        original_fit=fit,
        sealed=sealed,
        pins=pins,
    )
    if (
        original["report_hash"] != stage["positive_complete_hash"]
        or [r["receipt_hash"] for r in original["results"]] != stage["source_receipt_hashes"]
        or student["report_hash"] != own["source_complete_hash"]
        or student["declaration_hash"] != declaration["report_hash"]
        or [r["receipt_hash"] for r in student["results"]]
        != [r["source_receipt_hash"] for r in own["rows"]]
        or [r["job"] for r in own["rows"]] != declaration["scope"]["jobs"]
    ):
        raise ValueError(
            "all original source rows and exact actual own-state query memberships required"
        )

    def observed(receipts):
        rows = []
        for r in receipts:
            w = r["witnesses"]["trace"]
            if pins.get(w["path"]) != w["sha256"]:
                raise ValueError("original authenticated recorded actor inputs required")
            with np.load(w["path"], allow_pickle=False) as a:
                trace = dict(a)
            rows.append(
                original_actor_rows(
                    trace, captured_required=r.get("reference_student_model_hash") is not None
                )
            )
        return rows

    old_rows, own_rows = observed(original["results"]), observed(student["results"])
    old_x = np.stack([r["observation"] for r in old_rows])
    old_y = np.stack([r["same_state_reference_target_rad"] for r in old_rows])
    if (
        hash_bytes(old_x.tobytes()) != fit["whole_observations_hash"]
        or hash_bytes(old_y.tobytes()) != fit["whole_reference_labels_hash"]
        or [m["eligible"] for m in fit["final_same_state_diagnostics"]] != stage["eligibility"]
    ):
        raise ValueError(
            "exact whole original fitting observations, labels and eligibility required"
        )
    w = own["proposed_arrays"]
    if digest(w["path"]) != w["sha256"]:
        raise ValueError("actual original queried reference proposals required")
    pins[w["path"]] = w["sha256"]
    with np.load(w["path"], allow_pickle=False) as a:
        if set(a.files) != {f"queried_reference_rad_{i:02d}" for i in range(13)}:
            raise ValueError("whole thirteen queried proposal arrays required")
        y = np.stack([a[f"queried_reference_rad_{i:02d}"] for i in range(13)])
    if any(
        hash_bytes(a.tobytes()) != r["query"]["proposed_labels_hash"]
        for a, r in zip(y, own["rows"], strict=True)
    ):
        raise ValueError("each exact same-body query label identity required")
    batch, proposed = prepare_proposed_aggregation(
        old_x,
        old_y,
        np.stack([r["observation"] for r in own_rows]),
        y,
        original_receipts=original["results"],
        original_metadata=fit["final_same_state_diagnostics"],
        own_jobs=declaration["scope"]["jobs"],
        own_receipt_hashes=[r["receipt_hash"] for r in student["results"]],
        query_receipt_hashes=[r["source_receipt_hash"] for r in own["rows"]],
        parent_model_hash=pc["behavior_model_hash"],
        acknowledged_label_ceiling=LABEL_CEILING,
    )
    if args.preflight:
        print("PROPOSED_29_BODY_AGGREGATION_READY_ZERO_OPTIMIZER_OR_PHYSICS", flush=True)
        return
    root.mkdir()
    write_once(root / "started.json", dict(configuration=cfg, automatic_retry_authorized=False))
    try:
        path = root / "proposed-numeric-inputs-not-admitted.npz"
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                actual_observations=batch["original_observations"],
                reference_proposals=batch["targets"],
                partitions=np.asarray(batch["partitions"]),
                source_seeds=np.asarray(batch["source_seeds"]),
                experimental_supervision=np.asarray(batch["eligible"]),
            )
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("source/evidence changed during proposed aggregation")
        report = dict(
            schema="soccer.rsi.proposed_reference_aggregation_prepared.v1",
            configuration=cfg,
            input_file_hashes=pins,
            proposed_batch=proposed,
            source_fit_complete_hash=stage["report_hash"],
            source_own_query_complete_hash=own["report_hash"],
            actual_behavior_fit_report_hash=declaration["fit_report_hash"],
            actual_behavior_model_hash=declaration["models"]["student"]["model_hash"],
            bounded_current_own_state_window_not_cumulative_dagger=True,
            numeric_arrays=dict(path=str(path), sha256=digest(path)),
            original_query_eligibility_overwritten=False,
            experimental_numeric_selection_only=True,
            dataset_admission_authorized=False,
            teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
            optimizer_updates=0,
            new_physical_steps=0,
        )
        report["report_hash"] = hash_json(report)
        write_once(root / "complete.json", report)
        print(
            "PROPOSED_AGGREGATION_PREPARED_NOT_ADMITTED_OR_TRAINED",
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
