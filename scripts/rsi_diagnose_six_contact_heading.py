"""Whole closed C2/C3 collision-stage geometry; no integration or fitting."""

import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.initial_contact_heading import (
    initial_contact_heading,
    reconstruct_contact_orientation,
)
from rosclaw_soccer.rsi.foot_heading_diagnostics import heading_contact_readout
from rosclaw_soccer.rsi.original_ik_planner_diagnostics import ik_contact_readout
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_load_heading_cohort import load_closed_heading
from scripts.rsi_load_original_ik_cohort import load_closed_original_ik
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def stage_kind(stage):
    schemas = {
        "soccer.rsi.original_ik_first_touch_stage_pairs_readback.v1": "original-ik-pulse",
        "soccer.rsi.foot_heading_first_touch_stage_pairs_readback.v1": "foot-heading",
    }
    kind = schemas.get(stage.get("schema"))
    if (
        kind is None
        or stage["configuration"].get("stage_profile") != kind
        or stage["configuration"].get("mode") != "SIM_ONLY"
        or stage.get("all_six_receipts_and_failures_retained") is not True
        or any(
            type(stage.get(k)) is not int or stage[k] != n
            for k, n in (
                ("source_native_bodies", 6),
                ("derived_courses", 3),
                ("source_scene_clusters", 3),
                ("new_physical_steps", 0),
                ("new_neural_inferences", 0),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            stage.get(k) is not False
            for k in (
                "teacher_qualified",
                "dataset_admission_authorized",
                "receiving_skill_qualified",
                "successor_readiness_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("whole closed non-promoting six-body stage required")
    return kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-readback", required=True, type=Path)
    parser.add_argument("--stage-file-hash", required=True)
    parser.add_argument("--stage-unit", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    require_terminal(args.stage_unit)
    if (
        args.output.exists()
        or digest(args.stage_readback) != args.stage_file_hash
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused output, exact stage file and clean frozen reader source required")
    pins = {}

    def pin(path, expected=None):
        path = Path(path)
        h = digest(path)
        if (
            path.is_symlink()
            or (expected is not None and h != expected)
            or (str(path) in pins and pins[str(path)] != h)
        ):
            raise ValueError("ordinary unchanged source/geometry evidence required")
        pins[str(path)] = h

    def sealed(path, key="report_hash"):
        pin(path)
        d = json.loads(Path(path).read_text())
        if d[key] != hash_json({k: v for k, v in d.items() if k != key}):
            raise ValueError("original geometry source seal differs")
        return d

    stage = sealed(args.stage_readback)
    kind = stage_kind(stage)
    cfg = stage["configuration"]
    if Path(cfg["output"]).resolve() != args.stage_readback.resolve():
        raise ValueError("original stage-owned output required")
    for name, h in stage["input_file_hashes"].items():
        pin(name, h)
    old = kind == "original-ik-pulse"
    cohort = (load_closed_original_ik if old else load_closed_heading)(
        stage["effective_cohort_spec"], sealed=sealed, pins=pins
    )
    declaration, complete, audit = (cohort[k] for k in ("declaration", "complete", "audit"))
    contact_spec = stage["effective_contact_spec"]
    require_terminal(contact_spec["unit"])
    contact, replay, metrics = (
        sealed(contact_spec[k]) for k in ("complete", "replay_complete", "metrics")
    )
    measured = (ik_contact_readout if old else heading_contact_readout)(
        declaration, complete, replay, metrics
    )
    if (
        stage["native_complete_hash"] != complete["report_hash"]
        or stage["native_audit_hash"] != audit["report_hash"]
        or stage["contact_complete_hash"] != contact["report_hash"]
        or contact["report_hash"] != contact_spec["complete_hash"]
        or contact["replay_complete_hash"] != replay["report_hash"]
        or contact["metrics_hash"] != metrics["report_hash"]
        or contact["measured_contact_readout"] != measured
        or stage["stage_diagnostics"] != measured["assessment"]["stages"]
    ):
        raise ValueError("whole closed native/CPU/dense/stage linkage required")
    for document in (contact, metrics):
        for name, h in document["input_file_hashes"].items():
            pin(name, h)
    replay_path = Path(contact_spec["replay_complete"])
    replay_declaration = sealed(replay_path.parent / "declaration.json")
    world = complete["results"][0]["witnesses"]["world"]
    pin(world["path"], world["sha256"])
    if (
        world["sha256"] != declaration["mesh_task_contact_world_hash"]
        or world["sha256"] != replay_declaration["world_hash"]
        or replay_declaration["mujoco_version"] != mujoco.__version__
    ):
        raise ValueError("original same-version compiled contact world required")
    for name in (
        Path(__file__),
        source / "src/rosclaw_soccer/physics/initial_contact_heading.py",
    ):
        pin(name)
    # Deliberately no mj_forward/mj_step, controller, ONNX or optimizer call.
    model = mujoco.MjModel.from_binary_path(world["path"])
    rows = []
    for job, row in zip(declaration["scope"]["jobs"], replay["rows"], strict=True):
        original = sealed(replay_path.parent / f"{row['case_id']}-complete.json")
        if original != row:
            raise ValueError("all original ordered dense geometry rows required")
        witness = row["arrays"]
        pin(witness["path"], witness["sha256"])
        with np.load(witness["path"], allow_pickle=False) as archive:
            arrays = dict(archive)
        rows.append(
            dict(
                job=job,
                **initial_contact_heading(row, arrays),
                reconstructed_orientation=reconstruct_contact_orientation(model, row, arrays),
            )
        )
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError("source or whole geometry evidence changed during analysis")
    result = dict(
        schema="soccer.rsi.whole_six_collision_heading_readback.v1",
        source_head=args.source_head,
        stage_readback_hash=stage["report_hash"],
        stage_kind=kind,
        native_complete_hash=complete["report_hash"],
        dense_complete_hash=replay["report_hash"],
        rows=rows,
        input_file_hashes=pins,
        all6_including_nofoot_censored_and_failed_cases_retained=True,
        actual_collision_stage_kinematics_not_frame_start_heading=True,
        kinematic_model_allocated=True,
        physical_integration_performed=False,
        causal_failure_explanation_proven=False,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print("WHOLE_SIX_COLLISION_HEADINGS_NOT_NEW_PHYSICS", result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
