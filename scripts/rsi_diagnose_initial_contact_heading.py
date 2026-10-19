"""All16 previously authenticated first-touch directions, no simulation or fitting."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.initial_contact_heading import (
    initial_contact_heading,
    reconstruct_contact_orientation,
)
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-declaration", required=True, type=Path)
    parser.add_argument("--anchor-file-hash", required=True)
    parser.add_argument("--replay-complete", required=True, type=Path)
    parser.add_argument("--metrics-readback", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kinematic-proposals", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or digest(args.anchor_declaration) != args.anchor_file_hash:
        raise ValueError("unused output and exact input anchor declaration required")
    pins = {str(args.anchor_declaration): args.anchor_file_hash}

    def sealed(path):
        value = json.loads(Path(path).read_text())
        if value.get("report_hash") != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("whole original source logical seal required")
        pins[str(path)] = digest(path)
        return value

    anchor = sealed(args.anchor_declaration)
    original_pins = {**anchor["input_file_hashes"], **anchor["prepared_pins"]}
    if original_pins.get(str(args.replay_complete)) != digest(args.replay_complete):
        raise ValueError("whole replay must already be authenticated by input declaration")
    replay = sealed(args.replay_complete)
    if original_pins.get(str(args.metrics_readback)) != digest(args.metrics_readback):
        raise ValueError("original dense metrics readback must be source authenticated")
    metrics = sealed(args.metrics_readback)
    if (
        metrics.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1"
        or len(metrics["rows"]) != 16
        or any(
            type(metrics.get(k)) is not int or metrics[k] != 0
            for k in ("new_physical_steps", "new_neural_inferences", "new_optimizer_updates")
        )
        or any(
            m["source_receipt_hash"] != r["source_receipt_hash"]
            or m["replay_result_hash"] != r["report_hash"]
            or m["case_id"] != r["case_id"]
            for m, r in zip(metrics["rows"], replay["rows"], strict=True)
        )
    ):
        raise ValueError("whole original dense replay-to-metrics linkage required")
    declaration_path = args.replay_complete.parent / "declaration.json"
    if metrics["input_file_hashes"].get(str(declaration_path)) != digest(declaration_path):
        raise ValueError("original replay physics declaration must be source authenticated")
    replay_declaration = sealed(declaration_path)
    world = next(iter(anchor["reused_teachers"].values()))["witnesses"]["world"]
    if (
        world["sha256"] != replay_declaration["world_hash"]
        or world["sha256"] != anchor["mesh_task_contact_world_hash"]
        or digest(world["path"]) != world["sha256"]
        or replay_declaration["mujoco_version"] != mujoco.__version__
    ):
        raise ValueError("original same-version compiled world required for collision-stage FK")
    pins[world["path"]] = world["sha256"]
    model = mujoco.MjModel.from_binary_path(world["path"])
    expected = [f"{role}_positive_local_c{i}" for i in range(8) for role in ("control", "teacher")]
    if (
        replay.get("schema") != "soccer.rsi.fixed_control_contact_replay_complete.v1"
        or replay.get("integrated_substeps") != 48000
        or [r["case_id"] for r in replay["rows"]] != expected
        or len({r["source_receipt_hash"] for r in replay["rows"]}) != 16
    ):
        raise ValueError("all sixteen original ordered teacher/control cases required")
    rows = []
    for r in replay["rows"]:
        original = sealed(args.replay_complete.parent / f"{r['case_id']}-complete.json")
        if original != r:
            raise ValueError("whole original dense row membership required")
        w = r["arrays"]
        if digest(w["path"]) != w["sha256"]:
            raise ValueError("original dense array file bytes changed")
        pins[w["path"]] = w["sha256"]
        with np.load(w["path"], allow_pickle=False) as archive:
            arrays = dict(archive)
            row = dict(
                initial_contact_heading(r, arrays),
                reconstructed_orientation=reconstruct_contact_orientation(model, r, arrays),
            )
            if args.kinematic_proposals:
                from rosclaw_soccer.rsi.foot_heading_nullspace import check_recorded_pose_proposal

                event = r["first_foot_contact_event"]
                row["kinematic_proposal"] = (
                    None
                    if event is None
                    else check_recorded_pose_proposal(
                        model,
                        arrays["canonical_qpos_before_substep"][event["substep_index"]],
                        event["other_body_name"],
                        enabled=True,
                    )
                )
            rows.append(row)
    source = Path(__file__).resolve().parents[1]
    for p in (Path(__file__), source / "src/rosclaw_soccer/physics/initial_contact_heading.py"):
        pins[str(p)] = digest(p)
    if args.kinematic_proposals:
        p = source / "src/rosclaw_soccer/rsi/foot_heading_nullspace.py"
        pins[str(p)] = digest(p)
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError("source or input changed during no-physics direction analysis")
    result = dict(
        schema="soccer.rsi.original_sixteen_foot_heading_proposals.v1"
        if args.kinematic_proposals
        else "soccer.rsi.original_sixteen_initial_contact_directions.v1",
        source_anchor_hash=anchor["report_hash"],
        replay_complete_hash=replay["report_hash"],
        original_metrics_hash=metrics["report_hash"],
        rows=rows,
        input_file_hashes=pins,
        all16_failures_and_nofoot_cases_retained=True,
        current_source_ancestry_authenticated_here=False,
        kinematic_model_allocated=True,
        physical_integration_performed=False,
        kinematic_proposals_only=args.kinematic_proposals,
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
    print("ALL16_INITIAL_CONTACT_DIRECTIONS_NOT_NEW_PHYSICS", result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
