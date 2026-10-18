"""Read-only same-state reference labels; no dataset export, optimizer or policy trial."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1 import sonic_navigation, sonic_runup
from rosclaw_soccer.rsi import contact_reference_labels
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--study-complete", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.study_complete) != 4:
        raise ValueError("unused output and all four original studies required")
    pins = {}

    def digest(path):
        with Path(path).open("rb") as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()

    def sealed(path):
        value = json.loads(path.read_text())
        if value["report_hash"] != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("input logical seal differs")
        pins[str(path)] = digest(path)
        return value

    inventory = sealed(args.inventory)
    if (
        inventory.get("schema") != "soccer.rsi.contact_demonstration_inventory.v1"
        or inventory.get("whole_original_study_membership_verified") is not True
        or inventory.get("historical_endpoint_count") != 18
        or len(inventory["rows"]) != 18
        or any(
            inventory.get(k) is not False
            for k in (
                "teacher_qualified",
                "dataset_admission_authorized",
                "new_training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError(
            "whole quarantined historical inventory, not a teacher admission, required"
        )
    for path, h in inventory["input_file_hashes"].items():
        if digest(path) != h:
            raise ValueError("original inventory source or witness changed")
        pins[path] = h
    for module in (contact_reference_labels, sonic_navigation, sonic_runup):
        pins[module.__file__] = digest(module.__file__)
    pins[__file__] = digest(__file__)
    expected = {
        "soccer.rsi.mesh_public_draw_complete.v1": 4,
        "soccer.rsi.mesh_contact_catalog_complete.v1": 16,
        "soccer.rsi.mesh_option4_teacher_screen_complete.v1": 18,
        "soccer.rsi.mesh_surface_swing_pilot_complete.v1": 7,
    }
    seen, receipts = set(), {}
    for path in args.study_complete:
        if inventory["input_file_hashes"].get(str(path)) != digest(path):
            raise ValueError("source complete must belong to authenticated original inventory")
        study = sealed(path)
        schema = study.get("schema")
        if schema not in expected or schema in seen or len(study["results"]) != expected[schema]:
            raise ValueError("whole distinct original studies required")
        seen.add(schema)
        receipts.update({r["receipt_hash"]: r for r in study["results"]})
    rows = []
    for item in inventory["rows"]:
        receipt = receipts[item["source_receipt_hash"]]
        witness = receipt["witnesses"]["trace"]
        if (
            digest(witness["path"]) != witness["sha256"]
            or pins.get(witness["path"]) != witness["sha256"]
        ):
            raise ValueError("exact inventory-owned physical trace required")
        with np.load(witness["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        labels = contact_reference_labels.reference_labels(trace)
        delta = labels["pre_parent_reference_increment_rad"]
        rows.append(
            dict(
                case_id=item["case_id"],
                source_receipt_hash=item["source_receipt_hash"],
                original_outcome=item["original_outcome"],
                provisional_success_trace=item["provisional_success_trace"],
                maximum_abs_reference_increment_rad=float(np.max(np.abs(delta))),
                nonzero_reference_frames=int(np.sum(np.any(delta != 0, axis=(1, 2)))),
                actual_float64_navigation_capture_present=(
                    "foundation_navigation_joint_target_rad" in trace
                ),
                actual_taskspace_increment_capture_present=(
                    "taskspace_joint_increment_rad" in trace
                ),
                float32_foundation_vs_actual_navigation_max_difference_rad=float(
                    np.max(
                        np.abs(
                            trace["foundation_neural_joint_target_mujoco_rad"]
                            - labels["navigation_joint_target_rad"]
                        )
                    )
                ),
            )
        )
    result = dict(
        schema="soccer.rsi.contact_reference_label_reconstruction.v1",
        inventory_hash=inventory["report_hash"],
        input_file_hashes=pins,
        rows=rows,
        historical_endpoints=18,
        independent_courses=6,
        recorded_frames_reconstructed=5400,
        action_dimensions=12,
        action_kind="pre-parent leg joint reference increment, not direct torque",
        same_state_and_same_tick=True,
        counterfactual_trajectories_subtracted=False,
        captured_float64_navigation_targets_reconstructed_bit_exactly=12,
        original_none_targets_derived_from_exact_raw_action_commit_formula=6,
        future_outcomes_or_teacher_family_are_actor_inputs=False,
        training_dataset_exported=False,
        local_reproducibility_qualified=False,
        teacher_qualified=False,
        new_optimizer_updates=0,
        new_neural_inferences=0,
        new_physical_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if any(digest(path) != h for path, h in pins.items()):
        raise ValueError("input bytes changed during readback")
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
