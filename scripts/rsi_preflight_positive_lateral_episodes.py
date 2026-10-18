"""Read-only prospective eight-course body-input validation; zero NN/physics."""

import argparse
import json
import subprocess
from pathlib import Path

from rosclaw_soccer.rsi.positive_lateral_episode_inputs import episode_inputs, prospective_split
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-hash", required=True)
    parser.add_argument("--behavior-model-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if (
        args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused output and clean frozen source required")
    registry = json.loads(args.registry.read_text())
    pins = {str(args.registry): digest(args.registry)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    plan, manifest, contexts = episode_inputs(
        registry,
        expected_registry_hash=args.registry_hash,
        behavior_model_hash=args.behavior_model_hash,
        digest=digest,
        pins=pins,
    )
    if any(digest(Path(p)) != h for p, h in pins.items()):
        raise ValueError("registry or source bytes changed while validating")
    value = dict(
        schema="soccer.rsi.positive_lateral_body_input_preflight.v1",
        registry_hash=registry["report_hash"],
        source_plan_hash=plan.contract()["plan_hash"],
        source_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        prospective_split=prospective_split(registry["scope"]),
        physical_source_groups=[c.source_group for c in contexts],
        metadata_registry_rows=len(manifest["records"]),
        source_scene_clusters=4,
        distinct_derived_course_count=8,
        input_file_hashes=pins,
        full_numerical_parent_reconstruction_reused_from_original_registry=True,
        new_full_parent_reconstruction=False,
        new_nn_allocations=0,
        new_physical_steps=0,
        new_optimizer_updates=0,
        execution_authorized=False,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        fresh_exam=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    value["report_hash"] = hash_json(value)
    write_once(args.output, value)
    print(
        "EIGHT_EPISODE_INPUTS_FOUR_CLUSTERS_PREDECLARED_SPLIT_ZERO_NN_PHYSICS", value["report_hash"]
    )


if __name__ == "__main__":
    main()
