"""Prepare new public local-course metadata/views; zero physics or NN allocation."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from rosclaw_soccer.rsi import positive_lateral_receding_scope as curriculum
from rosclaw_soccer.rsi import positive_lateral_sampling_registry as registry_module
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-declaration", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    root = args.output_root
    if (
        root.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("new unused registry output and frozen clean source required")

    def digest(path):
        with Path(path).open("rb") as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()

    upstream = json.loads(args.upstream_declaration.read_text())
    if (
        upstream["report_hash"]
        != hash_json({k: v for k, v in upstream.items() if k != "report_hash"})
        or upstream.get("schema") != "soccer.rsi.receding_catalog2_declaration.v1"
    ):
        raise ValueError("sealed original receding declaration required")
    pins = {
        **upstream["prepared_pins"],
        **upstream["input_file_hashes"],
        str(args.upstream_declaration): digest(args.upstream_declaration),
    }
    if pins.get(str(args.source_manifest)) != digest(args.source_manifest):
        raise ValueError("source sampling manifest must belong to original pinned preparation")
    original = json.loads(args.source_manifest.read_text())
    if original["report_hash"] != hash_json(
        {k: v for k, v in original.items() if k != "report_hash"}
    ):
        raise ValueError("original complete source manifest seal required")
    view_witness = original["records"][0]["witnesses"]["sampling_view"]
    payload_witness = upstream["reused_endpoints"][0]["receipt"]["witnesses"]["proof_payload"]
    for witness in (view_witness, payload_witness):
        if (
            pins.get(witness["path"]) != witness["sha256"]
            or digest(witness["path"]) != witness["sha256"]
        ):
            raise ValueError("whole original source view and mean payload must be byte-pinned")
    for folder in (source / "src", source / "scripts"):
        pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError("original input or source bytes changed")
    envelope = load_json_artifact(Path(view_witness["path"]))
    mean = load_json_artifact(Path(payload_witness["path"]))
    scope = curriculum.build_scope(mean["model_hash"])
    views = registry_module.registry_views(scope, envelope, mean)
    root.mkdir()
    write_once(root / "scope.json", scope)
    write_once(
        root / "started.json",
        dict(
            input_file_hashes=pins,
            automatic_retry_authorized=False,
            no_nn_or_physics_allocated=True,
        ),
    )
    try:
        (root / "models").mkdir()
        (root / ".shared-models").mkdir()
        target_payload = root / ".shared-models" / Path(payload_witness["path"]).name
        shutil.copyfile(payload_witness["path"], target_payload)
        if digest(target_payload) != payload_witness["sha256"]:
            raise ValueError("local whole mean copy differs")
        records = []
        for draw in scope["source_plan"]["jobs"]:
            group = draw["group"]
            witnesses = {}
            if group in views:
                path = root / "models" / f"source-{group:04d}.json.gz"
                write_once(path, views[group])
                witnesses["sampling_view"] = dict(path=str(path), sha256=digest(path))
            records.append(
                dict(draw=draw, witnesses=witnesses, metadata_only_no_physical_trajectory=True)
            )
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("input/source changed during materialization")
        result = dict(
            schema="soccer.rsi.positive_lateral_sampling_registry.v1",
            upstream_declaration_hash=upstream["report_hash"],
            scope_hash=scope["scope_hash"],
            scope=scope,
            records=records,
            input_file_hashes=pins,
            shared_mean=dict(
                path=str(target_payload),
                sha256=digest(target_payload),
                payload_hash=envelope["payload_hash"],
                model_hash=mean["model_hash"],
            ),
            declared_physical_source_groups=list(views),
            source_view_count=8,
            metadata_registry_rows=len(records),
            source_registry_fillers_are_not_physical_trials=True,
            proposed_new_native_bodies=16,
            proposed_independent_cpu_bodies=16,
            proposed_combined_body_substeps=96000,
            new_nn_model_allocations=0,
            new_neural_inferences=0,
            new_physical_steps=0,
            new_optimizer_updates=0,
            all_old_receding_failures_retained_not_relabelled=True,
            execution_authorized=False,
            dataset_admission_authorized=False,
            teacher_qualified=False,
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
            source_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip(),
        )
        result["report_hash"] = hash_json(result)
        write_once(root / "registry.json", result)
        print(
            "FOUR_SOURCE_CLUSTERS_EIGHT_VIEWS_ZERO_NN_ZERO_PHYSICS",
            result["report_hash"],
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
