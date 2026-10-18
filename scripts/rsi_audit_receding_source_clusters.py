"""Additive metadata clarification, never rewrite running frozen study evidence."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

from rosclaw_soccer.rsi.public_course_cluster_accounting import cluster_contract
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts
from rosclaw_soccer.rsi.receding_teacher_scope import receding_contexts, scope_contract
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--behavior-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if (
        args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused additive output and clean frozen metadata reader required")

    def digest(path):
        with Path(path).open("rb") as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()

    declaration = json.loads(args.declaration.read_text())
    if (
        declaration.get("schema") != "soccer.rsi.receding_catalog2_declaration.v1"
        or declaration["report_hash"]
        != hash_json({k: v for k, v in declaration.items() if k != "report_hash"})
        or declaration["prepared_pins"].get(str(args.behavior_source))
        != digest(args.behavior_source)
    ):
        raise ValueError(
            "sealed original receding declaration and pinned behavior location source required"
        )
    # Read a literal only: do not execute the old collector or import its policy.
    assignments = [
        n.value
        for n in ast.parse(args.behavior_source.read_text()).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MEAN" for t in n.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("one original literal mean identity required")
    behavior_hash = ast.literal_eval(assignments[0])
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), behavior_hash, 18)
    if declaration["scope_contract"] != scope_contract(plan):
        raise ValueError("whole original scope must reconstruct exactly; no count rewriting")
    pins = {
        str(args.declaration): digest(args.declaration),
        **declaration["prepared_pins"],
        **declaration["input_file_hashes"],
    }
    for folder in (source / "src", source / "scripts"):
        pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    if any(digest(path) != h for path, h in pins.items()):
        raise ValueError("original source or prepared metadata bytes changed")
    contexts = receding_contexts(plan)
    union = {c.context_hash: c for c in (*planning_contexts(plan)[2:], *contexts)}
    result = dict(
        schema="soccer.rsi.receding_source_cluster_clarification.v1",
        original_declaration_hash=declaration["report_hash"],
        original_frozen_independent_courses_field=declaration["scope_contract"][
            "independent_courses"
        ],
        correct_interpretation=(
            "6 distinct perturbed courses, 3 source-scene clusters; no independence proof"
        ),
        receding_scope=cluster_contract(plan, contexts),
        expected_postprocessing_course_union_metadata=cluster_contract(plan, tuple(union.values())),
        whole_declaration_includes_already_known_historical_outcomes=True,
        outcome_labels_used_for_cluster_counts=False,
        original_documents_unchanged=True,
        execution_budgets_unchanged=True,
        runtime_and_exam_unchanged=True,
        teacher_qualified=False,
        statistical_independence_proven=False,
        uncertainty_must_use_source_clusters=True,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
        input_file_hashes=pins,
        reader_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print("SIX_DERIVED_COURSES_THREE_SOURCE_CLUSTERS", result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
