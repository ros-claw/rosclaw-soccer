"""Three actual executions: same-source parent, NN control, declared new actor.

The NN must reproduce ALL four historical physical/action trace arrays before
the new actor runs. This fixed consumed counterexample is not a fresh exam.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.current_memory_motor import validate_model
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.physical_report_io import resolve_physical_report
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_failed_step_courses import qualified_memory_failure_rows
from scripts.rsi_compressed_bank_storage import capacity_check
from scripts.rsi_retest_risk_margin_out_counterexample import COURSE, risk_gate
from scripts.rsi_train_protected_online_motor_v308 import _head
from scripts.rsi_validate_physical_report_transport import check_declared_model_hash

CORE_FILES = ("anchor_kernel.py", "anchor_output_memory.py", "correlated_exploration.py")
MOTOR_FILES = (
    "compiled_step_inference.py",
    "contact_motor_phase.py",
    "contact_motor_primitive.py",
    "step_motor_phase_context.py",
    "stochastic_step_execution.py",
    "kernel_guarded_step_execution.py",
    "output_memory_step_motor.py",
    "smooth_memory_motor.py",
)
TRACES = ("body_trace.npz", "contact_motor_trace.npz", "late_swing_action_trace.npz", "trace.npz")


def check_trace_arrays(old: dict[str, Any], new: dict[str, Any]) -> None:
    if old.keys() != new.keys() or any(
        old[k].dtype != new[k].dtype or not np.array_equal(old[k], new[k]) for k in old
    ):
        raise ValueError(
            "new runner must reproduce every unchanged-NN physical/action array exactly"
        )


def check_execution_parent(report: dict[str, Any], *, runner_hash: str, asset_hash: str) -> None:
    if report.get("source_hash") != runner_hash or report.get("asset_hash") != asset_hash:
        raise ValueError("actor must use a same-source same-body newly executed parent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-root",
        "reference-source",
        "reference-core",
        "qualified-parent-bank",
        "qualified-parent-model",
        "candidate",
        "learning-report",
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "system-reserve-path",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-model-hash", required=True)
    parser.add_argument("--expected-learning-hash", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), default=3)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    model = load_json_artifact(args.candidate)
    advantage = model.get("schema") == "soccer.rsi.advantage_memory_motor.v1"
    if advantage:
        from rosclaw_soccer.rsi.advantage_memory_motor import validate_model as advantage_validate

        advantage_validate(model)
    else:
        validate_model(model)
    learning = _sealed(args.learning_report)
    check_declared_model_hash(model["model_hash"], args.expected_model_hash)
    check_declared_model_hash(learning["report_hash"], args.expected_learning_hash)
    qualified_nn = load_json_artifact(args.qualified_parent_model)
    baseline_model = model["initial_actor"]["baseline"] if advantage else model["baseline"]
    learning_schema = (
        "soccer.rsi.advantage_memory_gradient_learning.v1"
        if advantage
        else "soccer.rsi.current_memory_gradient_learning.v1"
    )
    if (
        learning["schema"] != learning_schema
        or learning["model_hash"] != model["model_hash"]
        or learning["learning_receipt"] != model["learning_receipt"]
        or learning["new_physical_executions"] != 0
        or baseline_model["base_model"]["frozen_parent"] != qualified_nn
        or any(
            learning.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError(
            "exact declared completed current-memory learner and qualified parent required"
        )
    reference = load_json_artifact(args.reference_root / "commitment.json")
    bank, review = (
        _sealed(args.qualified_parent_bank / n)
        for n in ("validation_summary.json", "independent_review.json")
    )
    qualified_memory_failure_rows(bank, review)
    old_runner = args.reference_source / "scripts/rsi_isaac_vector_first_touch.py"
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    if (
        bank["report_hash"] != reference["parent_bank_hash"]
        or bank["commitment"]["model_hash"] != qualified_nn["model_hash"]
        or reference["source_commit"] != _head(args.reference_source)
        or reference["runner_hash"] != hash_bytes(old_runner.read_bytes())
        or reference["core_commit"] != _head(args.reference_core)
        or reference["asset_hash"] != hash_bytes(args.g1_usd.read_bytes())
    ):
        raise ValueError("historical controls must retain exact original source/body bindings")
    unchanged = {}
    for old_root, new_root, relative, files in (
        (args.reference_core, args.core_root, "src/rosclaw/growth", CORE_FILES),
        (args.reference_source, source, "src/rosclaw_soccer/rsi", MOTOR_FILES),
    ):
        for name in files:
            old, new = old_root / relative / name, new_root / relative / name
            digest = hash_bytes(old.read_bytes())
            if digest != hash_bytes(new.read_bytes()):
                raise ValueError("historical NN/motor numerical modules must stay byte-identical")
            unchanged[str(new.resolve())] = digest
    seed, lane = COURSE
    stem = f"seed{seed}-lane{lane}"
    parent_path = args.reference_root / f"{stem}-reproduction-parent/report.json"
    parent = _sealed(parent_path)
    warm_folder = args.reference_root / f"{stem}-warm-actor"
    warm = _sealed(warm_folder / "report.json")
    if (
        (warm["training_course_seed"], warm["single_course_lane"]) != COURSE
        or warm["parent_report_hash"] != parent["report_hash"]
        or warm["contact_motor_policy"]["step_motor_proof"]["model"] != qualified_nn
    ):
        raise ValueError("exact unchanged qualified NN physical control required")
    baseline = _outcome(warm_folder, warm["contact_motor_policy_hash"], reference)["outcome"]
    inputs = [
        args.candidate,
        args.learning_report,
        args.qualified_parent_model,
        args.reference_root / "commitment.json",
        args.qualified_parent_bank / "validation_summary.json",
        args.qualified_parent_bank / "independent_review.json",
        runner,
        args.g1_usd,
        args.late_swing_policy,
        Path(__file__),
        source / "src/rosclaw_soccer/rsi/current_memory_motor.py",
        args.core_root / "src/rosclaw/growth/correlated_residual_gradient.py",
    ]
    if advantage:
        inputs += [
            source / "src/rosclaw_soccer/rsi/advantage_memory_motor.py",
            source / "src/rosclaw_soccer/rsi/advantage_memory_learning.py",
            args.core_root / "src/rosclaw/growth/bounded_advantage_regression.py",
        ]
    pins = {str(p.resolve()): hash_bytes(p.read_bytes()) for p in inputs}
    pins.update(unchanged)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    commitment = dict(
        schema="soccer.rsi.current_memory_counterexample_commitment.v1",
        course=list(COURSE),
        runner_hash=hash_bytes(runner.read_bytes()),
        asset_hash=reference["asset_hash"],
        execution_source_commit=_head(source),
        core_commit=_head(args.core_root),
        historical_execution_source_commit=reference["source_commit"],
        historical_core_commit=reference["core_commit"],
        model_hash=model["model_hash"],
        learning_report_hash=learning["report_hash"],
        input_hashes=pins,
        capacity=capacity_check(args.output_root.parent, args.system_reserve_path, 768 * 1024**2),
        parent_report_hash=parent["report_hash"],
        qualified_nn_report_hash=warm["report_hash"],
        new_physical_executions_planned=3,
        physical_gpu=args.gpu,
        historical_controls_consumed=2,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    args.output_root.mkdir(exist_ok=False)
    (args.output_root / "logs").mkdir()
    write_once(args.output_root / "commitment.json", commitment)
    new_parent, _ = _run(
        root=args.output_root,
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        seed=seed,
        lane=lane,
        gpu=args.gpu,
        gain=1.2,
        negative_only=True,
        core_root=args.core_root,
        arm="reproduction",
        kind="parent",
        compressed_report=True,
        execution_timeout_s=900,
    )
    new_parent_folder = args.output_root / f"{stem}-reproduction-parent"
    check_execution_parent(
        new_parent, runner_hash=commitment["runner_hash"], asset_hash=commitment["asset_hash"]
    )
    for name in ("body_trace.npz", "trace.npz"):
        with (
            np.load(parent_path.parent / name, allow_pickle=False) as old,
            np.load(new_parent_folder / name, allow_pickle=False) as new,
        ):
            check_trace_arrays(dict(old), dict(new))
    if any(
        new_parent[k] != parent[k]
        for k in ("asset_hash", "sonic_qualification_hash", "environments")
    ):
        raise ValueError("same-source parent must reproduce historical physical course exactly")
    new_parent_path = resolve_physical_report(new_parent_folder / "report.json")
    print("CURRENT_MEMORY_PARENT_CONTROL_EXACT", flush=True)
    for arm, actor_path in (
        ("nn-control", args.qualified_parent_model),
        ("current-memory", args.candidate),
    ):
        raw, _ = _run(
            root=args.output_root,
            runner=runner,
            isaac_python=args.isaac_python,
            g1_usd=args.g1_usd,
            model_root=args.model_root,
            actor=args.late_swing_policy,
            seed=seed,
            lane=lane,
            gpu=args.gpu,
            gain=1.2,
            negative_only=True,
            core_root=args.core_root,
            arm=arm,
            kind="actor",
            motor_step=actor_path,
            parent_report_override=new_parent_path,
            compressed_report=True,
            execution_timeout_s=900,
        )
        folder = args.output_root / f"{stem}-{arm}-actor"
        outcome = _outcome(folder, raw["contact_motor_policy_hash"], commitment)["outcome"]
        if arm == "nn-control":
            for name in TRACES:
                with (
                    np.load(warm_folder / name, allow_pickle=False) as old,
                    np.load(folder / name, allow_pickle=False) as new,
                ):
                    check_trace_arrays(dict(old), dict(new))
            # Audit hashes include the raw report's runner source identity.
            # Keep both receipts; do not demand identical identity after a
            # declared dispatch-only runner revision. ALL measured fields and
            # ALL physical/action arrays must still be exactly equal.
            if {k: v for k, v in outcome.items() if k != "command_audit_hash"} != {
                k: v for k, v in baseline.items() if k != "command_audit_hash"
            }:
                raise ValueError("unchanged NN control outcome diverged before candidate execution")
            new_nn_outcome = outcome
            print("CURRENT_MEMORY_NN_CONTROL_EXACT", flush=True)
        else:
            candidate_outcome = outcome
    if (
        any(hash_bytes(Path(p).read_bytes()) != digest for p, digest in pins.items())
        or _head(source) != commitment["execution_source_commit"]
        or _head(args.core_root) != commitment["core_commit"]
        or _sealed(parent_path) != parent
        or _sealed(warm_folder / "report.json") != warm
        or raw["parent_report_hash"] != new_parent["report_hash"]
        or raw["contact_motor_policy"]["step_motor_proof"]["model"]["model_hash"]
        != model["model_hash"]
    ):
        raise ValueError("physical control/model/source drift")
    result = dict(
        schema="soccer.rsi.current_memory_counterexample_review.v1",
        commitment_hash=hash_json(commitment),
        model_hash=model["model_hash"],
        course=list(COURSE),
        new_physical_executions=3,
        historical_controls_consumed=2,
        motor_frames_reconstructed=900,
        exact_unchanged_nn_trace_equivalence=True,
        exact_same_source_parent_trace_equivalence=True,
        same_source_parent_report_hash=new_parent["report_hash"],
        new_nn_control_outcome=new_nn_outcome,
        reference_outcome=baseline,
        candidate_outcome=candidate_outcome,
        **risk_gate(baseline, candidate_outcome),
        qualification="FIXED_COUNTEREXAMPLE_ONLY_NOT_FULL_BANK_OR_FRESH",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "independent_review.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
