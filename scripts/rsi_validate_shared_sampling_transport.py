"""Two actual runs prove lossless shared sampling inputs AND physical reports.

Use the fixed first sample of the sealed 104-rollout current-parent curriculum.
No new noise, training, fresh exam, or activation; retain all original evidence.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import rosclaw.growth.shared_proof_payload as shared_module

from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.physical_report_io import resolve_physical_report
from rosclaw_soccer.rsi.sampling_model_io import load_sampling_model
from rosclaw_soccer.rsi.smooth_memory_motor import make_preview
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once, write_shared_sampling_model
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_compressed_bank_storage import capacity_check
from scripts.rsi_prepare_smooth_memory_round_two import folder_bytes
from scripts.rsi_retest_current_memory_counterexample import check_trace_arrays
from scripts.rsi_train_protected_online_motor_v308 import _head
from scripts.rsi_transport_equivalence import compare_transport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-root",
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "system-reserve-path",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-mean-model-hash", required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), default=0)
    parser.add_argument("--fast-shared-sampling-build", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    for root in (source, args.core_root):
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
            raise ValueError("immutable clean execution source required")
    if (
        Path(shared_module.__file__).resolve()
        != (args.core_root / "src/rosclaw/growth/shared_proof_payload.py").resolve()
    ):
        raise ValueError("declared imported Core transport required")
    reference = _sealed(args.reference_root / "training_summary.json")
    if (
        reference["schema"] != "soccer.rsi.smooth_memory_failure_exploration.v1"
        or reference["independent_contexts"] != 13
        or reference["exploration_executions"] != 104
        or reference["physical_executions"] != 130
        or reference["commitment"]["partition"] != "TRAIN_CONSUMED"
    ):
        raise ValueError("complete fixed original 104-rollout physical reference required")
    row = reference["rows"][0]
    if (row["seed"], row["lane"]) != (20261378, 0):
        raise ValueError("fixed first reference sample required")
    old_path = args.reference_root / "models/sample-0.json.gz"
    view = load_sampling_model(old_path)
    make_preview(view)
    if (
        view["model_hash"] != row["samples"][0]["view_hash"]
        or view["mean_model"]["model_hash"] != args.expected_mean_model_hash
        or any(
            reference.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("exact preregistered physical behavior/sample identity required")
    stem = "seed20261378-lane0"
    old_parent_folder = args.reference_root / f"{stem}-reproduction-parent"
    old_actor_folder = args.reference_root / f"{stem}-sample-0-actor"
    old_parent, old_actor = (
        _sealed(p / "report.json") for p in (old_parent_folder, old_actor_folder)
    )
    if (
        old_parent["report_hash"] != row["parent_report_hash"]
        or old_actor["report_hash"] != row["samples"][0]["report_hash"]
        or old_actor["contact_motor_policy"]["step_motor_proof"]["model"] != view
    ):
        raise ValueError("historical reference seals and complete policy required")
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    files = [
        old_path,
        args.reference_root / "training_summary.json",
        runner,
        Path(__file__),
        args.g1_usd,
        args.late_swing_policy,
        Path(shared_module.__file__),
        source / "src/rosclaw_soccer/rsi/sampling_model_io.py",
        source / "scripts/rsi_atomic_artifacts.py",
        source / "scripts/rsi_collect_approach_lateral_tracking_v286.py",
        resolve_physical_report(old_parent_folder / "report.json"),
        resolve_physical_report(old_actor_folder / "report.json"),
    ]
    if args.fast_shared_sampling_build:
        import rosclaw.growth.frozen_payload_field as cached_module

        if (
            Path(cached_module.__file__).resolve()
            != (args.core_root / "src/rosclaw/growth/frozen_payload_field.py").resolve()
        ):
            raise ValueError("actual cached field must belong to declared Core")
        files.extend(
            [
                Path(cached_module.__file__),
                source / "scripts/rsi_exact_shared_sampling_views.py",
            ]
        )
    pins = {str(p.resolve()): hash_bytes(p.read_bytes()) for p in files}
    capacity = capacity_check(args.output_root.parent, args.system_reserve_path, 768 * 1024**2)
    commitment = dict(
        schema="soccer.rsi.shared_sampling_transport_commitment.v1",
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        runner_hash=hash_bytes(runner.read_bytes()),
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        input_hashes=pins,
        reference_summary_hash=reference["report_hash"],
        reference_parent_report_hash=old_parent["report_hash"],
        reference_actor_report_hash=old_actor["report_hash"],
        model_hash=view["model_hash"],
        mean_model_hash=args.expected_mean_model_hash,
        seed=20261378,
        lane=0,
        physical_executions_planned=2,
        capacity=capacity,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if args.fast_shared_sampling_build:
        commitment["sampling_construction"] = "EXACT_CACHED_COMPLETE_MEAN_V1"
    args.output_root.mkdir(exist_ok=False)
    (args.output_root / "models").mkdir()
    (args.output_root / "logs").mkdir()
    write_once(args.output_root / "commitment.json", commitment)
    path = args.output_root / "models/sample-0.json.gz"
    if args.fast_shared_sampling_build:
        from scripts.rsi_exact_shared_sampling_views import exact_shared_views, publish_shared_views

        cached_views = exact_shared_views(view["mean_model"], [view["seed"]])
        publish_shared_views(args.output_root, view["mean_model"], cached_views)
    else:
        write_shared_sampling_model(path, view)
    if load_sampling_model(path) != view:
        raise ValueError("whole sampling input transport changed")
    common = dict(
        root=args.output_root,
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        seed=20261378,
        lane=0,
        gpu=args.gpu,
        gain=1.2,
        negative_only=True,
        core_root=args.core_root,
        compressed_report=True,
        shared_model_report=True,
        execution_timeout_s=900,
    )
    parent, _ = _run(**common, arm="reproduction", kind="parent")
    parent_folder = args.output_root / f"{stem}-reproduction-parent"
    actor, _ = _run(
        **common,
        arm="sample-0",
        kind="actor",
        motor_step=path,
        parent_report_override=resolve_physical_report(parent_folder / "report.json"),
    )
    actor_folder = args.output_root / f"{stem}-sample-0-actor"
    for new_folder, old_folder in (
        (parent_folder, old_parent_folder),
        (actor_folder, old_actor_folder),
    ):
        names: tuple[str, ...] = ("body_trace.npz", "trace.npz")
        if new_folder == actor_folder:
            names += ("contact_motor_trace.npz", "late_swing_action_trace.npz")
        for name in names:
            with (
                np.load(old_folder / name, allow_pickle=False) as old,
                np.load(new_folder / name, allow_pickle=False) as new,
            ):
                check_trace_arrays(dict(old), dict(new))
    measured = _outcome(actor_folder, actor["contact_motor_policy_hash"], commitment)["outcome"]
    historical = _outcome(
        old_actor_folder,
        old_actor["contact_motor_policy_hash"],
        {"runner_hash": old_actor["source_hash"], "asset_hash": old_actor["asset_hash"]},
    )["outcome"]
    if actor["frames"] != 300 or old_actor["frames"] != 300:
        raise ValueError("both complete 300-frame motor traces required")
    if any(row["samples"][0][k] != v for k, v in historical.items()):
        raise ValueError("all historical physical labels must reconstruct exactly")
    equivalence = compare_transport(actor, old_actor, measured, historical)
    if any(hash_bytes(Path(p).read_bytes()) != h for p, h in pins.items()) or (
        _head(source) != commitment["source_commit"]
        or _head(args.core_root) != commitment["core_commit"]
        or load_sampling_model(path) != view
    ):
        raise ValueError("transport input/source changed during actual comparison")
    store = args.output_root / ".shared-models"
    payloads = list(store.iterdir())
    if len(payloads) != 1:
        raise ValueError("input and physical proof must share ONE WHOLE identical mean")
    report = dict(
        schema="soccer.rsi.shared_sampling_transport_review.v1",
        commitment_hash=hash_json(commitment),
        input_hashes=pins,
        mean_model_hash=args.expected_mean_model_hash,
        model_hash=view["model_hash"],
        complete_sampling_payload_equal=True,
        all_physical_trace_arrays_equal=True,
        physical_outcome_comparison=equivalence,
        measured_outcome=measured,
        new_physical_executions=2,
        motor_frames_reconstructed=600,
        largest_execution_bytes=max(folder_bytes(p) for p in (parent_folder, actor_folder)),
        largest_native_log_bytes=max(
            p.stat().st_size for p in (args.output_root / "logs").iterdir()
        ),
        shared_whole_model_bytes=payloads[0].stat().st_size,
        sampling_envelope_bytes=path.stat().st_size,
        qualification="LOSSLESS_INPUT_AND_REPORT_TRANSPORT_NOT_LEARNING_OR_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output_root / "independent_review.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
