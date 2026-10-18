"""Bind full native V2 SIM receipts to on-policy torque learning arrays.

Reads every declared success/failure and authenticates existing replay receipts.
Numeric policy/latent reconstruction is repeated here; physics is not rerun.
Only a newly declared research driver may use the resulting batch for training.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rosclaw_soccer.rsi import cpu_motor_transfer_evidence as cpu_module
from rosclaw_soccer.rsi import foundation_torch_capture_audit as foundation_module
from rosclaw_soccer.rsi.cerebellar_recovery_return import reward_contract, training_return
from rosclaw_soccer.rsi.cerebellar_torque_audit import reconstruct_native_cerebellar
from rosclaw_soccer.rsi.cerebellar_torque_policy import TIME_AWARE_SCHEMA, validate_model
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

if TYPE_CHECKING:
    from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def episode_arrays(
    actor: dict[str, Any],
    receipt: dict[str, Any],
    report: dict[str, Any],
    commitment: dict[str, Any],
    trace: dict[str, Any],
    review: dict[str, Any],
    foundation: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Numeric/label consistency ONLY. Caller must authenticate all documents."""
    validate_model(actor)
    if (
        actor["schema"] != TIME_AWARE_SCHEMA
        or receipt.get("cerebellar_model_hash") != actor["model_hash"]
        or receipt.get("cerebellar_sampling_draws") != 3000
        or receipt.get("cerebellar_inference_calls") != 3000
        or receipt.get("cerebellar_model_optimizer_steps") != actor["optimizer_steps"]
        or any(
            receipt.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "teacher_qualified",
                "private_fresh_accessed",
            )
        )
    ):
        raise ValueError(
            "complete sampled current V2 behavior receipt required, including failures"
        )
    if (
        report.get("report_hash") != receipt["physical_report_hash"]
        or review.get("reviewed_report_hash") != report["report_hash"]
        or review.get("source_hash") != hash_bytes(Path(cpu_module.__file__).read_bytes())
        or review.get("actual_mujoco_dynamics_replayed") is not True
        or review.get("actual_pd_torque_reconstructed") is not True
        or foundation.get("audit_source_hash")
        != hash_bytes(Path(foundation_module.__file__).read_bytes())
        or foundation.get("physical_report_hash") != report["report_hash"]
        or foundation.get("physical_trace_hash") != report.get("physical_trace_hash")
        or report.get("compiled_model_hash") != actor["world_hash"]
        or commitment.get("compiled_model_hash") != actor["world_hash"]
        or commitment.get("cerebellar_torque_contract", {}).get("model") != actor
    ):
        raise ValueError("source-bound physics/Foundation replay and exact behavior model required")
    outcome = label_first_touch(trace)
    alignment = require_first_touch_qualification(
        review,
        foundation,
        outcome,
        cerebellar_model_hash=actor["model_hash"],
        cerebellar_sampling_draws=3000,
    )
    if outcome.to_dict() != receipt["outcome"] or alignment != receipt["qualification_alignment"]:
        raise ValueError("all primitive success/failure labels must agree with independent replay")
    numeric = reconstruct_native_cerebellar(report, commitment, trace)
    if numeric is None or numeric["sampling_draws"] != 3000:
        raise ValueError("full current-body sampled policy reconstruction required")
    objective = training_return(trace)
    arrays = dict(
        features=trace["cerebellar_input_features"].reshape(3000, 127).copy(),
        latent_action=trace["cerebellar_sampling_latent_action"].reshape(3000, 29).copy(),
        old_log_probability=trace["cerebellar_sampling_log_probability"].reshape(3000).copy(),
        terminal_return=np.full(3000, objective["training_return"], dtype=np.float64),
    )
    return arrays, objective


def build_batch(
    receipt_paths: list[Path],
    *,
    actor: dict[str, Any],
    expected_jobs: list[dict[str, Any]],
    declaration_hash: str,
    evidence_root: Path,
    physical_payload_reader: "FixedPhysicalPayloadReader",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read all predeclared research episodes; no best-success or seed filter."""
    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact

    validate_model(actor)
    if actor["schema"] != TIME_AWARE_SCHEMA:
        raise ValueError("new V2 collection required; never relabel V1 physics")
    if (
        not 2 <= len(receipt_paths) <= 8
        or len(receipt_paths) != len(expected_jobs)
        or len(set(receipt_paths)) != len(receipt_paths)
        or len({job["job"] for job in expected_jobs}) != len(expected_jobs)
        or len({job["sampling_seed"] for job in expected_jobs}) != len(expected_jobs)
    ):
        raise ValueError("complete unique declared research collection required")
    root = evidence_root.resolve()
    chunks, rows, pins = [], [], {}

    def own(path: Path) -> Path:
        if (
            not path.is_absolute()
            or not path.resolve().is_relative_to(root)
            or path.is_symlink()
            or any(parent.is_symlink() for parent in path.parents)
        ):
            raise ValueError("ordinary experiment-owned input files required")
        return path

    declaration_path = own(root / "declaration.json")
    declaration = _sealed(declaration_path)
    declared_jobs = [
        job
        for job in declaration["jobs"]
        if job.get("kind") == "collection" and job.get("actor_seed") == actor["seed"]
    ]
    if (
        declaration["report_hash"] != declaration_hash
        or declared_jobs != expected_jobs
        or declaration.get("mode") != "SIM_ONLY"
        or declaration.get("neural_training_authorized") is not True
        or any(
            declaration.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "private_fresh_accessed")
        )
    ):
        raise ValueError(
            "entire predeclared actor collection and explicit SIM training scope required"
        )
    pins[str(declaration_path)] = _digest(declaration_path)

    for path, job in zip(receipt_paths, expected_jobs, strict=True):
        receipt = load_json_artifact(own(path))
        if receipt.get("receipt_hash") != hash_json(
            {k: v for k, v in receipt.items() if k != "receipt_hash"}
        ):
            raise ValueError("sealed actual receipt required")
        if (
            receipt.get("job") != job["job"]
            or receipt.get("group") != job["group"]
            or receipt.get("planning_context_hash") != job["context_hash"]
        ):
            raise ValueError("all predeclared jobs required in original order")
        if receipt.get("cerebellar_model_hash") != actor["model_hash"]:
            raise ValueError("actual on-policy behavior model required before loading payload")
        pins[str(path)] = _digest(path)
        witnesses = dict(receipt["witnesses"])
        witnesses.update(
            {key: receipt[key] for key in ("intent_witness", "encoder_witness", "model_witness")}
        )
        if witnesses["world"]["sha256"] != actor["world_hash"]:
            raise ValueError("compiled physical world file must match behavior model binding")
        for witness in witnesses.values():
            file = own(Path(witness["path"]))
            if _digest(file) != witness["sha256"]:
                raise ValueError("whole original witness file changed")
            pins[str(file)] = witness["sha256"]
        intent = _sealed(Path(receipt["intent_witness"]["path"]))
        if (
            any(intent.get(k) != v for k, v in job.items())
            or intent.get("declaration_hash") != declaration_hash
            or intent.get("model_hash") != actor["model_hash"]
        ):
            raise ValueError("pre-physical intent and behavior binding required")
        witness = receipt["witnesses"]
        report = _sealed(
            Path(witness["report"]["path"]), physical_payload_reader=physical_payload_reader
        )
        if report["physical_trace_hash"] != witness["trace"]["sha256"]:
            raise ValueError("actual report must bind the full raw physical trace")
        commitment = load_json_artifact(Path(witness["commitment"]["path"]))
        # Native commitment is hashed by the sealed physical report, not self-sealed.
        if report.get("commitment_hash") != hash_json(commitment):
            raise ValueError("actual full execution commitment seal differs")
        with np.load(witness["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        encoder = _sealed(Path(receipt["encoder_witness"]["path"]))
        if (
            encoder.get("compared_encoder_frames") != 300
            or encoder.get("compared_features_per_frame") != 640
            or encoder.get("actual_cpu_onnx_planner_calls") != 15
            or encoder.get("recorded_features_hash")
            != hash_bytes(np.asarray(trace["foundation_neural_encoder_features"]).tobytes())
            or encoder.get("comparison_atol") != 1e-5
            or not 0 <= encoder.get("maximum_absolute_error", float("inf")) <= 1e-5
        ):
            raise ValueError("complete independent planner-to-encoder consumption required")
        review = _sealed(Path(witness["independent_review"]["path"]))
        foundation = _sealed(Path(witness["foundation_review"]["path"]))
        if (
            review["report_hash"] != receipt["independent_review_hash"]
            or foundation["report_hash"] != receipt["foundation_review_hash"]
        ):
            raise ValueError("receipt must bind the exact independent replay reviews")
        chunk, objective = episode_arrays(
            actor,
            receipt,
            report,
            commitment,
            trace,
            review,
            foundation,
        )
        if (
            commitment["cerebellar_torque_contract"]["sampling"]["sampling_seed"]
            != job["sampling_seed"]
        ):
            raise ValueError("actual sampling seed differs from predeclared draw")
        chunks.append(chunk)
        rows.append(dict(job=job, receipt_hash=receipt["receipt_hash"], objective=objective))
        del report, trace
    for name, expected in pins.items():
        if _digest(Path(name)) != expected:
            raise ValueError("training source data changed during admission")
    arrays: dict[str, Any] = {
        key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]
    }
    arrays.update(
        behavior_model_hash=actor["model_hash"],
        reward_contract_hash=reward_contract()["contract_hash"],
        trajectory_index=np.repeat(np.arange(len(chunks), dtype=np.int64), 3000),
        terminated=np.ones(len(chunks), dtype=np.bool_),
        truncated=np.zeros(len(chunks), dtype=np.bool_),
    )
    manifest = dict(
        schema="soccer.rsi.cerebellar_native_training_batch.v1",
        declaration_hash=declaration_hash,
        behavior_model_hash=actor["model_hash"],
        reward_contract_hash=arrays["reward_contract_hash"],
        episodes=rows,
        input_file_hashes=pins,
        whole_declared_collection_included=True,
        new_physical_replays=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
    )
    return arrays, dict(manifest, report_hash=hash_json(manifest))
