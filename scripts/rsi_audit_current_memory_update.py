"""Read-only predeclared state-sample diagnostics, never physical qualification.

Reports how much a learned mean changes raw latents and the pre-slew joint
target on OLD recorded states. It does not infer future trajectories or safety.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.current_memory_motor import (
    CompiledCurrentMemoryMotor,
    make_preview,
    validate_model,
)
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_compressed_bank_storage import capacity_check
from scripts.rsi_train_protected_online_motor_v308 import _head
from scripts.rsi_validate_physical_report_transport import check_declared_model_hash


def numeric_change(before: Any, after: Any, gates: Any) -> dict[str, Any]:
    old, new, gate = [np.asarray(v) for v in (before, after, gates)]
    if (
        old.ndim != 2
        or old.shape[1] != 12
        or not 1 <= len(old) <= 200000
        or new.shape != old.shape
        or gate.shape != (len(old),)
        or not all(np.isfinite(v).all() for v in (old, new, gate))
        or np.any((gate < 0) | (gate > 1))
    ):
        raise ValueError("finite aligned sampled old-state motor means required")
    delta = new - old
    target_delta = 0.25 * (np.tanh(new) - np.tanh(old))
    protected = gate == 0
    return dict(
        states=len(old),
        exact_guard_zero_states=int(protected.sum()),
        raw_latent_delta_rms=float(np.sqrt(np.mean(delta**2))),
        raw_latent_delta_absolute_max=float(np.max(np.abs(delta))),
        pre_slew_target_delta_rms_rad=float(np.sqrt(np.mean(target_delta**2))),
        pre_slew_target_delta_absolute_max_rad=float(np.max(np.abs(target_delta))),
        protected_raw_mean_exact=(
            np.array_equal(new[protected], old[protected]) if np.any(protected) else None
        ),
        physically_applied=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate",
        "initial-model",
        "learning-report",
        "rollout-manifest",
        "rollout-npz",
        "output-root",
        "system-reserve-path",
        "core-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-model-hash", required=True)
    parser.add_argument("--sample-count", type=int, choices=(64, 128), default=128)
    args = parser.parse_args()
    model, initial = (load_json_artifact(p) for p in (args.candidate, args.initial_model))
    validate_model(model)
    validate_model(initial)
    check_declared_model_hash(model["model_hash"], args.expected_model_hash)
    learning, manifest = _sealed(args.learning_report), _sealed(args.rollout_manifest)
    if (
        model["generation"] != 1
        or initial["generation"] != 0
        or model["learning_receipt"]["learner_parent_hash"] != initial["model_hash"]
        or model["baseline"] != initial["baseline"]
        or learning["model_hash"] != model["model_hash"]
        or learning["learning_receipt"] != model["learning_receipt"]
        or manifest["report_hash"] != model["learning_receipt"]["physical_batch_hash"]
        or manifest["parent_model_hash"] != model["baseline"]["base_model"]["model_hash"]
        or manifest["data_hash"] != hash_bytes(args.rollout_npz.read_bytes())
    ):
        raise ValueError(
            "exact declared learned/initial policy and unchanged consumed batch required"
        )
    with np.load(args.rollout_npz, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    n = len(arrays["observation"])
    if n != manifest["frame_sample_count"] or n != 28080:
        raise ValueError("complete sealed 104-rollout bank required")
    selected = np.linspace(0, n - 1, args.sample_count, dtype=np.int64)
    previous = np.maximum(selected - 1, 0)
    indices = np.unique(np.concatenate((selected, previous)))
    source = Path(__file__).resolve().parent.parent
    paths = [
        args.candidate,
        args.initial_model,
        args.learning_report,
        args.rollout_manifest,
        args.rollout_npz,
        Path(__file__),
        source / "src/rosclaw_soccer/rsi/current_memory_motor.py",
    ]
    pins = {str(p.resolve()): hash_bytes(p.read_bytes()) for p in paths}
    commitment = dict(
        schema="soccer.rsi.current_memory_update_sample_commitment.v1",
        inputs=pins,
        model_hash=model["model_hash"],
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        selection="UNIFORM_SEALED_ROW_INDICES_WITH_PREVIOUS_ROW_FOR_CONDITIONAL_MEAN",
        selected_rows=selected.tolist(),
        computed_rows=indices.tolist(),
        capacity=capacity_check(args.output_root.parent, args.system_reserve_path, 1024**2),
        partition="TRAIN_CONSUMED",
        new_physical_executions=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    commitment["report_hash"] = hash_json(commitment)
    args.output_root.mkdir(exist_ok=False)
    write_once(args.output_root / "commitment.json", commitment)
    decoder = CompiledCurrentMemoryMotor(make_preview(model))
    old, new, gates = {}, {}, {}
    for index in indices:
        x, phase = arrays["observation"][index], int(arrays["phase_index"][index])
        old[index] = decoder._parent.raw_mean(x, phase)
        new[index] = decoder.raw_mean(x, phase)
        gates[index] = decoder._guard.gate(np.concatenate((decoder.features(x)[:134], [phase])))
    conditional_kl = []
    for index, prev in zip(selected, previous, strict=True):
        reset = index % 270 == 0
        mean0 = (
            old[index] if reset else old[index] + 0.9 * (arrays["latent_action"][prev] - old[prev])
        )
        mean1 = (
            new[index] if reset else new[index] + 0.9 * (arrays["latent_action"][prev] - new[prev])
        )
        sigma = float(arrays["std_raw"][index]) * (1 if reset else np.sqrt(1 - 0.9**2))
        density = float(
            np.sum(
                -0.5 * ((arrays["latent_action"][index] - mean0) / sigma) ** 2
                - np.log(sigma)
                - 0.5 * np.log(2 * np.pi)
            )
        )
        if not np.isclose(density, arrays["old_log_probability"][index], atol=1e-8, rtol=0):
            raise ValueError(
                "sample is not the independently reconstructed conditional NN behavior"
            )
        conditional_kl.append(float(np.sum((mean1 - mean0) ** 2 / (2 * sigma**2))))
    rows = {}
    for phase in range(3):
        ids = [i for i in selected if arrays["phase_index"][i] == phase]
        rows[str(phase)] = (
            numeric_change(
                np.stack([old[i] for i in ids]),
                np.stack([new[i] for i in ids]),
                np.asarray([gates[i] for i in ids]),
            )
            if ids
            else dict(states=0)
        )
    if any(hash_bytes(Path(p).read_bytes()) != digest for p, digest in pins.items()):
        raise ValueError("model/data/diagnostic source changed during read-only inference")
    result = dict(
        schema="soccer.rsi.current_memory_update_sample_review.v1",
        commitment_hash=commitment["report_hash"],
        model_hash=model["model_hash"],
        selected_states=len(selected),
        computed_states=len(indices),
        sampled_mean_conditional_kl=float(np.mean(conditional_kl)),
        aggregate=numeric_change(
            np.stack([old[i] for i in selected]),
            np.stack([new[i] for i in selected]),
            np.asarray([gates[i] for i in selected]),
        ),
        phases=rows,
        phase_names=["PRE_CONTACT", "FIRST_20_POST_CONTACT_FRAMES", "LATER_RECOVERY"],
        qualification="OLD_STATE_SAMPLE_ONLY_NOT_ACTUAL_JOINT_MOVEMENT_OR_PHYSICAL_GROWTH",
        new_physical_executions=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "review.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
