"""Measure exploration/actor mismatch without changing a policy or its data.

Large successful exploration draws are NOT proof that those exact actions are
necessary. This diagnostic cannot establish causality, fresh gain or promotion.
It uses the complete previously audited bank, including failures.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth.anchor_kernel import AnchorKernelGuard

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.output_memory_step_motor import (
    CompiledOutputMemoryMotor,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def transfer_statistics(
    noise: Any, shift: Any, gates: Any, std: Any, *, residual_cap: float
) -> dict[str, Any]:
    noise, shift, gates, std = [np.asarray(v, dtype=np.float64) for v in (noise, shift, gates, std)]
    n = len(noise)
    if (
        n == 0
        or noise.shape != (n, 12)
        or shift.shape != noise.shape
        or gates.shape != (n,)
        or std.shape != (n,)
        or not all(np.isfinite(v).all() for v in (noise, shift, gates, std))
        or np.any((gates < 0) | (gates > 1))
        or np.any(std <= 0)
        or type(residual_cap) not in (float, int)
        or not np.isfinite(residual_cap)
        or not 0 < residual_cap <= 1
    ):
        raise ValueError("finite aligned bounded actor diagnostics required")
    envelope = residual_cap * gates[:, None]
    if np.any(np.abs(shift) > envelope + 1e-8):
        raise ValueError("observed actor shift violates its declared residual envelope")
    return dict(
        frame_count=n,
        exploration_noise_rms=float(np.sqrt(np.mean(noise**2))),
        learned_mean_shift_rms=float(np.sqrt(np.mean(shift**2))),
        exact_mean_latent_kl=float(np.mean(np.sum(shift**2 / (2 * std[:, None] ** 2), axis=1))),
        gate_quantiles=np.quantile(gates, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        fraction_noise_components_outside_residual_envelope=float(
            np.mean(np.abs(noise) > envelope + 1e-8)
        ),
        residual_envelope_rms=float(np.sqrt(np.mean(envelope**2))),
        noise_shift_inner_product_mean=float(np.mean(np.sum(noise * shift, axis=1))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("learning-root", "behavior-model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest = _sealed(args.learning_root / "rollout_manifest.json")
    parent = json.loads(args.behavior_model.read_text())
    candidate = json.loads((args.learning_root / "model.json").read_text())
    validate_model(parent)
    validate_model(candidate)
    data_path = args.learning_root / "rollouts.npz"
    if (
        parent["generation"] != 0
        or candidate["generation"] != 1
        or manifest["schema"] != "soccer.rsi.output_memory_on_policy_bank.v1"
        or manifest["partition"] != "TRAIN_CONSUMED"
        or manifest["parent_model_hash"] != parent["model_hash"]
        or candidate["learning_receipt"]["learner_parent_hash"] != parent["model_hash"]
        or candidate["learning_receipt"]["physical_batch_hash"] != manifest["report_hash"]
        or candidate["frozen_parent"] != parent["frozen_parent"]
        or candidate["output_memory"] != parent["output_memory"]
        or manifest["data_hash"] != hash_bytes(data_path.read_bytes())
        or any(
            obj.get(k) is not False
            for obj in (manifest, parent, candidate)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("complete sealed zero-to-first-generation learning bank required")
    with np.load(data_path, allow_pickle=False) as data:
        x, phase, action, old_logp, returns, std, groups = [
            data[k]
            for k in (
                "observation",
                "phase_index",
                "latent_action",
                "old_log_probability",
                "terminal_return",
                "std_raw",
                "trajectory_index",
            )
        ]
    records = manifest["records"]
    count = len(records)
    n = count * 270
    if (
        count == 0
        or manifest["physical_rollout_count"] != count
        or manifest["frame_sample_count"] != n
        or [r["group"] for r in records] != list(range(count))
        or x.shape != (n, 134)
        or action.shape != (n, 12)
        or any(a.shape != (n,) for a in (phase, old_logp, returns, std, groups))
        or phase.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or not np.array_equal(groups, np.repeat(np.arange(count), 270))
        or not all(np.isfinite(v).all() for v in (x, phase, action, old_logp, returns, std, groups))
        or np.any((std < 0.01) | (std > 0.15))
        or set(phase.tolist()) != {0, 1, 2}
    ):
        raise ValueError("complete ordered physical trajectories required")
    baseline = CompiledOutputMemoryMotor(make_preview(parent))
    learned = CompiledOutputMemoryMotor(make_preview(candidate))
    means = []
    contexts = []
    shift_rows = []
    for v, p in zip(x, phase, strict=True):
        mean = baseline.raw_mean(v, int(p))
        means.append(mean)
        shift_rows.append(learned.raw_mean(v, int(p)) - mean)
        contexts.append(baseline.context(v, int(p)))
    noise = action - np.asarray(means)
    density = np.sum(
        -0.5 * (noise / std[:, None]) ** 2 - np.log(std[:, None]) - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(density, old_logp, atol=1e-8, rtol=0):
        raise ValueError("data density differs from the exact behavior actor")
    gates = AnchorKernelGuard(
        parent["output_memory"]["observations"], bandwidth=parent["output_memory"]["bandwidth"]
    ).gates(np.asarray(contexts))
    shifts = np.asarray(shift_rows)
    rows = []
    for r in records:
        ids = groups == r["group"]
        if not np.all(returns[ids] == terminal_return(r["outcome"])):
            raise ValueError("learning returns do not match the audited physical outcome")
        rows.append(
            dict(
                group=r["group"],
                seed=r["seed"],
                lane=r["lane"],
                sample=r["sample"],
                high_quality=r["outcome"]["high_quality"],
                terminal_return=float(returns[ids][0]),
                statistics=transfer_statistics(
                    noise[ids],
                    shifts[ids],
                    gates[ids],
                    std[ids],
                    residual_cap=parent["raw_residual_cap"],
                ),
            )
        )
    total = transfer_statistics(noise, shifts, gates, std, residual_cap=parent["raw_residual_cap"])
    if not np.isclose(
        total["exact_mean_latent_kl"],
        candidate["learning_receipt"]["exact_mean_latent_kl"],
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("diagnostic inference KL differs from the sealed training receipt")
    report = dict(
        schema="soccer.rsi.memory_learning_transfer_diagnostic.v1",
        manifest_hash=manifest["report_hash"],
        data_hash=manifest["data_hash"],
        behavior_model_hash=parent["model_hash"],
        candidate_model_hash=candidate["model_hash"],
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        statistics=total,
        rows=rows,
        qualification="OBSERVED_ACTION_RANGE_ONLY_NOT_CAUSAL_NOT_FRESH_GAIN",
        exact_exploration_reproduction_is_not_necessary_for_success=True,
        physical_executions_added=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(json.dumps(dict(report_hash=report["report_hash"], statistics=total)), flush=True)


if __name__ == "__main__":
    main()
