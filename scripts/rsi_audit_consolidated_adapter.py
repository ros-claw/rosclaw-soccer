"""Own CPU histories vs separately qualified frozen features; not physics.

All raw274 actual observations are consumed from private zero memory. Compare
runtime inference with the independent already-qualified own-feature path,
including original reference, bound, features and stateful causal input count.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("training-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("training-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument(
        "--checkpoint-kind",
        choices=("consolidated-adapter-v1", "consolidated-refinement-v1"),
        default="consolidated-adapter-v1",
    )
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    rp = a.training_root / "complete.json"
    if (
        a.output.exists()
        or rp.is_symlink()
        or digest(rp) != a.training_hash
        or (a.training_root / "failed.json").exists()
    ):
        raise ValueError("exact closed training and unused CPU-own output required")
    t = json.loads(rp.read_text())
    pins = dict(t["input_file_hashes"])
    pins[str(rp)] = a.training_hash
    from rosclaw_soccer.rsi.consolidated_adapter_checkpoint import training_data_hash

    refinement = a.checkpoint_kind == "consolidated-refinement-v1"
    data_hash = training_data_hash(t, a.checkpoint_kind)
    bp, bank = receipt(pins, data_hash)
    fp, features = receipt(pins, t["feature_hash"])
    for name in ("checkpoint_path",):
        path = Path(t[name])
        pins[str(path)] = t["checkpoint_hash"]
    pins[str(a.training_root / "declaration.json")] = t["declaration_hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    calls = 0
    try:
        import numpy as np
        import torch

        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        _, model = open_motor(
            pins,
            a.training_hash,
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind=a.checkpoint_kind,
        )
        results = []
        for row, proof in zip(bank["records"], features["records"], strict=True):
            raw_path = bp.parent / row["raw_path"]
            feature_path = fp.parent / proof["feature_path"]
            if (
                raw_path.is_symlink()
                or feature_path.is_symlink()
                or raw_path.parent.resolve() != bp.parent.resolve()
                or feature_path.parent.resolve() != fp.parent.resolve()
                or digest(raw_path) != row["raw_hash"]
                or digest(feature_path) != proof["feature_hash"]
            ):
                raise ValueError(
                    "exact original whole observation and independent own-feature archive required"
                )
            with np.load(raw_path, allow_pickle=False) as z:
                obs, targets, scale = (
                    z["observations"].copy(),
                    z["targets"].copy(),
                    z["action_scale"].copy(),
                )
                reference = (
                    z["reference_nominal"].copy()
                    if refinement
                    else z["reference_nominal"] + z["reference_delta"] / scale
                )
                original_nominal = z["reference_nominal"].copy()
            with np.load(feature_path, allow_pickle=False) as z:
                f = {k: z[k].copy() for k in z.files}
            if len(obs) != row["rows"] or proof["source_raw_hash"] != row["raw_hash"]:
                raise ValueError("all274 matched source histories required")
            private = model.initial_state()
            means = []
            feature_error = mean_error = 0.0
            for i, o in enumerate(obs):
                with torch.no_grad():
                    mean, private, e = model.forward_with_evidence(
                        torch.from_numpy(o).reshape(1, 1, 160), private
                    )
                    feature = torch.from_numpy(f["features"][i]).reshape(1, 1, 288)
                    delta = torch.from_numpy(
                        f["parent_delta" if refinement else "memory_delta"][i]
                    ).reshape(1, 1, 29)
                    if refinement:
                        _, projected_total = model.adapt_features(feature, delta)
                    else:
                        _, projected_total = model.adapt_features(
                            feature, delta, torch.from_numpy(scale)
                        )
                    other = original_nominal[i] + projected_total[0, 0].numpy() / scale
                calls += 1
                feature_error = max(
                    feature_error,
                    float(np.abs(e["critic_features"][0, 0].numpy() - f["features"][i]).max()),
                )
                mean_error = max(mean_error, float(np.abs(mean[0, 0].numpy() - other).max()))
                if (
                    not np.array_equal(
                        e["reference_nominal" if refinement else "reference_mean"][0, 0].numpy(),
                        reference[i],
                    )
                    or (e["final_mean_delta_radians"].abs() > 0.2).any()
                    or feature_error > 2e-5
                    or mean_error > 2e-5
                ):
                    raise ValueError(
                        "own history/reference/head replay or original authority differs"
                    )
                if refinement and not np.array_equal(
                    e["refinement_parent_mean"][0, 0].numpy(),
                    original_nominal[i] + f["parent_delta"][i] / scale,
                ):
                    raise ValueError("whole acquired parent differs from qualified own replay")
                means.append(mean[0, 0].numpy().copy())
            actual = np.stack(means)
            out = a.output / f"prediction{row['index']}.npz"
            np.savez_compressed(out, means=actual)
            result = dict(
                index=row["index"],
                role=row["role"],
                rows=len(obs),
                motor_MSE=float(np.mean(((actual - targets) * scale) ** 2)),
                feature_max_error=feature_error,
                head_mean_max_error=mean_error,
                prediction_path=out.name,
                prediction_hash=digest(out),
            )
            results.append(result)
            print(json.dumps(result), flush=True)
        if calls != features["own_feature_decisions"] or len(results) != (
            62 if refinement else 274
        ):
            raise ValueError("ALL actual histories required before CPU model claim")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS",
                checkpoint_kind=a.checkpoint_kind,
                source_head=a.expected_head,
                source_hash=a.training_hash,
                checkpoint_hash=t["checkpoint_hash"],
                data_hash=data_hash,
                feature_hash=t["feature_hash"],
                input_file_hashes=pins,
                own_actor_decisions=calls,
                separate_feature_head_queries=calls,
                results=results,
                comparison="runtime_private_zero_sequence_vs_independently_qualified_own_features",
                numerical_tolerance=2e-5,
                exact_frozen_reference_required=True,
                physical_steps_executed=0,
                optimizer_reexecuted=False,
                closed_loop_executed=False,
                old_retention_proven=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                own_actor_decisions=calls,
                physical_steps=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
