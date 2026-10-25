"""Three fixed supervised predictor seeds; no motor Actor updates or physics."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--bank-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    bp = a.bank / "complete.json"
    if a.output.exists() or bp.is_symlink() or digest(bp) != a.bank_hash:
        raise ValueError("new training output and exact reconstructed data bank required")
    bank = json.loads(bp.read_text())
    if bank["status"] != "CAUSAL_KICK_ENTRY_BANK_RECONSTRUCTED_FIRST_ACTIONS_EXACT_NOT_TRAINED" or (
        bank["total_original_records"] != 1224
        or bank["independent_initial_states"] != 56
        or bank["promotion_authorized"] is not False
    ):
        raise ValueError("complete causal data bank with all initial-state groups required")
    path = a.bank / bank["artifact"]["path"]
    if (
        path.parent.resolve() != a.bank.resolve()
        or path.is_symlink()
        or digest(path) != bank["artifact"]["hash"]
    ):
        raise ValueError("exact confined training arrays required")
    pins = dict(bank["input_file_hashes"])
    pins.update({str(bp): a.bank_hash, str(path): bank["artifact"]["hash"]})
    source_pins(pins)
    check_pins(pins)
    import numpy as np
    import torch
    import torch.nn.functional as F

    from rosclaw_soccer.rsi.kick_outcome_predictor import (
        group_weights,
        grouped_metrics,
        grouped_partition,
        make_predictor,
    )
    from rosclaw_soccer.sim.contracts import hash_json

    torch.set_num_threads(1)
    with np.load(path) as saved:
        x, y, groups, strata = (saved[k] for k in ("features", "labels", "groups", "strata"))
    if (
        x.shape != (bank["unique_training_rows"], 549)
        or y.shape != (len(x), 3)
        or (
            x.dtype != np.float32
            or y.dtype != np.float32
            or not np.isfinite(x).all()
            or not np.isin(y, [0, 1]).all()
            or (y[:, 2] > y[:, :2].min(axis=1)).any()
        )
    ):
        raise ValueError("finite causal features and logically consistent physical labels required")
    split = grouped_partition(groups, strata)
    train = np.isin(groups, split["train_groups"])
    validation = np.isin(groups, split["validation_groups"])
    weights = group_weights(groups[train])
    mean64 = (weights[:, None] * x[train]).sum(axis=0)
    var64 = (weights[:, None] * (x[train] - mean64) ** 2).sum(axis=0)
    mean, scale = mean64.astype(np.float32), np.maximum(np.sqrt(var64), 0.1).astype(np.float32)
    train_ids = split["train_groups"]
    index_groups = [np.flatnonzero(groups == g) for g in train_ids]
    train_group_rates = np.stack([y[index].mean(axis=0) for index in index_groups])
    global_rate = train_group_rates.mean(axis=0)
    class_rates = np.stack(
        [
            np.stack([y[idx].mean(axis=0) for idx in index_groups if strata[idx[0]] == k]).mean(
                axis=0
            )
            for k in range(4)
        ]
    )
    baseline = dict(
        global_frequency=grouped_metrics(
            np.broadcast_to(global_rate, (validation.sum(), 3)), y[validation], groups[validation]
        ),
        class_frequency=grouped_metrics(
            class_rates[strata[validation]], y[validation], groups[validation]
        ),
    )
    a.output.mkdir(parents=True)
    context_hash = hash_json(
        dict(
            bank_hash=a.bank_hash,
            body_hash=bank["body_hash"],
            policy_model_hash=bank["policy_model_hash"],
            encoder="causal547+referenceclock+remainingtask.v1",
        )
    )
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.g1.supervised_entry_outcome_prediction.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            context_hash=context_hash,
            split=split,
            seeds=[926071, 926072, 926073],
            updates_per_seed=512,
            batch_size=64,
            optimizer="Adam",
            learning_rate=0.0003,
            weight_decay=0.0001,
            gradient_clip_norm=1.0,
            hidden=[128, 64],
            normalization="TRAIN group-weighted mean/std, .1 floor",
            sampling="uniform TRAIN initial group, then uniform proposal within that group",
            loss="safeBCE+footBCE+conditional_qualityBCE(mask=safe_and_foot)",
            pass_probability="sigmoid(safe)*sigmoid(foot)*sigmoid(conditional_quality)",
            no_validation_gradient=True,
            no_best_checkpoint_or_seed_selection=True,
            review_only_criteria=(
                "all3: task Brier better than class baseline, safe Brier no worse; "
                "safe>=.9 coverage>=.1 and error<=.05"
            ),
            validation_initial_groups=16,
            validation_is_public_consumed_not_sealed_fresh=True,
            reinforcement_learning=False,
            actor_updates=0,
            physical_steps_added=0,
            prediction_is_motion_authorization=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    results = []
    try:
        for seed in (926071, 926072, 926073):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed + 1000)
            model = make_predictor(mean, scale)
            initial = {k: v.detach().clone() for k, v in model.named_parameters()}
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
            losses, empty = [], 0
            for _update in range(512):
                indices = np.array(
                    [rng.choice(index_groups[i]) for i in rng.integers(0, len(index_groups), 64)]
                )
                xb, yb = torch.from_numpy(x[indices]), torch.from_numpy(y[indices])
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                mask = yb[:, 0] * yb[:, 1]
                empty += int(not bool(mask.sum()))
                conditional = (
                    F.binary_cross_entropy_with_logits(logits[:, 2], yb[:, 2], reduction="none")
                    * mask
                ).sum() / mask.sum().clamp_min(1)
                loss = (
                    F.binary_cross_entropy_with_logits(logits[:, :2], yb[:, :2]) * 2 + conditional
                )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss; no repair or skipped update")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite gradient; no repair or skipped update")
                optimizer.step()
                losses.append(float(loss.detach()))
            steps = [int(state["step"]) for state in optimizer.state.values()]
            if len(steps) != 6 or set(steps) != {512}:
                raise ValueError("all six parameter tensors must have512 actual Adam updates")
            changed = sum(
                int((parameter.detach() != initial[name]).sum())
                for name, parameter in model.named_parameters()
            )
            if not changed:
                raise ValueError("training changed no network parameters")
            checkpoint = a.output / f"outcome-{seed}.pt"
            torch.save(
                dict(
                    schema="soccer.g1.kick_outcome_prediction_checkpoint.v1",
                    context_hash=context_hash,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),
                    numpy_rng=rng.bit_generator.state,
                    actual_optimizer_updates=512,
                    actor_updates=0,
                    seed=seed,
                    split=split,
                    bank_hash=a.bank_hash,
                    motor_authority=False,
                ),
                checkpoint,
            )
            # Validate portable safe loading; no custom objects or unsafe pickle.
            restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if restored["context_hash"] != context_hash:
                raise ValueError("checkpoint context changed")
            model.eval()
            with torch.no_grad():
                prediction = model.probabilities(torch.from_numpy(x[validation])).numpy()
            metrics = grouped_metrics(prediction, y[validation], groups[validation])
            base = baseline["class_frequency"]["group_weighted_brier"]
            values = metrics["group_weighted_brier"]
            eligible = (
                values[2] < base[2]
                and values[0] <= base[0]
                and metrics["safe_high_confidence_coverage"] >= 0.1
                and metrics["safe_high_confidence_error"] is not None
                and metrics["safe_high_confidence_error"] <= 0.05
            )
            prediction_path = a.output / f"validation-{seed}.npz"
            np.savez_compressed(
                prediction_path,
                prediction=prediction,
                targets=y[validation],
                groups=groups[validation],
                strata=strata[validation],
            )
            row = dict(
                seed=seed,
                actual_optimizer_updates=512,
                actor_updates=0,
                changed_parameter_elements=changed,
                empty_conditional_batches=empty,
                first_loss=losses[0],
                last_loss=losses[-1],
                losses=losses,
                validation=metrics,
                eligible_for_physical_ranking_review=eligible,
                checkpoint=dict(path=checkpoint.name, hash=digest(checkpoint)),
                predictions=dict(path=prediction_path.name, hash=digest(prediction_path)),
            )
            results.append(row)
            write_once(a.output / f"record-{seed}.json", row)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="THREE_SUPERVISED_OUTCOME_PREDICTORS_TRAINED_NOT_MOTOR_POLICIES",
                source_head=a.expected_head,
                input_file_hashes=pins,
                context_hash=context_hash,
                bank_hash=a.bank_hash,
                split=split,
                baseline=baseline,
                results=results,
                actual_predictor_optimizer_updates=1536,
                actor_updates=0,
                all_seeds_eligible_for_physical_ranking_review=all(
                    r["eligible_for_physical_ranking_review"] for r in results
                ),
                physical_steps_added=0,
                physical_football_gain_proven=False,
                reinforcement_learning=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                completed_seeds=len(results),
                no_automatic_retry=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
