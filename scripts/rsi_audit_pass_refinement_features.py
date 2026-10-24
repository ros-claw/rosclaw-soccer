"""Own private-state feature replay and zero-refinement identity, NOT new physics."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("bank-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("bank-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_output_capacity import require_output_capacity

    path = a.bank_root / "complete.json"
    if (
        a.output.exists()
        or a.bank_root.is_symlink()
        or path.is_symlink()
        or digest(path) != a.bank_hash
        or (a.bank_root / "failed.json").exists()
    ):
        raise ValueError("exact whole qualified refinement bank and unused feature root required")
    require_output_capacity(a.output, 2 * 1024**3)
    bank = json.loads(path.read_text())
    if (
        bank["status"] != "QUALIFIED_RESCUE_REFINEMENT_BANK_READY_PENDING_OWN_FEATURE_AUDIT"
        or len(bank["records"]) != bank["fit_plan"]["fit_histories"]
        or bank["observation_width"] != 160
        or bank["no_teacher_hidden_state"] is not True
        or bank["original_negative_histories_retained"] is not True
        or bank["own_feature_audit_required"] is not True
        or any(
            bank[k] != 0
            for k in (
                "physical_steps",
                "student_optimizer_updates",
                "DEV_fit_rows",
                "Fresh_fit_rows",
            )
        )
        or any(
            bank[k] is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("complete unchanged TRAIN-only17+41+rescues contract required")
    pins = dict(bank["input_file_hashes"])
    pins[str(path)] = a.bank_hash
    _, trained = receipt(pins, bank["parent_training_hash"])
    if trained["config"]["seed"] != 105025 or trained["head_optimizer_steps"] != 2048:
        raise ValueError("exact whole acquired105025 motor including its learned head required")
    for r in bank["records"]:
        f = a.bank_root / r["raw_path"]
        if f.is_symlink() or f.resolve().parent != a.bank_root.resolve():
            raise ValueError("confined complete curated raw history required")
        pins[str(f)] = r["raw_hash"]
        if r["source_format"] == "shared_player":
            pins[r["source_path"]] = r["source_hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            bank_hash=a.bank_hash,
            parent_training_hash=bank["parent_training_hash"],
            initial_refinement_seed=115023,
            feature_contract="own_current105025_normalized160+private_hidden128",
            teacher_hidden_state_executed=False,
            physical_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    records = []
    decisions = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        _, parent = open_motor(
            pins,
            bank["parent_training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="consolidated-adapter-v1",
        )
        torch.manual_seed(115023)
        torch.use_deterministic_algorithms(True)
        refined = RefinementMotorAdapter(parent).requires_grad_(False).eval()
        old_weights = {k: v.clone() for k, v in parent.state_dict().items()}
        scale = parent.memory.parent.get_buffer("action_scale").detach().cpu().numpy().copy()
        for row in bank["records"]:
            with np.load(a.bank_root / row["raw_path"], allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            if raw["observations"].shape != (row["rows"], 160) or not np.array_equal(
                raw["action_scale"], scale
            ):
                raise ValueError("whole actual observations and original action scale required")
            trace = None
            if row["source_format"] == "shared_player":
                with np.load(row["source_path"], allow_pickle=False) as z:
                    trace = {
                        k: z[k].copy()
                        for k in (
                            "NN_obs",
                            "NN_action",
                            "NN_reference_nominal",
                            "NN_h_before",
                            "NN_c_before",
                            "NN_h_after",
                            "NN_c_after",
                        )
                    }
                if not np.array_equal(trace["NN_obs"], raw["observations"]):
                    raise ValueError("curated observations differ from whole physical source")
            pstate, rstate = parent.initial_state(), refined.initial_state()
            features, deltas, targets = [], [], []
            for i, obs in enumerate(raw["observations"]):
                if trace is not None and any(
                    not np.array_equal(t.numpy(), trace[k][i])
                    for t, k in zip(pstate, ("NN_h_before", "NN_c_before"), strict=True)
                ):
                    raise ValueError(
                        "own computed current-parent prehistory differs from physical trace"
                    )
                value = torch.from_numpy(obs).reshape(1, 1, 160)
                with torch.no_grad():
                    mean, after, e = parent.forward_with_evidence(value, pstate)
                    identity, own_after, _ = refined.forward_with_evidence(value, rstate)
                decisions += 1
                if not torch.equal(mean, identity) or any(
                    not torch.equal(x, y) for x, y in zip(after, own_after, strict=True)
                ):
                    raise ValueError(
                        "zero refinement changed whole acquired motor or private state"
                    )
                nominal = e["reference_nominal"][0, 0].numpy()
                if trace is not None and (
                    not np.array_equal(mean[0, 0].numpy(), trace["NN_action"][i])
                    or not np.array_equal(nominal, trace["NN_reference_nominal"][i])
                    or any(
                        not np.array_equal(t.numpy(), trace[k][i])
                        for t, k in zip(after, ("NN_h_after", "NN_c_after"), strict=True)
                    )
                ):
                    raise ValueError(
                        "own parent action/reference/posthistory differs from physical trace"
                    )
                delta = e["final_mean_delta_radians"][0, 0].numpy().copy()
                # Anchors self-distill the CURRENT whole motor, not a different
                # teacher on an already successful state. Original measured
                # actions remain intact in the source bank for inspection.
                target = (
                    (raw["targets"][i] - nominal) * scale
                    if row["role"] == "qualified_rescue"
                    else delta.copy()
                )
                if not np.isfinite(target).all() or abs(target).max() > 0.20002:
                    raise ValueError(
                        "finite actual teacher target inside original total authority required"
                    )
                features.append(e["critic_features"][0, 0].numpy().copy())
                deltas.append(delta)
                targets.append(target)
                pstate, rstate = after, own_after
            output = a.output / f"features{row['index']}.npz"
            np.savez_compressed(
                output,
                features=np.stack(features),
                parent_delta=np.stack(deltas),
                target_delta=np.stack(targets),
                loss_weights=raw["loss_weights"],
                action_scale=scale,
            )
            record = dict(
                index=row["index"],
                role=row["role"],
                rows=row["rows"],
                feature_path=output.name,
                feature_hash=digest(output),
                source_raw_hash=row["raw_hash"],
                zero_refinement_bit_identity=True,
                parent_trace_compared=trace is not None,
                parent_trace_identity=True if trace is not None else None,
                target_semantics="actual_executed_teacher_delta"
                if row["role"] == "qualified_rescue"
                else "current_parent_self_distillation",
                borrowed_hidden_state=False,
            )
            records.append(record)
            write_once(a.output / f"features{row['index']}.json", record)
        if any(not torch.equal(v, old_weights[k]) for k, v in parent.state_dict().items()):
            raise ValueError("whole parent weights changed during feature audit")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PASS_REFINEMENT_OWN_FEATURES_ZERO_IDENTITY_NOT_PHYSICS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                bank_hash=a.bank_hash,
                parent_training_hash=bank["parent_training_hash"],
                parent_checkpoint_hash=trained["checkpoint_hash"],
                parent_lineage_hash=bank["parent_lineage_hash"],
                fit_plan=bank["fit_plan"],
                records=records,
                own_feature_decisions=decisions,
                actor_forwards=2 * decisions,
                zero_refinement_bit_identity=True,
                all_shared_parent_traces_match=True,
                teacher_hidden_state_executed=False,
                parent_weights_unchanged=True,
                physical_retention_proven=False,
                physical_steps=0,
                student_optimizer_updates=0,
                DEV_fit_rows=0,
                Fresh_fit_rows=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as e:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(e).__name__,
                error=str(e),
                records=records,
                own_feature_decisions=decisions,
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
