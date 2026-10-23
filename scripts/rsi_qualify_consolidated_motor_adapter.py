"""Recompute own frozen learned-memory features; zero-head identity, NOT physics.

Exports ALL274 unchanged positive/negative TRAIN histories. No fitting occurs;
data eligibility, native execution, and checkpoint admission are separate work.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("data-root", "memory-own-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("data-hash", "memory-training-hash", "memory-own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    dp, ap = a.data_root / "complete.json", a.memory_own_root / "complete.json"
    if (
        a.output.exists()
        or any(x.is_symlink() or (x.parent / "failed.json").exists() for x in (dp, ap))
        or digest(dp) != a.data_hash
        or digest(ap) != a.memory_own_hash
    ):
        raise ValueError("unused output, exact complete bank and memory own-proof required")
    data, own = [json.loads(x.read_text()) for x in (dp, ap)]
    from rosclaw_soccer.rsi.single_mode_pass_curriculum import single_mode_pass_curriculum
    from rosclaw_soccer.rsi.team_pass_correction import CORRECTION_BANK_STATUS

    if (
        data["status"] != CORRECTION_BANK_STATUS
        or data["whole_actual_sequences"] is not True
        or data["teacher_hidden_state_supplied_to_learner"] is not False
        or data["observation_width"] != 160
        or data["learner_hidden_width"] != 256
        or own["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or own["training_hash"] != a.memory_training_hash
        or own["checkpoint_kind"] != "contact-consolidated-v2"
        or own["worlds"] != 44
        or any(
            d[k] is not False
            for d in (data, own)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
        or data["DEV_fit_rows"] != 0
        or data["Fresh_fit_rows"] != 0
    ):
        raise ValueError(
            "whole TRAIN-only actual histories and independently replayed memory required"
        )
    selection = single_mode_pass_curriculum(data["records"])
    pins = dict(data["input_file_hashes"])
    pins.update(own["input_file_hashes"])
    pins.update({str(dp): a.data_hash, str(ap): a.memory_own_hash})
    tp, trained = receipt(pins, a.memory_training_hash)
    if (
        trained["cumulative_optimizer_steps"] != 112
        or trained["cumulative_critic_optimizer_steps"] != 48
        or trained["checkpoint_hash"] != own["checkpoint_hash"]
        or trained["parent_lineage_hash"] != data["parent_lineage_hash"]
    ):
        raise ValueError("whole learned112/48 slow memory, not older foundation alone, required")
    for row in data["records"]:
        path = a.data_root / row["raw_path"]
        if path.is_symlink() or path.parent.resolve() != a.data_root.resolve():
            raise ValueError("confined ordinary whole trajectory required")
        pins[str(path)] = row["raw_hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            data_hash=a.data_hash,
            memory_training_hash=a.memory_training_hash,
            memory_own_hash=a.memory_own_hash,
            memory_checkpoint_hash=trained["checkpoint_hash"],
            input_file_hashes=pins,
            feature_contract="own_normalized160+own_learned_hidden128_private_zero256",
            selected_pass_plan=selection,
            note="No fit authorization. Future anchors must use original learned17, not warm75024.",
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    decisions = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
        from scripts.rsi_stroke_search_common import open_motor

        torch.manual_seed(105023)
        torch.use_deterministic_algorithms(True)
        contexts = (
            {int(k): v for k, v in data["contexts"].items()}
            if isinstance(data["contexts"], dict)
            else dict(enumerate(data["contexts"]))
        )
        _, memory = open_motor(
            pins,
            a.memory_training_hash,
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="contact-consolidated-v2",
        )
        adapter = ConsolidatedMotorAdapter(memory).requires_grad_(False).eval()
        before = {k: v.detach().clone() for k, v in memory.state_dict().items()}
        records = []
        for row in data["records"]:
            path = a.data_root / row["raw_path"]
            with np.load(path, allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            obs = torch.from_numpy(raw["observations"]).unsqueeze(1)
            if obs.shape != (row["rows"], 1, 160):
                raise ValueError("whole actual observation history required")
            with torch.no_grad():
                expected, old_after = memory(obs, memory.initial_state())
                actual, own_after, evidence = adapter.forward_with_evidence(
                    obs, adapter.initial_state()
                )
            if not torch.equal(expected, actual) or any(
                not torch.equal(x, y) for x, y in zip(old_after, own_after, strict=True)
            ):
                raise ValueError("zero-head entire learned memory and private h/c changed")
            scale = memory.parent.get_buffer("action_scale").cpu().numpy()
            if not np.array_equal(scale, raw["action_scale"]):
                raise ValueError("original HG29 motor scale changed")
            delta = evidence["final_mean_delta_radians"].squeeze(1).cpu().numpy()
            means = actual.squeeze(1).cpu().numpy()
            target_delta = (raw["targets"] - means) * scale
            output = a.output / f"features{row['index']}.npz"
            np.savez_compressed(
                output,
                features=evidence["critic_features"].squeeze(1).cpu().numpy(),
                memory_means=means,
                memory_delta=delta,
                target_delta=target_delta,
                loss_weights=raw["loss_weights"],
                action_scale=scale,
            )
            item = dict(
                index=row["index"],
                role=row["role"],
                source_raw_hash=row["raw_hash"],
                raw_task_success=row["task_success"],
                rows=row["rows"],
                feature_path=output.name,
                feature_hash=digest(output),
                maximum_teacher_difference_rad=float(np.abs(target_delta).max()),
                rows_exceeding_new_head_04rad=int((np.abs(target_delta).max(axis=1) > 0.04).sum()),
                zero_head_bit_identity=True,
            )
            records.append(item)
            decisions += row["rows"]
            write_once(a.output / f"history{row['index']}.json", item)
            print(json.dumps(dict(history=row["index"], decisions=decisions)), flush=True)
        if any(not torch.equal(v, before[k]) for k, v in adapter.memory.state_dict().items()):
            raise ValueError("entire learned slow memory changed")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="CONSOLIDATED_LEARNED_MEMORY_OWN274_ZERO_HEAD_IDENTITY_NOT_PHYSICS",
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                input_file_hashes=pins,
                data_hash=a.data_hash,
                memory_training_hash=a.memory_training_hash,
                memory_checkpoint_hash=trained["checkpoint_hash"],
                records=records,
                own_feature_decisions=decisions,
                independent_identity_reference_decisions=decisions,
                optimizer_updates=0,
                physical_steps=0,
                DEV_fit_rows=0,
                Fresh_fit_rows=0,
                teacher_hidden_state_supplied=False,
                zero_head_bit_identity=True,
                physical_retention_proven=False,
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
                decisions=decisions,
                optimizer_updates=0,
                physical_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
