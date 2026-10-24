"""Own recurrent CPU replay of ALL79 histories; exact protected motor identity.

This is recorded-observation inference, not a new physical success claim.
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
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    tp = a.training_root / "complete.json"
    if (
        a.output.exists()
        or tp.is_symlink()
        or digest(tp) != a.training_hash
        or (a.training_root / "failed.json").exists()
    ):
        raise ValueError("unused CPU audit and exact assembled motor required")
    t = json.loads(tp.read_text())
    from rosclaw_soccer.rsi.supported_motor_checkpoint import KIND, STATUS, support_cpu_scope

    if t["status"] != STATUS or t["checkpoint_kind"] != KIND:
        raise ValueError("explicit supported neural reference bundle required")
    pins = dict(t["input_file_hashes"])
    pins[str(tp)] = a.training_hash
    pins[t["checkpoint_path"]] = t["checkpoint_hash"]
    dp = a.training_root / "declaration.json"
    pins[str(dp)] = t["declaration_hash"]
    bp, bank = receipt(pins, t["bank_hash"])
    fp, feature = receipt(pins, t["feature_hash"])
    rp, old = receipt(pins, t["retention_feature_hash"])
    expected = support_cpu_scope(t)
    histories = []
    for row, f in zip(bank["records"], feature["records"], strict=True):
        raw_path, feature_path = bp.parent / row["raw_path"], fp.parent / f["feature_path"]
        if row["index"] != f["index"] or row["raw_hash"] != f["source_raw_hash"]:
            raise ValueError("same whole62 qualified histories required")
        histories.append((row["role"], row["rows"], raw_path, feature_path))
        for path, h, root in (
            (raw_path, row["raw_hash"], bp.parent),
            (feature_path, f["feature_hash"], fp.parent),
        ):
            if path.is_symlink() or path.resolve().parent != root.resolve():
                raise ValueError("confined complete bank/features required")
            pins[str(path)] = h
    for row in old["records"]:
        path = rp.parent / row["feature_path"]
        if path.is_symlink() or path.resolve().parent != rp.parent.resolve():
            raise ValueError("confined actual current motion history required")
        pins[str(path)] = row["feature_hash"]
        histories.append(("retained_current_motion17", row["rows"], path, path))
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
        _, actor = open_motor(
            pins, a.training_hash, {i: contexts[i] for i in (1, 4, 20, 21)}, checkpoint_kind=KIND
        )
        results = []
        for index, (role, rows, raw_path, feature_path) in enumerate(histories):
            with np.load(raw_path, allow_pickle=False) as z:
                obs, nominal, scale = [
                    z[k].copy() for k in ("observations", "reference_nominal", "action_scale")
                ]
            with np.load(feature_path, allow_pickle=False) as z:
                f = {k: z[k].copy() for k in ("features", "parent_delta")}
            if (
                obs.shape != (rows, 160)
                or nominal.shape != (rows, 29)
                or f["features"].shape != (rows, 288)
            ):
                raise ValueError("whole actual features and observations required")
            state = actor.initial_state()
            active, means, gates = 0, [], []
            for i, o in enumerate(obs):
                with torch.no_grad():
                    mean, state, e = actor.forward_with_evidence(
                        torch.from_numpy(o).reshape(1, 1, 160), state
                    )
                    _, delta = actor.adapt_features(
                        torch.from_numpy(f["features"][i]).reshape(1, 1, 288),
                        torch.from_numpy(f["parent_delta"][i]).reshape(1, 1, 29),
                    )
                calls += 1
                parent_mean = nominal[i] + f["parent_delta"][i] / scale
                predicted = nominal[i] + delta[0, 0].numpy() / scale
                gate = float(e["support_activation"].item())
                if (
                    not np.array_equal(e["critic_features"][0, 0].numpy(), f["features"][i])
                    or not np.array_equal(e["refinement_parent_mean"][0, 0].numpy(), parent_mean)
                    or not np.array_equal(mean[0, 0].numpy(), predicted)
                    or (e["final_mean_delta_radians"].abs() > 0.2).any()
                ):
                    raise ValueError("own parent/features/masked-head/bound replay differs")
                if role != "qualified_rescue" and (
                    gate != 0 or not np.array_equal(mean[0, 0].numpy(), parent_mean)
                ):
                    raise ValueError("protected history lost exact acquired-parent action identity")
                active += gate > 0
                means.append(mean[0, 0].numpy().copy())
                gates.append(gate)
            output = a.output / f"history{index}.npz"
            np.savez_compressed(output, means=np.stack(means), support_activation=np.array(gates))
            record = dict(
                index=index,
                role=role,
                rows=rows,
                support_active_rows=active,
                prediction_path=output.name,
                prediction_hash=digest(output),
                protected_action_identity=(role != "qualified_rescue"),
            )
            results.append(record)
            print(json.dumps(record), flush=True)
        if calls != expected or len(results) != 79:
            raise ValueError("ALL79 private recurrent histories required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=a.training_hash,
                checkpoint_kind=KIND,
                checkpoint_hash=t["checkpoint_hash"],
                data_hash=t["bank_hash"],
                own_actor_decisions=calls,
                results=results,
                protected_histories=75,
                protected_action_bit_identity=True,
                physical_steps_executed=0,
                optimizer_updates=0,
                old_retention_proven=False,
                Fresh_verified=False,
                team_game_verified=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                own_actor_decisions=calls,
                physical_steps_executed=0,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
