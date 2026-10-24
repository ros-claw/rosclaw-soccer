"""Recover CURRENT acquired parent's actual successful whole retention histories.

Historical65023 anchors in bank62 are off-policy for105025. They are retained,
not renamed. These additional own-state features can support a later separately
declared output-protection study. No new fitting or physics occurs here.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("retention-own-root", "feature-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("retention-own-hash", "feature-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    op, fp = a.retention_own_root / "complete.json", a.feature_root / "complete.json"
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (op, fp))
        or digest(op) != a.retention_own_hash
        or digest(fp) != a.feature_hash
        or any((p.parent / "failed.json").exists() for p in (op, fp))
    ):
        raise ValueError("exact whole parent proof and unused own-feature output required")
    own, feature = [json.loads(p.read_text()) for p in (op, fp)]
    if (
        own["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or own["worlds"] != 44
        or own["independent_whole_world_replay"] is not True
        or own["checkpoint_kind"] != "consolidated-adapter-v1"
        or own["training_hash"] != feature["parent_training_hash"]
        or own["checkpoint_hash"] != feature["parent_checkpoint_hash"]
        or feature["status"] != "PASS_REFINEMENT_OWN_FEATURES_ZERO_IDENTITY_NOT_PHYSICS"
        or any(
            d[k] is not False
            for d in (own, feature)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("same whole acquired parent and independent physical44 required")
    pins = dict(feature["input_file_hashes"])
    for key, value in own["input_file_hashes"].items():
        if key in pins and pins[key] != value:
            raise ValueError("conflicting qualified parent ancestry")
        pins[key] = value
    pins.update({str(op): a.retention_own_hash, str(fp): a.feature_hash})
    sp, source = receipt(pins, own["source_hash"])
    _, bank = receipt(pins, feature["bank_hash"])
    if (
        source["status"] != "DUAL_RECURRENT_NATIVE_TRAIN_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT"
        or source["training_hash"] != own["training_hash"]
        or len(source["records"]) != 44
    ):
        raise ValueError("same unchanged whole native retention source required")
    source_pins(pins)
    check_pins(pins)
    from rosclaw_soccer.rsi.acquired_motor_retention import native_successes

    contexts = sorted(native_successes(own["results"]))
    rows = [r for r in source["records"] if r["context"] in contexts and r["repeat"] == 0]
    if len(rows) != len(contexts) or any(not r["healthy_quality"] for r in rows):
        raise ValueError("all current parent's physically successful contexts required")
    sources = []
    for row in rows:
        paths = []
        for phase in ("kick", "stand"):
            art = row["artifacts"][phase]
            path = sp.parent / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent
                != (sp.parent / f"context{row['context']}-repeat0").resolve()
                or pins.get(str(path)) != art["hash"]
                or digest(path) != art["hash"]
            ):
                raise ValueError("exact already replayed actual motion and recovery required")
            paths.append(path)
        sources.append((row, paths))
    a.output.mkdir()
    calls = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.private_motion_history import join_private_motion_history
        from scripts.rsi_stroke_search_common import open_motor

        mapping = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        _, actor = open_motor(
            pins,
            own["training_hash"],
            {i: mapping[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="consolidated-adapter-v1",
        )
        scale = actor.memory.parent.get_buffer("action_scale").numpy().copy()
        records = []
        for row, paths in sources:
            phases = []
            for path in paths:
                with np.load(path, allow_pickle=False) as z:
                    phases.append({k: z[k].copy() for k in z.files if k.startswith("NN_")})
            raw = join_private_motion_history(*phases)
            state = actor.initial_state()
            features, deltas, means, nominal = [], [], [], []
            for i, observation in enumerate(raw["NN_obs"]):
                if any(
                    not np.array_equal(t.numpy(), raw[k][i])
                    for t, k in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
                ):
                    raise ValueError("own current parent prehistory differs from actual trace")
                with torch.no_grad():
                    mean, state, e = actor.forward_with_evidence(
                        torch.from_numpy(observation).reshape(1, 1, 160), state
                    )
                calls += 1
                if not np.array_equal(mean[0, 0].numpy(), raw["NN_action"][i]) or any(
                    not np.array_equal(t.numpy(), raw[k][i])
                    for t, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
                ):
                    raise ValueError("own action or recurrent state differs from actual parent")
                features.append(e["critic_features"][0, 0].numpy().copy())
                deltas.append(e["final_mean_delta_radians"][0, 0].numpy().copy())
                means.append(mean[0, 0].numpy().copy())
                nominal.append(e["reference_nominal"][0, 0].numpy().copy())
            out = a.output / f"context{row['context']}.npz"
            np.savez_compressed(
                out,
                observations=raw["NN_obs"],
                features=np.stack(features),
                parent_delta=np.stack(deltas),
                parent_mean=np.stack(means),
                reference_nominal=np.stack(nominal),
                action_scale=scale,
            )
            record = dict(
                context=row["context"],
                rows=len(features),
                feature_path=out.name,
                feature_hash=digest(out),
                sources=[dict(path=str(p), hash=digest(p)) for p in paths],
                private_state_and_action_bit_identity=True,
                borrowed_hidden_state=False,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="CURRENT_PARENT_RETENTION_OWN_FEATURES_MATCH_ACTUAL_PHYSICS_NOT_NEW_LEARNING",
                source_head=a.expected_head,
                input_file_hashes=pins,
                retention_own_hash=a.retention_own_hash,
                feature_hash=a.feature_hash,
                parent_training_hash=own["training_hash"],
                parent_checkpoint_hash=own["checkpoint_hash"],
                contexts=contexts,
                records=records,
                own_actor_decisions=calls,
                historical_bank62_unchanged=True,
                onpolicy_for_current_parent=True,
                physical_steps_executed=0,
                optimizer_updates=0,
                Fresh_rows_read=0,
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
