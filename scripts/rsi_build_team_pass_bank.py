"""Build177 complete TRAIN sequences from all four audited pass shards.

17 retained motor-skill anchors plus ALL160 new paired trajectories, including
every negative. Targets are actual executed A actions, never teacher hidden
states. No physics, optimizer update, DEV/Fresh fit or promotion happens here.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("frontier-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("frontier-head", "frontier-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument(
        "--shard",
        nargs=4,
        action="append",
        required=True,
        metavar=("ROOT", "HASH", "OWN_ROOT", "OWN_HASH"),
    )
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists() or len(a.shard) != 4:
        raise ValueError("unused bank and all4 source/own shard commitments required")
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    if [c["seed"] for c in candidates] != [65023]:
        raise ValueError("same loss-free motor65023 required")
    candidate = candidates[0]
    sources = []
    for root, h, own_root, oh in a.shard:
        root, own_root = Path(root), Path(own_root)
        paths = [root / "complete.json", own_root / "complete.json", root / "declaration.json"]
        if (
            any(x.is_symlink() for x in paths)
            or any((r / "failed.json").exists() for r in (root, own_root))
            or [digest(x) for x in paths[:2]] != [h, oh]
        ):
            raise ValueError("all exact closed independently replayed shards required")
        s, own, d = [json.loads(x.read_text()) for x in paths]
        if (
            s["status"] != "PASS_TRAIN_PERTURBATION40_SHARD_COLLECTED_PENDING_ALL_SHARDS_OWN"
            or own["status"] != "PASS_TRAIN_PERTURBATION40_SHARD_WHOLE_OWN_NATIVE_REPLAY_MATCH"
            or own["producer_hash"] != h
            or own["counts"] != s["counts"]
            or own["context"] != s["context"]
            or s["declaration_hash"] != digest(paths[2])
            or d["candidate"] != candidate
            or len(s["records"]) != 40
            or len(own["records"]) != 40
            or len(s["calibration"]) != 2
            or len(own["calibration"]) != 2
            or any(
                r[k] is not False
                for r in (s, own)
                for k in (
                    "goal_complete",
                    "Fresh_verified",
                    "promotion_authorized",
                    "hardware_authorized",
                )
            )
            or any(
                {k: v for k, v in r.items() if k != "artifacts"} != o
                for key in ("records", "calibration")
                for r, o in zip(s[key], own[key], strict=True)
            )
        ):
            raise ValueError("all42 own native/privateNN outcomes required before bank admission")
        for path, value in own["input_file_hashes"].items():
            if path in pins and pins[path] != value:
                raise ValueError("conflicting physical evidence lineage")
            pins[path] = value
        pins.update({str(path): digest(path) for path in paths})
        sources.append((root, s))
    if [s["context"] for _, s in sources] != [1, 4, 20, 21]:
        raise ValueError("all four TRAIN context shards in declared order required")
    frontier = json.loads((a.frontier_root / "complete.json").read_text())
    report = next(r for r in frontier["reports"] if r["seed"] == 65023)
    _, native_own = receipt(pins, report["own_audit_hash"])
    ep, exam = receipt(pins, native_own["source_hash"])
    required = report["required_contexts"]
    if (
        required != [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21]
        or native_own["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or native_own["worlds"] != 44
        or native_own["training_hash"] != candidate["training_hash"]
        or exam["training_hash"] != candidate["training_hash"]
        or len(exam["records"]) != 44
    ):
        raise ValueError("exact all17 acquired old skills of current65023 required")
    anchors = [r for r in exam["records"] if r["context"] in required and r["repeat"] == 0]
    if sorted(r["context"] for r in anchors) != required or any(
        not r["healthy_quality"] for r in anchors
    ):
        raise ValueError("ALL17 actual healthy retained anchors, not selected old subset, required")
    wp, warm = receipt(pins, candidate["training_hash"])
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.team_pass_training_bank177.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            candidate=candidate,
            warm_start_hash=candidate["training_hash"],
            parent_lineage_hash=warm["parent_lineage_hash"],
            original_retained_contexts=required,
            records=177,
            new_physical_initial_conditions=40,
            new_paired_trajectories=160,
            posthoc_training_labels=True,
            teacher_hidden_state_supplied_to_learner=False,
            DEV_fit_rows=0,
            Fresh_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records = []
    queries = 0
    try:
        import numpy as np

        from rosclaw_soccer.rsi.contact_consolidation_data import (
            contact_loss_weights,
            own_reference_sequence,
        )
        from rosclaw_soccer.rsi.pass_transfer_learning import (
            controlled_pass_label,
            paired_pass_learning_summary,
        )
        from scripts.rsi_stroke_search_common import open_motor

        newrows = [r for _, s in sources for r in s["records"]]
        summary = paired_pass_learning_summary(newrows)
        bench, model = open_motor(
            pins,
            candidate["training_hash"],
            {i: contexts[i] for i in required},
            checkpoint_kind=candidate["kind"],
        )
        scale = model.parent.action_scale.cpu().numpy().copy()
        episodes = [(ep.parent, r, "retained_neural17", True, False) for r in anchors]
        episodes.extend(
            (
                root,
                r,
                "new_teacher" if r["course"]["assisted"] else "new_unassisted",
                controlled_pass_label(r["outcome"]),
                True,
            )
            for root, s in sources
            for r in s["records"]
        )
        for index, (root, row, role, positive, team) in enumerate(episodes):
            if team:
                art = row["artifacts"]["A"]
                path = root / art["path"]
                if path.is_symlink() or digest(path) != art["hash"]:
                    raise ValueError("actual A inference sequence changed")
                with np.load(path, allow_pickle=False) as z:
                    obs, targets = z["NN_obs"].copy(), z["executed_action"].copy()
                    standing = z["standing"].copy()
                stand = int(standing.sum())
                kick = len(obs) - stand
                if not np.array_equal(standing, np.r_[np.zeros(kick, bool), np.ones(stand, bool)]):
                    raise ValueError(
                        "one complete reference then contiguous actual recovery required"
                    )
                first = row["outcome"]["incoming"]["A_first_foot_strike_substep"]
                measured = row["outcome"]
            else:
                raw = {}
                for key in ("kick", "stand"):
                    art = row["artifacts"][key]
                    path = root / art["path"]
                    if path.is_symlink() or digest(path) != art["hash"]:
                        raise ValueError("all actual current-policy anchor artifacts required")
                    with np.load(path, allow_pickle=False) as z:
                        raw[key] = {k: z[k].copy() for k in ("NN_obs", "NN_action")}
                obs = np.concatenate([raw[k]["NN_obs"] for k in ("kick", "stand")])
                targets = np.concatenate([raw[k]["NN_action"] for k in ("kick", "stand")])
                kick, stand = len(raw["kick"]["NN_obs"]), len(raw["stand"]["NN_obs"])
                first = row["result"]["postcontact"]["first_foot_contact_substep"]
                measured = row["result"]
            if (
                obs.shape != (len(targets), 160)
                or targets.shape != (len(obs), 29)
                or any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (obs, targets))
                or not 1 <= len(obs) <= 400
                or not 1 <= kick <= 300
                or not 0 <= stand <= 100
                or (positive and stand != 100)
                or obs[0, 125:154].any()
                or not np.array_equal(obs[1:, 125:154], targets[:-1])
            ):
                raise ValueError(
                    "complete actual float32 history and executed-action feedback required"
                )
            nominal, prior = own_reference_sequence(model.parent, obs)
            queries += len(obs)
            if np.abs((targets - nominal) * scale).max() > 0.20002:
                raise ValueError("actual target escaped original total motor authority")
            weights = contact_loss_weights(kick, stand, first)
            path = a.output / f"sequence{index}.npz"
            np.savez_compressed(
                path,
                observations=obs,
                targets=targets,
                action_scale=scale,
                reference_nominal=nominal,
                reference_delta=prior,
                loss_weights=weights,
            )
            record = dict(
                index=index,
                role=role,
                bc_weight=1 if positive else 0,
                task_success=positive,
                rows=len(obs),
                kick_rows=kick,
                stand_rows=stand,
                raw_path=path.name,
                raw_hash=digest(path),
                source_root=str(root),
                source_context=row.get("context") if not team else row["course"]["context"],
                source_index=row.get("index"),
                source_course=row.get("course"),
                measured_outcome=measured,
                teacher_hidden_state_supplied_to_learner=False,
            )
            records.append(record)
            print(
                json.dumps(dict(index=index, role=role, positive=positive, rows=len(obs))),
                flush=True,
            )
        if len(records) != 177 or bench.native_steps:
            raise ValueError("complete177 bank with ZERO new physics required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="TEAM_PASS_WHOLE_HISTORY_BANK177_READY_NOT_TRAINED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                candidate=candidate,
                contexts=contexts,
                baseline_training_hash=base_hash,
                warm_start_hash=candidate["training_hash"],
                parent_lineage_hash=warm["parent_lineage_hash"],
                records=records,
                teacher_summary=summary,
                own_reference_queries=queries,
                observation_width=160,
                learner_hidden_width=256,
                teacher_hidden_state_supplied_to_learner=False,
                whole_actual_sequences=True,
                physical_steps=0,
                optimizer_updates=0,
                DEV_fit_rows=0,
                Fresh_fit_rows=0,
                student_success_claim=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="TEAM_PASS_BANK_FAILED_PRESERVE_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                completed_sequences=len(records),
                own_reference_queries=queries,
                physical_steps=0,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
