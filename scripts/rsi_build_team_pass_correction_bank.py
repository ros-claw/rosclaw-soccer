"""Build274 complete histories after ALL four continuation native own proofs.

177 original histories +17 actual current-student retention anchors +80 actual
student-prefix/expert-suffix histories. Preserve every negative with BC0.
The80 prefix0 teacher controls remain calibration evidence, not duplicate fit.
No physics, training update, private Fresh access or promotion occurs here.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bank-root", "retention-own-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("bank-hash", "retention-own-hash", "expected-head"):
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
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    bp, rp = a.bank_root / "complete.json", a.retention_own_root / "complete.json"
    if (
        a.output.exists()
        or len(a.shard) != 4
        or any(x.is_symlink() for x in (bp, rp))
        or [digest(bp), digest(rp)] != [a.bank_hash, a.retention_own_hash]
    ):
        raise ValueError(
            "unused bank and exact complete original/current retention witnesses required"
        )
    bank, retained = [json.loads(x.read_text()) for x in (bp, rp)]
    from rosclaw_soccer.rsi.acquired_motor_retention import native_successes
    from rosclaw_soccer.rsi.team_pass_consolidation import BANK_STATUS, RETAINED, pass_pools

    pass_pools(bank["records"])
    if (
        bank["status"] != BANK_STATUS
        or retained["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or retained["checkpoint_kind"] != "team-pass-consolidated-v1"
        or not set(RETAINED) <= native_successes(retained["results"])
    ):
        raise ValueError(
            "all original177 histories and all17 actual current retained skills required"
        )
    pins = dict(bank["input_file_hashes"])

    def merge(doc):
        for key, value in doc["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting source proof commitment")
            pins[key] = value

    merge(retained)
    pins.update({str(bp): a.bank_hash, str(rp): a.retention_own_hash})
    _, training = receipt(pins, retained["training_hash"])
    ep, exam = receipt(pins, retained["source_hash"])
    if (
        training["config"]["seed"] != 75024
        or training["data_hash"] != a.bank_hash
        or len(exam["records"]) != 44
    ):
        raise ValueError(
            "same fully retained research student75024 and all44 actual anchors required"
        )
    dp = ep.parent / "declaration.json"
    scope = json.loads(dp.read_text())
    if (
        pins.get(str(dp)) != digest(dp)
        or digest(dp) != exam["declaration_hash"]
        or scope["contexts"] != bank["contexts"]
    ):
        raise ValueError("unchanged original native22 context scope required")
    sources = []
    for root, h, own_root, oh in a.shard:
        root, own_root = Path(root), Path(own_root)
        paths = [root / "complete.json", own_root / "complete.json", root / "declaration.json"]
        if (
            any(x.is_symlink() for x in paths)
            or any((x / "failed.json").exists() for x in (root, own_root))
            or [digest(x) for x in paths[:2]] != [h, oh]
        ):
            raise ValueError("ALL four whole closed source and own continuation proofs required")
        s, own, d = [json.loads(x.read_text()) for x in paths]
        if (
            s["status"] != "STUDENT_PASS_CONTINUATION_SHARD_COLLECTED_PENDING_WHOLE_OWN_REPLAY"
            or own["status"] != "STUDENT_PASS_CONTINUATION_SHARD_WHOLE_OWN_NATIVE_REPLAY_MATCH"
            or own["producer_hash"] != h
            or own["counts"] != s["counts"]
            or s["declaration_hash"] != digest(paths[2])
            or d["bank_hash"] != a.bank_hash
            or d["retention_own_hash"] != a.retention_own_hash
            or d["student_training_hash"] != retained["training_hash"]
            or len(s["records"]) != 40
            or len(own["records"]) != 40
            or any(
                {k: v for k, v in x.items() if k != "artifacts"} != y
                for x, y in zip(s["records"], own["records"], strict=True)
            )
            or any(
                x[k] is not False
                for x in (s, own, d)
                for k in (
                    "hardware_authorized",
                    "promotion_authorized",
                    "goal_complete",
                    "Fresh_verified",
                )
            )
        ):
            raise ValueError(
                "same actual unpromoted student corrections and whole own histories required"
            )
        merge(own)
        pins.update({str(x): digest(x) for x in paths})
        sources.append((root, s))
    if [s["context"] for _, s in sources] != [1, 4, 20, 21]:
        raise ValueError("ALL four context shards in declared order required")
    anchors = [x for x in exam["records"] if x["repeat"] == 0 and x["context"] in RETAINED]
    if sorted(x["context"] for x in anchors) != list(RETAINED) or any(
        not x["healthy_quality"] for x in anchors
    ):
        raise ValueError("complete actual current17 positive skill anchors required")
    corrections = [
        (root, x) for root, s in sources for x in s["records"] if x["student_prefix_controls"] == 80
    ]
    if len(corrections) != 80 or {
        (x["course"]["state_id"], x["course"]["task"]) for _, x in corrections
    } != {(i, j) for i in range(40) for j in (0, 1)}:
        raise ValueError("all80 actual corrections, including every failure, required")
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.team_pass_actual_student_correction_bank274.v2",
            source_head=a.expected_head,
            input_file_hashes=pins,
            base_data_hash=a.bank_hash,
            warm_start_hash=retained["training_hash"],
            parent_lineage_hash=training["parent_lineage_hash"],
            actual_sequences=274,
            original_sequences=177,
            current_skill_anchors=17,
            actual_corrections=80,
            teacher_calibration_worlds_not_fit=80,
            prior_TRAIN_initial_conditions=40,
            learner_h_borrowed=False,
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records, queries = [], 0
    try:
        import numpy as np

        from rosclaw_soccer.rsi.contact_consolidation_data import (
            contact_loss_weights,
            own_reference_sequence,
        )
        from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label
        from rosclaw_soccer.rsi.team_pass_correction import CORRECTION_BANK_STATUS, correction_pools
        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        bench, model = open_motor(
            pins,
            retained["training_hash"],
            {i: contexts[i] for i in RETAINED},
            checkpoint_kind="team-pass-consolidated-v1",
        )
        scale = model.parent.action_scale.cpu().numpy().copy()
        episodes = [("original", a.bank_root, x) for x in bank["records"]]
        episodes += [("current_acquired17", ep.parent, x) for x in anchors]
        episodes += [("own_state_expert", root, x) for root, x in corrections]
        for mode, root, row in episodes:
            if mode == "original":
                path = root / row["raw_path"]
                if digest(path) != row["raw_hash"]:
                    raise ValueError("whole original177 histories required unchanged")
                with np.load(path, allow_pickle=False) as z:
                    arrays = {k: z[k].copy() for k in z.files}
                obs, targets, weights = [
                    arrays[k] for k in ("observations", "targets", "loss_weights")
                ]
                role, positive, kick, stand = [
                    row[k] for k in ("role", "task_success", "kick_rows", "stand_rows")
                ]
                context, course, measured = [
                    row[k] for k in ("source_context", "source_course", "measured_outcome")
                ]
            else:
                if mode == "current_acquired17":
                    raw = []
                    for name in ("kick", "stand"):
                        art = row["artifacts"][name]
                        path = root / art["path"]
                        if pins.get(str(path)) != digest(path) or digest(path) != art["hash"]:
                            raise ValueError("bound actual current NN anchor required")
                        with np.load(path, allow_pickle=False) as z:
                            raw.append({k: z[k].copy() for k in ("NN_obs", "NN_action")})
                    obs = np.concatenate([z["NN_obs"] for z in raw])
                    targets = np.concatenate([z["NN_action"] for z in raw])
                    kick, stand = len(raw[0]["NN_obs"]), len(raw[1]["NN_obs"])
                    positive, context, course, measured = True, row["context"], None, row["result"]
                    first = row["result"]["postcontact"]["first_foot_contact_substep"]
                else:
                    art = row["artifacts"]["A"]
                    path = root / art["path"]
                    if pins.get(str(path)) != digest(path) or digest(path) != art["hash"]:
                        raise ValueError("bound actual expert continuation required")
                    with np.load(path, allow_pickle=False) as z:
                        obs, targets = z["NN_obs"].copy(), z["executed_action"].copy()
                        mask = z["student_executed"]
                        if not np.array_equal(mask, np.arange(len(obs)) < 80) or not np.array_equal(
                            z["STUDENT_NN_obs"], z["TEACHER_NN_obs"]
                        ):
                            raise ValueError("actual private own-state causal prefix required")
                    stand = row["stand_controls"]
                    kick = len(obs) - stand
                    measured = row["outcome"]
                    first = measured["incoming"]["A_first_foot_strike_substep"]
                    positive = controlled_pass_label(measured)
                    context = row["course"]["context"]
                    course = row["course"]
                    if positive != row["actual_continuation_controlled_pass"]:
                        raise ValueError("own qualified correction label differs")
                role = mode
                weights = contact_loss_weights(kick, stand, first)
            if (
                obs.shape != (len(targets), 160)
                or targets.shape != (len(obs), 29)
                or not 1 <= len(obs) <= 400
                or any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (obs, targets))
                or obs[0, 125:154].any()
                or not np.array_equal(obs[1:, 125:154], targets[:-1])
                or (positive and stand != 100)
            ):
                raise ValueError(
                    "whole causal actual float32 sequence without h borrowing required"
                )
            nominal, prior = own_reference_sequence(model.parent, obs)
            queries += len(obs)
            if np.abs((targets - nominal) * scale).max() > 0.20002:
                raise ValueError("original total motor authority cannot change")
            if mode == "original" and any(
                not np.array_equal(arrays[k], v)
                for k, v in (
                    ("reference_nominal", nominal),
                    ("reference_delta", prior),
                    ("action_scale", scale),
                )
            ):
                raise ValueError("original frozen reference history changed")
            path = a.output / f"sequence{len(records)}.npz"
            np.savez_compressed(
                path,
                observations=obs,
                targets=targets,
                action_scale=scale,
                reference_nominal=nominal,
                reference_delta=prior,
                loss_weights=weights,
            )
            records.append(
                dict(
                    index=len(records),
                    role=role,
                    bc_weight=int(positive),
                    task_success=positive,
                    rows=len(obs),
                    kick_rows=kick,
                    stand_rows=stand,
                    raw_path=path.name,
                    raw_hash=digest(path),
                    source_root=str(root),
                    source_index=row.get("index"),
                    source_context=context,
                    source_course=course,
                    measured_outcome=measured,
                    teacher_hidden_state_supplied_to_learner=False,
                )
            )
            print(
                json.dumps(
                    dict(index=len(records) - 1, role=role, positive=positive, rows=len(obs))
                ),
                flush=True,
            )
        correction_pools(records)
        if bench.native_steps:
            raise ValueError("bank construction cannot integrate physics")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=CORRECTION_BANK_STATUS,
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                records=records,
                base_data_hash=a.bank_hash,
                warm_start_hash=retained["training_hash"],
                parent_lineage_hash=training["parent_lineage_hash"],
                contexts=bank["contexts"],
                own_reference_queries=queries,
                observation_width=160,
                learner_hidden_width=256,
                whole_actual_sequences=True,
                teacher_hidden_state_supplied_to_learner=False,
                physical_steps=0,
                optimizer_updates=0,
                DEV_fit_rows=0,
                Fresh_fit_rows=0,
                student_success_claim=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="PASS_CORRECTION_BANK_FAILED_PRESERVE_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                records=records,
                own_reference_queries=queries,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
