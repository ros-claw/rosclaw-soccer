"""Curate qualified rescues and all17+41 old anchors, never teacher hidden states."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("coverage-root", "native-root", "native-own-root", "retained-bank-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "coverage-hash",
        "coverage-head",
        "native-hash",
        "native-own-hash",
        "retained-bank-hash",
        "parent-training-hash",
        "expected-head",
    ):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    from rosclaw_soccer.rsi.pass_refinement_curriculum import refinement_curriculum
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_output_capacity import require_output_capacity

    roots = (a.coverage_root, a.native_root, a.native_own_root, a.retained_bank_root)
    hashes = (a.coverage_hash, a.native_hash, a.native_own_hash, a.retained_bank_hash)
    paths = [r / "complete.json" for r in roots]
    if (
        a.output.exists()
        or any(r.is_symlink() or (r / "failed.json").exists() for r in roots)
        or any(p.is_symlink() or digest(p) != h for p, h in zip(paths, hashes, strict=True))
    ):
        raise ValueError("unused bank and exact ordinary whole input proofs required")
    require_output_capacity(a.output, 2 * 1024**3)
    c, n, o, old = [json.loads(p.read_text()) for p in paths]
    if (
        c["status"] != "PASS_TEACHER_TRAIN20_COVERAGE_CLOSED_NOT_STUDENT_LEARNING"
        or c["source_head"] != a.coverage_head
        or n["status"] != "TEAM_PASS_CANDIDATE80_COLLECTED_PENDING_OWN_NATIVE_AND_RETENTION"
        or o["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or o["producer_hash"] != a.native_hash
        or any(v["training_hash"] != a.parent_training_hash for v in (n, o))
        or n["checkpoint_kind"] != "consolidated-adapter-v1"
        or o["records"] != [{k: v for k, v in r.items() if k != "artifacts"} for r in n["records"]]
        or o["counts"] != n["counts"]
        or old["whole_actual_sequences"] is not True
        or old["teacher_hidden_state_supplied_to_learner"] is not False
        or any(
            v[k] is not False
            for v in (c, n, o, old)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
        or old["DEV_fit_rows"] != 0
        or old["Fresh_fit_rows"] != 0
    ):
        raise ValueError(
            "qualified coverage and unchanged actual105025/TRAIN-only old bank required"
        )
    plan = refinement_curriculum(c, n["records"], old["records"])
    pins = {}
    for v in (c, o, old):
        for key, h in v["input_file_hashes"].items():
            if key in pins and pins[key] != h:
                raise ValueError("conflicting complete bank ancestry")
            pins[key] = h
    pins.update({str(p): h for p, h in zip(paths, hashes, strict=True)})
    _, parent = receipt(pins, a.parent_training_hash)
    if (
        parent["config"]["seed"] != 105025
        or parent["parent_lineage_hash"] != old["parent_lineage_hash"]
    ):
        raise ValueError("same current105025 and original slow-memory ancestry required")

    def raw_path(root, art):
        path = root / art["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("confined actual source arrays required")
        pins[str(path)] = art["hash"]
        return path

    # Bind all negatives too, not just the fitting subset.
    for row in n["records"]:
        for art in row["artifacts"].values():
            raw_path(a.native_root, art)
    for row in old["records"]:
        raw_path(a.retained_bank_root, dict(path=row["raw_path"], hash=row["raw_hash"]))
    sources = []
    for i in plan["original_motion_anchor_indices"]:
        row = old["records"][i]
        sources.append(
            dict(
                role="retained_original17",
                format="retained_bank",
                path=a.retained_bank_root / row["raw_path"],
                row=row,
                evidence_hash=a.retained_bank_hash,
            )
        )
    for i in plan["current_pass_anchor_indices"]:
        row = n["records"][i]
        sources.append(
            dict(
                role="retained_current41",
                format="shared_player",
                path=raw_path(a.native_root, row["artifacts"]["A"]),
                row=row,
                evidence_hash=a.native_own_hash,
            )
        )
    for ptr in plan["correction_sources"]:
        sp, source = receipt(pins, ptr["source_hash"])
        _, own = receipt(pins, ptr["own_hash"])
        row = source["records"][ptr["trial_index"]]
        if (
            sp.parent != Path(ptr["source_root"])
            or row["outcome"] != ptr["outcome"]
            or row["course"] != ptr["course"]
            or row["role"] != "selected_repeat"
            or own["producer_hash"] != ptr["source_hash"]
        ):
            raise ValueError("exact independently replayed rescue raw pointer required")
        sources.append(
            dict(
                role="qualified_rescue",
                format="shared_player",
                path=raw_path(sp.parent, row["artifacts"]["A"]),
                row=row,
                evidence_hash=ptr["own_hash"],
            )
        )
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.current105025_rescue_refinement_bank.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            coverage_hash=a.coverage_hash,
            parent_training_hash=a.parent_training_hash,
            retained_bank_hash=a.retained_bank_hash,
            native_hash=a.native_hash,
            native_own_hash=a.native_own_hash,
            fit_plan=plan,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    records = []
    try:
        import numpy as np

        scale = None
        for item in sources:
            row, path = item["row"], item["path"]
            with np.load(path, allow_pickle=False) as z:
                if item["format"] == "retained_bank":
                    raw = {
                        k: z[k].copy()
                        for k in ("observations", "targets", "action_scale", "reference_nominal")
                    }
                    if scale is None:
                        scale = raw["action_scale"].copy()
                    if not np.array_equal(scale, raw["action_scale"]):
                        raise ValueError("unchanged original policy-order action scale required")
                else:
                    raw = dict(
                        observations=z["NN_obs"].copy(),
                        targets=z["executed_action"].copy(),
                        action_scale=scale.copy(),
                        reference_nominal=z["NN_reference_nominal"].copy(),
                        source_parent_mean=z["NN_action"].copy(),
                    )
            length = len(raw["observations"])
            if (
                scale.shape != (29,)
                or scale.dtype != np.float32
                or not (scale > 0).all()
                or raw["observations"].shape != (length, 160)
                or raw["targets"].shape != (length, 29)
                or raw["reference_nominal"].shape != (length, 29)
                or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in raw.values())
                or not 100 < length <= 400
                or np.max(abs((raw["targets"] - raw["reference_nominal"]) * scale)) > 0.20002
            ):
                raise ValueError(
                    "whole finite actual160->29 histories inside original authority required"
                )
            weights = np.ones(length, dtype=np.float32)
            if item["role"] == "retained_current41":
                if not np.array_equal(raw["targets"], raw["source_parent_mean"]):
                    raise ValueError("protected current histories must be truly unassisted")
            elif item["role"] == "qualified_rescue":
                changed = np.max(abs((raw["targets"] - raw["source_parent_mean"]) * scale), axis=1)
                if changed.max() <= 1e-6 or row["stand_controls"] != 100:
                    raise ValueError(
                        "actual nonzero physical correction and whole100NNstand required"
                    )
                weights[changed > 1e-4] = 6.0
            raw["loss_weights"] = weights
            index = len(records)
            output = a.output / f"sequence{index}.npz"
            np.savez_compressed(output, **raw)
            record = dict(
                index=index,
                role=item["role"],
                rows=length,
                raw_path=output.name,
                raw_hash=digest(output),
                source_path=str(path),
                source_hash=digest(path),
                source_format=item["format"],
                source_evidence_hash=item["evidence_hash"],
                source_index=row["index"],
                teacher_hidden_state_supplied=False,
                task_success=True,
                success_definition=(
                    "original_retained_motion_task"
                    if item["role"] == "retained_original17"
                    else "actual_controlled_Afoot_then_Blegal_contact_NOT_precision_exam"
                ),
                source_outcome=row.get("outcome", row.get("measured_outcome")),
            )
            records.append(record)
            write_once(a.output / f"sequence{index}.json", record)
        if len(records) != plan["fit_histories"]:
            raise ValueError("every planned whole history required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="QUALIFIED_RESCUE_REFINEMENT_BANK_READY_PENDING_OWN_FEATURE_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                coverage_hash=a.coverage_hash,
                parent_training_hash=a.parent_training_hash,
                native_hash=a.native_hash,
                native_own_hash=a.native_own_hash,
                retained_bank_hash=a.retained_bank_hash,
                fit_plan=plan,
                records=records,
                parent_lineage_hash=parent["parent_lineage_hash"],
                contexts=old["contexts"],
                current80_outcomes=n["records"],
                original274_metadata=old["records"],
                all_transfer_outcomes=c["transfers"],
                observation_width=160,
                no_teacher_hidden_state=True,
                task_labels_modified=False,
                task_success_is_role_specific_not_precision_exam=True,
                original_negative_histories_retained=True,
                own_feature_audit_required=True,
                training_authorized=False,
                physical_steps=0,
                student_optimizer_updates=0,
                DEV_fit_rows=0,
                Fresh_fit_rows=0,
                correction_bank_ready=True,
                learned_policy_improvement_verified=False,
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
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
