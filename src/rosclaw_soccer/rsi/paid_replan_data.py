"""Admit executed expert rows only after ALL current-student CEM trials are audited."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_intervention_replan import replan_config
from rosclaw_soccer.rsi.paid_interventional_data import append_unique, executed_features


def admitted_replan_windows(
    producer: dict[str, Any], audit: dict[str, Any], producer_hash: str
) -> list[dict[str, Any]]:
    config = replan_config(104, 289)
    if (
        producer.get("status") != "PAID_INTERVENTIONAL_OWN_PREFIX_REPLAN_CLOSED_PENDING_AUDIT"
        or audit.get("status") != "PAID_INTERVENTIONAL_OWN_PREFIX_ALL_TRIAL_NN_IK_NATIVE_AUDIT_PASS"
        or audit.get("source_hash") != producer_hash
        or producer.get("config") != config
        or producer["counts"]["episodes"] != config["max_episodes"]
        or producer["counts"]["CEM_updates"] != 4
        or audit["counts"]["episodes"] != config["max_episodes"]
        or audit["counts"]["native"] != producer["actual_native_steps"]
        or audit["counts"]["neural"] != producer["actual_neural_rows"]
        or audit["counts"]["R2_planner"] != producer["counts"]["R2_planner"]
        or audit["counts"]["IK"] != producer["counts"]["IK"]
        or any(
            v.get("optimizer_updates") != 0 or v.get("Fresh_rows_read") != 0
            for v in (producer, audit)
        )
        or any(
            v.get(k) is not False
            for v in (producer, audit)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete all-trial same-source TRAIN-only independent witness required")
    if [s["switch_control"] for s in producer["searches"]] != config["switch_controls"] or [
        s["switch_control"] for s in audit["summaries"]
    ] != config["switch_controls"]:
        raise ValueError("both declared recovery windows required")
    admitted = []
    for search, summary in zip(producer["searches"], audit["summaries"], strict=True):
        cut = search["switch_control"]
        if len(search["selected"]) != 2 or len(summary["outcomes"]) != 2:
            raise ValueError("two fully audited selected repeats required")
        for rows in search["selected"] + summary["outcomes"]:
            if [r["case"] for r in rows] != [4, 5]:
                raise ValueError("complete paired source cases required")
        for index, row in enumerate(search["selected"][0]):
            outcomes = [rep[index] for rep in summary["outcomes"]]
            for repeat, own in enumerate(outcomes):
                recorded = search["selected"][repeat][index]
                healthy = all(
                    own.get(k) is True for k in ("quality", "stand_healthy", "stand_stable")
                )
                if (
                    recorded.get("healthy_success") is not healthy
                    or own.get("healthy_success") is not healthy
                    or recorded.get("quality") is not own.get("quality")
                    or recorded["switch_control"] != cut
                    or recorded["latent"] != search["latent"]
                ):
                    raise ValueError("actual expert contact and recovery witness differs")
            if all(o["healthy_success"] is True for o in outcomes):
                admitted.append(
                    dict(
                        case=row["case"],
                        switch_control=cut,
                        record=row,
                        canonical_repeat=0,
                        actual_physical_repeats=2,
                    )
                )
    if not admitted:
        raise ValueError("no independently successful current-state expert windows")
    return admitted


def aggregate_replan(
    parent: Any,
    previous: dict[str, np.ndarray],
    windows: list[dict[str, Any]],
    read: Any,
    *,
    source_hash: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    additions = []
    for window in windows:
        cut, row = window["switch_control"], window["record"]
        raw = read(row["kick"])
        z, frames = executed_features(parent, raw, cut)
        additions.append(
            (
                z,
                frames,
                dict(
                    source_hash=source_hash,
                    case=window["case"],
                    switch_control=cut,
                    canonical_repeat=0,
                    physical_repeats=2,
                    raw_path=row["kick"]["raw_path"],
                    raw_hash=row["kick"]["raw_hash"],
                    whole_source_rows=len(raw["NN_action"]),
                    student_prefix_rows_excluded=cut,
                ),
            )
        )
    return append_unique(previous, additions)


def load_replanned_data(
    root: Path, expected: str, policy: Path
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
    """Strict generation-two loader; the generation-one loader remains unchanged."""
    from rosclaw_soccer.rsi.paid_full_entry_student import digest
    from rosclaw_soccer.rsi.paid_interventional_data import load_interventional_data

    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact ordinary closed second-generation TRAIN data required")
    r, d = [json.loads(path.read_text()) for path in (rp, dp)]
    keys = (
        "previous_root",
        "previous_hash",
        "replan_root",
        "replan_hash",
        "audit_root",
        "audit_hash",
        "source_student_hash",
        "original_root",
        "original_hash",
    )
    if (
        r.get("status") != "PAID_REPLANNED_INTERVENTIONAL_TRAIN_FEATURES_CAUSALLY_RECOMPUTED"
        or r.get("schema") != "paid.interventional_dagger.TRAIN_features.v2"
        or r.get("schema") != d.get("schema")
        or r.get("declaration_hash") != digest(dp)
        or r.get("source_head") != d.get("source_head")
        or any(r.get(k) != d.get(k) for k in keys)
        or any(
            v.get("bank") != "train" or v.get("Fresh_fit_rows") != 0 or v.get("DEV_fit_rows") != 0
            for v in (r, d)
        )
        or r.get("optimizer_updates") != 0
        or r.get("own_native_steps") != 0
        or any(
            v.get(k) is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("explicit generation-two TRAIN-only extension required")
    parent, baseline, old = load_interventional_data(
        Path(d["previous_root"]), d["previous_hash"], policy
    )
    pr, ar = Path(d["replan_root"]), Path(d["audit_root"])
    for folder, h in ((pr, d["replan_hash"]), (ar, d["audit_hash"])):
        path = folder / "complete.json"
        if path.is_symlink() or digest(path) != h or (folder / "failed.json").exists():
            raise ValueError("exact successful all-trial teacher and audit contracts required")
    p, a = [json.loads((folder / "complete.json").read_text()) for folder in (pr, ar)]
    pdp = pr / "declaration.json"
    pd = json.loads(pdp.read_text())
    sp = Path(pd["source_arguments"]["student_root"]) / "complete.json"
    if (
        pdp.is_symlink()
        or digest(pdp) != p["declaration_hash"]
        or sp.is_symlink()
        or digest(sp) != d["source_student_hash"]
        or d["source_student_hash"] != pd["source_arguments"]["student_hash"]
        or pd["body_scene_hash"] != baseline["body_scene_hash"]
        or any(
            r[k] != baseline[k]
            for k in (
                "original_root",
                "original_hash",
                "body_scene_hash",
                "parent_hash",
                "parent_checkpoint_hash",
            )
        )
    ):
        raise ValueError("same current-student and original frozen motor lineage required")
    student = json.loads(sp.read_text())
    if (
        student["data_hash"] != d["previous_hash"]
        or student["config"]["seed_index"] != 0
        or student["status"]
        != "PAID_INTERVENTIONAL_DAGGER_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
        or any(
            student[k] != baseline[k]
            for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")
        )
    ):
        raise ValueError("current source student's own curriculum and body must match")
    windows = admitted_replan_windows(p, a, d["replan_hash"])

    def read(link: dict[str, Any]) -> dict[str, np.ndarray]:
        path = pr / link["raw_path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(pr.resolve())
            or digest(path) != link["raw_hash"]
            or a["input_file_hashes"].get(str(path)) != link["raw_hash"]
        ):
            raise ValueError("own audited executed expert trajectory required")
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    recomputed, groups = aggregate_replan(parent, old, windows, read, source_hash=d["replan_hash"])
    f, outcomes = Path(r["feature_path"]), root / "source-outcomes.json"
    if (
        f.is_symlink()
        or f.parent != root
        or digest(f) != r["feature_hash"]
        or outcomes.is_symlink()
        or digest(outcomes) != r["source_outcomes_hash"]
    ):
        raise ValueError("confined exact features and complete success/failure memory required")
    with np.load(f, allow_pickle=False) as archive:
        z = {k: archive[k].copy() for k in archive.files}
    if (
        set(z) != set(recomputed)
        or any(z[k].dtype != v.dtype or not np.array_equal(z[k], v) for k, v in recomputed.items())
        or r["new_groups"] != groups
        or r["rows"] != len(z["features"])
        or r["previous_rows"] != len(old["features"])
        or r["original_rows"] != baseline["original_rows"]
        or r["protected_rows"] != int((~z["teacher"]).sum())
        or r.get("previous_arrays_exact_prefix") is not True
    ):
        raise ValueError("generation-two features differ from complete causal recomputation")
    return parent, r, z
