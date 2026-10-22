"""TRAIN-only canonical successful-behavior choice, not averaging different intents.

Old demonstrations remain immutable evidence. Only the materialized supervised
choice for a novel, unadmitted skill may change. Protected old rows never change.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DATA_STATUS = "PAID_CANONICAL_SUCCESSFUL_BEHAVIOR_TRAIN_CAUSALLY_RECOMPUTED"
SCHEMA = "paid.canonical_successful_behavior.TRAIN_features.v1"
SOURCE_NAMES = ("original", "teacher", "teacher_audit", "replan", "replan_audit")


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_witnesses(identity: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    sources: dict[str, Any] = {}
    pins: dict[str, str] = {}
    for name in SOURCE_NAMES:
        root = Path(identity[name + "_root"])
        path = root / "complete.json"
        if (
            path.is_symlink()
            or digest(path) != identity[name + "_hash"]
            or (root / "failed.json").exists()
        ):
            raise ValueError("exact closed canonical-choice source required")
        r = json.loads(path.read_text())
        if any(r.get(k) is not False for k in ("hardware_authorized", "promotion_authorized")):
            raise ValueError("SIM-only canonical-choice source required")
        sources[name] = r
        for f, h in r["input_file_hashes"].items():
            if f in pins and pins[f] != h:
                raise ValueError("conflicting canonical-choice provenance")
            pins[f] = h
        pins[str(path)] = digest(path)
    original, teacher, ta, replan, ra = [sources[n] for n in SOURCE_NAMES]
    if (
        original["bank"] != "train"
        or teacher["bank"] != "train"
        or teacher["config"]["seed_index"] != 0
        or teacher["episodes"] != 108
        or ta["status"] != "PAID_FIRST_CONTACT_ALL_TRIAL_OWN_NN_IK_AND_NATIVE_AUDIT_PASS"
        or ta["source_hash"] != identity["teacher_hash"]
        or ta["counts"]["episodes"] != 108
        or ta["own_native_steps"] != teacher["actual_native_steps"]
        or ta["own_trajectory_neural_rows"] != teacher["actual_neural_rows"]
        or ta["selected_whole_repeats_exact"] is not True
        or teacher["searches"][0]["pair"] != [4, 5]
        or any(
            v.get("optimizer_updates") != 0 or v.get("Fresh_rows_read") != 0
            for v in (teacher, ta, replan, ra)
        )
        or any(
            original["input_file_hashes"].get(str(Path(identity[n + "_root"]) / "complete.json"))
            != identity[n + "_hash"]
            for n in ("teacher", "teacher_audit")
        )
    ):
        raise ValueError("original audited paired TRAIN teacher lineage required")
    pdp = Path(identity["replan_root"]) / "declaration.json"
    pd = json.loads(pdp.read_text())
    if (
        pdp.is_symlink()
        or digest(pdp) != replan["declaration_hash"]
        or ra["input_file_hashes"].get(str(pdp)) != digest(pdp)
        or pd["source_arguments"]["teacher_hash"] != identity["teacher_hash"]
        or pd["source_arguments"]["teacher_audit_hash"] != identity["teacher_audit_hash"]
        or pd["body_scene_hash"] != original["body_scene_hash"]
    ):
        raise ValueError("same-body independently bound recovery source required")
    pins[str(pdp)] = digest(pdp)
    return sources, pins


def quality_score(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    """Worst lateral error and minimum progress of TWO observed full repeats."""
    if len(rows) != 2:
        raise ValueError("two whole-repeat outcomes required")
    lateral, progress = [], []
    for row in rows:
        k, stand = row["kick"], row["stand"]
        p = k["postcontact"]
        if (
            k["body_failed"] is not False
            or k["forbidden"] is not False
            or any(
                p.get(key) is not True
                for key in ("correct_foot_first", "post_contact_quality_observed")
            )
            or p.get("other_foot_contact_in_window") is not False
            or not stand
            or stand.get("healthy") is not True
            or stand.get("stability_observed") is not True
        ):
            return None
        values = [
            p[k]
            for k in (
                "goal_lateral_1p2s_m",
                "goal_progress_1p2s_m",
                "maximum_pelvis_retreat_after_contact_m",
            )
        ]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError("finite observed physical quality required")
        if abs(values[0]) > 0.75 or values[1] < 1 or not 0 <= values[2] <= 0.05:
            return None
        lateral.append(abs(values[0]))
        progress.append(values[1])
    return -max(lateral), min(progress)


def canonical_choices(sources: dict[str, Any], replan_hash: str) -> list[dict[str, Any]]:
    from rosclaw_soccer.rsi.paid_replan_data import admitted_replan_windows

    p, a = sources["replan"], sources["replan_audit"]
    windows = admitted_replan_windows(p, a, replan_hash)
    teacher = sources["teacher"]["searches"][0]
    if len(teacher["selected"]) != 2:
        raise ValueError("complete original teacher repeats required")
    choices = []
    for case in (4, 5):
        original_rows = [
            next(r for r in repeat if r["case"] == case) for repeat in teacher["selected"]
        ]
        options = []
        score = quality_score(original_rows)
        if score is not None:
            options.append(
                dict(
                    case=case,
                    kind="original",
                    switch_control=None,
                    score=list(score),
                    record=original_rows[0],
                    original_tie_preferred=1,
                )
            )
        for window in windows:
            if window["case"] != case:
                continue
            search = next(
                s for s in p["searches"] if s["switch_control"] == window["switch_control"]
            )
            rows = [next(r for r in repeat if r["case"] == case) for repeat in search["selected"]]
            score = quality_score(rows)
            if score is not None:
                options.append(
                    dict(
                        case=case,
                        kind="replan",
                        switch_control=window["switch_control"],
                        score=list(score),
                        record=rows[0],
                        original_tie_preferred=0,
                    )
                )
        if not options:
            raise ValueError("no physically supported complete successful behavior")
        selected = max(
            options,
            key=lambda v: (*v["score"], v["original_tie_preferred"], -(v["switch_control"] or 0)),
        )
        choices.append(dict(selected=selected, alternatives=options))
    return choices


def assemble_canonical(
    parent: Any,
    baseline: dict[str, Any],
    old: dict[str, Any],
    choices: list[dict[str, Any]],
    read: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np

    from rosclaw_soccer.rsi.paid_interventional_data import successful_behavior_features

    z = {k: v.copy() for k, v in old.items()}
    trajectories, retired = copy.deepcopy(baseline["trajectories"]), []
    changed = np.zeros(len(z["features"]), np.bool_)
    for option in choices:
        choice = option["selected"]
        t = next(
            t
            for t in trajectories
            if t["role"] == "novel_first_cartesian_teacher" and t["case"] == choice["case"]
        )
        if choice["kind"] == "original":
            if t["source_hash"] != choice["record"]["kick"]["raw_hash"]:
                raise ValueError("original selected teacher archive differs")
            continue
        cut = choice["switch_control"]
        raw = read(choice["record"]["kick"])
        rows, frames = successful_behavior_features(parent, raw, cut)
        sl = slice(t["start"], t["stop"])
        if frames != list(range(t["rows"])) or len(rows["features"]) != t["stop"] - t["start"]:
            raise ValueError("same complete horizon and canonical offsets required")
        if not old["teacher"][sl].all():
            raise ValueError("cannot replace protected motor experience")
        retired.append(copy.deepcopy(t))
        for k in ("features", "target", "parent_delta"):
            z[k][sl] = rows[k]
        z["active"][sl] = np.abs(rows["target"]).max(axis=1) > 1e-6
        changed[sl] = True
        t.update(
            role="canonical_successful_mixed_behavior",
            recorded_policy="actual_student_prefix_then_expert",
            source_path=choice["record"]["kick"]["raw_path"],
            source_hash=choice["record"]["kick"]["raw_hash"],
            student_prefix_rows=cut,
            executed_expert_rows=t["rows"] - cut,
            switch_control=cut,
            original_teacher_retained_in_archive=True,
        )
    if not changed.any():
        raise ValueError("no canonical behavior change; no duplicate fitting job")
    if any(not np.array_equal(z[k][~changed], old[k][~changed]) for k in old):
        raise ValueError("unselected canonical or protected rows changed")
    known: dict[bytes, tuple[Any, Any]] = {}
    for x, y, p in zip(z["features"], z["target"], z["parent_delta"], strict=True):
        key = x.tobytes()
        if key in known and (
            not np.allclose(y, known[key][0], rtol=0, atol=2e-6)
            or not np.array_equal(p, known[key][1])
        ):
            raise ValueError(
                "canonical choice conflicts at identical context; retain protected data"
            )
        known.setdefault(key, (y, p))
    return z, trajectories, retired


def reconstruct(
    identity: dict[str, Any], policy: Path
) -> tuple[
    Any,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    import numpy as np

    from rosclaw_soccer.rsi.paid_full_entry_student import load_full_entry_data

    sources, _ = read_witnesses(identity)
    parent, baseline, old = load_full_entry_data(
        Path(identity["original_root"]), identity["original_hash"], policy
    )
    choices = canonical_choices(sources, identity["replan_hash"])
    root = Path(identity["replan_root"])

    def read(link: dict[str, Any]) -> dict[str, Any]:
        path = root / link["raw_path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or digest(path) != link["raw_hash"]
            or sources["replan_audit"]["input_file_hashes"].get(str(path)) != link["raw_hash"]
        ):
            raise ValueError("independently audited complete successful raw required")
        with np.load(path, allow_pickle=False) as a:
            return {k: a[k].copy() for k in a.files}

    z, trajectories, retired = assemble_canonical(parent, baseline, old, choices, read)
    return parent, baseline, old, z, choices, trajectories, retired


def load_canonical_data(
    root: Path, expected: str, policy: Path
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import numpy as np

    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact closed canonical behavior data required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r.get("status") != DATA_STATUS
        or r.get("schema") != SCHEMA
        or d.get("schema") != SCHEMA
        or r.get("declaration_hash") != digest(dp)
        or r.get("source_head") != d.get("source_head")
        or any(r.get(k) != d.get(k) for n in SOURCE_NAMES for k in (n + "_root", n + "_hash"))
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
        raise ValueError("explicit TRAIN-only canonical-choice contract required")
    parent, baseline, old, recomputed, choices, trajectories, retired = reconstruct(d, policy)
    f, archive = Path(r["feature_path"]), root / "canonical-choice-memory.json"
    if (
        f.is_symlink()
        or f.parent != root
        or digest(f) != r["feature_hash"]
        or archive.is_symlink()
        or digest(archive) != r["choice_memory_hash"]
    ):
        raise ValueError("confined features and immutable alternative behavior memory required")
    with np.load(f, allow_pickle=False) as a:
        z = {k: a[k].copy() for k in a.files}
    if (
        set(z) != set(recomputed)
        or any(z[k].dtype != v.dtype or not np.array_equal(z[k], v) for k, v in recomputed.items())
        or r["choices"] != choices
        or r["trajectories"] != trajectories
        or r["retired_fit_trajectories"] != retired
        or json.loads(archive.read_text())
        != dict(choices=choices, retired=retired, original_artifacts_unchanged=True)
        or any(
            r[k] != baseline[k]
            for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")
        )
        or r["rows"] != len(old["features"])
        or r["protected_rows"] != int((~z["teacher"]).sum())
    ):
        raise ValueError("canonical data differ from complete causal reconstruction")
    return parent, r, z
