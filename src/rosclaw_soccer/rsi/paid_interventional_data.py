"""Executed-expert-only aggregation on independently audited student states.

Original full-entry data retain their strict original schema. This is an
explicit extension, not relabelling counterfactual failed teachers as success.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_full_entry_student import digest, load_full_entry_data

DATA_STATUS = "PAID_INTERVENTIONAL_DAGGER_TRAIN_FEATURES_CAUSALLY_RECOMPUTED"


def load_recovery_witness(
    root: Path,
    expected: str,
    audit_root: Path,
    audit_hash: str,
    *,
    original_hash: str,
    scene_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for folder, h in ((root, expected), (audit_root, audit_hash)):
        rp = folder / "complete.json"
        if rp.is_symlink() or digest(rp) != h or (folder / "failed.json").exists():
            raise ValueError("exact closed recovery witnesses required")
    p, a = [json.loads((folder / "complete.json").read_text()) for folder in (root, audit_root)]
    dp = root / "declaration.json"
    if (
        dp.is_symlink()
        or digest(dp) != p["declaration_hash"]
        or a["input_file_hashes"].get(str(dp)) != digest(dp)
    ):
        raise ValueError("independently bound recovery declaration required")
    d = json.loads(dp.read_text())
    srp = Path(d["source_arguments"]["student_root"]) / "complete.json"
    if (
        srp.is_symlink()
        or digest(srp) != d["source_arguments"]["student_hash"]
        or a["input_file_hashes"].get(str(srp)) != digest(srp)
    ):
        raise ValueError("exact source student identity required")
    s = json.loads(srp.read_text())
    if (
        d["body_scene_hash"] != scene_hash
        or s["body_scene_hash"] != scene_hash
        or s["data_hash"] != original_hash
        or d["config"] != p["config"]
        or s["status"] != "PAID_ACTIVE_COMPACT_SUPPORT_AUDITED_NOT_PHYSICALLY_QUALIFIED"
        or s["config"]["seed_index"] != 0
    ):
        raise ValueError("same original TRAIN curriculum and physical body required")
    admitted_windows(p, a, expected)
    return p, a


def admitted_windows(p: dict[str, Any], a: dict[str, Any], expected: str) -> list[dict[str, Any]]:
    if (
        p.get("status") != "PAID_OWN_STATE_TEACHER_HANDOFF_CLOSED_PENDING_AUDIT"
        or a.get("status") != "PAID_OWN_STATE_TEACHER_NN_PLANNER_IK_ALL_NATIVE_AUDIT_PASS"
        or a.get("source_hash") != expected
        or p.get("config") != a.get("config")
        or p.get("episodes") != 20
        or a.get("counts", {}).get("episodes") != 20
        or a.get("own_native_steps") != p.get("actual_native_steps")
        or a.get("own_actor_neural_rows") != p.get("actual_neural_rows")
        or a.get("own_R2_planner_head_rows") != p.get("planner_R2_head_rows")
        or a.get("counts", {}).get("IK") != p.get("IK_queries")
        or p.get("config", {}).get("protocol") != "paid.TRAIN.own_state_teacher_handoff.v1"
        or p["config"].get("cases") != [4, 5]
        or p["config"].get("repeats") != 2
        or any(o.get("optimizer_updates") != 0 or o.get("Fresh_rows_read") != 0 for o in (p, a))
        or any(
            o.get(k) is not False
            for o in (p, a)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete same-source own-NN/planner/IK/native TRAIN witness required")
    arms = p["config"]["arms"]
    if set(arms) != {"teacher", "phase_start", "middle", "late", "student"}:
        raise ValueError("original bounded handoff design required")
    summaries = {r["arm"]: r for r in a["summary"]}
    result = []
    for arm in p["arms"]:
        name = arm["arm"]
        if name not in ("middle", "late"):
            continue
        cut = arm["switch_control"]
        if (
            type(cut) is not int
            or not arms["phase_start"] < cut < arms["student"]
            or cut != arms[name]
        ):
            raise ValueError("only declared post-deviation recovery branches may add data")
        if len(arm["repeats"]) != 2:
            raise ValueError("both independently audited whole repeats required")
        for row in arm["repeats"][0]:
            case = row["case"]
            own = next(v for v in summaries[name]["cases"] if v["case"] == case)
            if row["healthy_success"] is not own["healthy_success"]:
                raise ValueError("source recovery claim differs from independent witness")
            if own["healthy_success"] is not True:
                continue
            if (
                own["quality"] is not True
                or own["stand_healthy"] is not True
                or own["stand_stable"] is not True
            ):
                raise ValueError("successful contact and observed recovery required")
            result.append(dict(arm=name, case=case, switch_control=cut, record=row))
    if not result:
        raise ValueError("no physically supported new teacher recovery data")
    return result


def executed_features(
    parent: Any, raw: dict[str, np.ndarray], cut: int
) -> tuple[dict[str, np.ndarray], list[int]]:
    return _mixed_features(parent, raw, cut, include_student_prefix=False)


def successful_behavior_features(
    parent: Any, raw: dict[str, np.ndarray], cut: int
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Entire externally audited successful behavior; prefix is NOT expert action."""
    return _mixed_features(parent, raw, cut, include_student_prefix=True)


def _mixed_features(
    parent: Any, raw: dict[str, np.ndarray], cut: int, *, include_student_prefix: bool
) -> tuple[dict[str, np.ndarray], list[int]]:
    n = len(raw["NN_action"])
    shapes: dict[str, tuple[int, ...]] = dict(
        NN_obs=(n, 160),
        NN_action=(n, 29),
        PLAN_R2_nominal_action=(n, 29),
        PLAN_teacher_action=(n, 29),
    )
    if include_student_prefix:
        shapes["NN_nominal_action"] = (n, 29)
    shapes.update(
        {"NN_" + k: (n, 2, 1, 128) for k in ("h_before", "c_before", "h_after", "c_after")}
    )
    if (
        type(cut) is not int
        or not 0 < cut < n <= 300
        or any(
            raw[k].shape != v or raw[k].dtype != np.float32 or not np.isfinite(raw[k]).all()
            for k, v in shapes.items()
        )
        or raw["PLAN_teacher_applied"].dtype != np.bool_
        or not np.array_equal(raw["PLAN_teacher_applied"], np.arange(n) >= cut)
        or not np.array_equal(raw["PLAN_context_frame"], np.arange(n))
        or not np.array_equal(raw["PLAN_switch_control"], np.full(n, cut))
        or not np.array_equal(raw["NN_action"][cut:], raw["PLAN_teacher_action"][cut:])
        or include_student_prefix
        and not np.array_equal(raw["NN_action"][:cut], raw["NN_nominal_action"][:cut])
        or raw["NN_obs"][0, 125:154].any()
        or not np.array_equal(raw["NN_obs"][1:, 125:154], raw["NN_action"][:-1])
    ):
        raise ValueError("actual executed expert window and complete own-history raw required")
    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    xs, ys, ps = [], [], []
    with torch.no_grad():
        for j, item in enumerate(raw["NN_obs"]):
            if any(
                not np.array_equal(t.numpy(), raw[k][j])
                for t, k in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
            ):
                raise ValueError("student own before-history differs; never reset at takeover")
            obs = torch.from_numpy(item.copy()).reshape(1, 1, 160)
            nominal, state, hidden = parent.base.forward_with_features(obs, state)
            if not np.array_equal(nominal[0, 0].numpy(), raw["PLAN_R2_nominal_action"][j]) or any(
                not np.array_equal(t.numpy(), raw[k][j])
                for t, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
            ):
                raise ValueError("own R2 nominal/after-history differs")
            if j < cut and not include_student_prefix:
                continue
            x = torch.cat([(obs - parent.base.mean) / parent.base.denominator, hidden], dim=-1)[
                0, 0
            ]
            prior = parent.residual(x.reshape(1, 288))[0]
            actual = torch.from_numpy(raw["NN_action"][j].copy())
            y = (actual - (nominal[0, 0] + prior / parent.action_scale)) * parent.action_scale
            total = (actual - nominal[0, 0]) * parent.action_scale
            if (
                not torch.isfinite(y).all()
                or total.abs().max() > 0.20002
                or y.abs().max() > 0.40002
            ):
                raise ValueError("original R2-relative authority exceeded")
            xs.append(x.numpy().copy())
            ys.append(y.numpy().copy())
            ps.append(prior.numpy().copy())
    return dict(features=np.stack(xs), target=np.stack(ys), parent_delta=np.stack(ps)), list(
        range(0 if include_student_prefix else cut, n)
    )


def append_unique(
    old: dict[str, np.ndarray],
    additions: list[tuple[dict[str, np.ndarray], list[int], dict[str, Any]]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    # Preserve all original rows byte-for-byte. Only new compatible duplicates
    # may be skipped, never average different actions at one identical state.
    known: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for x, y, p in zip(old["features"], old["target"], old["parent_delta"], strict=True):
        key = x.tobytes()
        if key in known and (
            not np.array_equal(known[key][1], p)
            or not np.allclose(known[key][0], y, rtol=0, atol=2e-6)
        ):
            raise ValueError("original curriculum has conflicting identical features")
        known.setdefault(key, (y, p))
    groups = []
    chunks = {k: [v.copy()] for k, v in old.items()}
    cursor = len(old["features"])
    group_id = int(old["group"].max()) + 1
    for z, frames, metadata in additions:
        kept = []
        for j, (x, y, p) in enumerate(
            zip(z["features"], z["target"], z["parent_delta"], strict=True)
        ):
            key = x.tobytes()
            if key in known:
                oy, op = known[key]
                if not np.array_equal(op, p) or not np.allclose(oy, y, rtol=0, atol=2e-6):
                    raise ValueError("conflicting expert actions at identical own features")
            else:
                known[key] = (y, p)
                kept.append(j)
        if not kept:
            continue
        n = len(kept)
        for k in ("features", "target", "parent_delta"):
            chunks[k].append(z[k][kept].copy())
        chunks["teacher"].append(np.ones(n, np.bool_))
        chunks["active"].append(np.abs(z["target"][kept]).max(axis=1) > 1e-6)
        chunks["group"].append(np.full(n, group_id, np.int32))
        groups.append(
            dict(
                **metadata,
                group=group_id,
                start=cursor,
                stop=cursor + n,
                rows=n,
                source_frames=[frames[j] for j in kept],
                duplicate_rows_skipped=len(frames) - n,
                role="executed_teacher_on_student_states",
            )
        )
        cursor += n
        group_id += 1
    if not groups:
        raise ValueError("no new unique executed teacher rows")
    return {k: np.concatenate(v) for k, v in chunks.items()}, groups


def aggregate_handoff(
    parent: Any,
    old: dict[str, np.ndarray],
    root: Path,
    p: dict[str, Any],
    a: dict[str, Any],
    expected: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    additions = []
    for window in admitted_windows(p, a, expected):
        link = window["record"]["kick"]
        f = root / link["raw_path"]
        if (
            f.is_symlink()
            or not f.resolve().is_relative_to(root.resolve())
            or digest(f) != link["raw_hash"]
            or a["input_file_hashes"].get(str(f)) != link["raw_hash"]
        ):
            raise ValueError("exact confined independently audited recovery raw required")
        with np.load(f, allow_pickle=False) as archive:
            raw = {k: archive[k].copy() for k in archive.files}
        z, frames = executed_features(parent, raw, window["switch_control"])
        metadata = dict(
            arm=window["arm"],
            case=window["case"],
            switch_control=window["switch_control"],
            source_path=str(f),
            source_hash=link["raw_hash"],
            canonical_repeat=0,
            whole_source_rows=len(raw["NN_action"]),
            excluded_student_prefix_rows=window["switch_control"],
        )
        additions.append((z, frames, metadata))
    return append_unique(old, additions)


def load_interventional_data(
    root: Path, expected: str, policy: Path
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact closed interventional curation required")
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    if (
        r.get("status") != DATA_STATUS
        or d.get("schema") != "paid.interventional_dagger.TRAIN_features.v1"
        or r.get("declaration_hash") != digest(dp)
        or r.get("source_head") != d.get("source_head")
        or r.get("optimizer_updates") != 0
        or r.get("own_native_steps") != 0
        or any(
            o.get(k) != d.get(k)
            for o in (r,)
            for k in (
                "original_root",
                "original_hash",
                "pilot_root",
                "pilot_hash",
                "audit_root",
                "audit_hash",
            )
        )
        or any(
            o.get("bank") != "train" or o.get("Fresh_fit_rows") != 0 or o.get("DEV_fit_rows") != 0
            for o in (r, d)
        )
        or any(
            o.get(k) is not False
            for o in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("TRAIN-only explicit interventional extension contract required")
    parent, baseline, old = load_full_entry_data(
        Path(d["original_root"]), d["original_hash"], policy
    )
    pilot_root, audit_root = Path(d["pilot_root"]), Path(d["audit_root"])
    p, a = load_recovery_witness(
        pilot_root,
        d["pilot_hash"],
        audit_root,
        d["audit_hash"],
        original_hash=d["original_hash"],
        scene_hash=baseline["body_scene_hash"],
    )
    recomputed, groups = aggregate_handoff(parent, old, pilot_root, p, a, d["pilot_hash"])
    f = Path(r["feature_path"])
    if f.is_symlink() or f.parent != root or digest(f) != r["feature_hash"]:
        raise ValueError("confined exact interventional archive required")
    outcomes = root / "source-outcomes.json"
    if outcomes.is_symlink() or digest(outcomes) != r["source_outcomes_hash"]:
        raise ValueError("durable source success/failure memory changed")
    with np.load(f, allow_pickle=False) as archive:
        z = {k: archive[k].copy() for k in archive.files}
    if set(z) != set(recomputed) or any(
        z[k].dtype != v.dtype or not np.array_equal(z[k], v) for k, v in recomputed.items()
    ):
        raise ValueError(
            "interventional features/labels differ from own-state source recomputation"
        )
    if r["new_groups"] != groups or any(
        r[k] != baseline[k] for k in ("parent_hash", "parent_checkpoint_hash", "body_scene_hash")
    ):
        raise ValueError("immutable old motor lineage or new teacher window differs")
    if (
        r["rows"] != len(z["features"])
        or r["original_rows"] != len(old["features"])
        or r["protected_rows"] != int((~z["teacher"]).sum())
    ):
        raise ValueError("actual old/new row accounting differs")
    return parent, r, z
