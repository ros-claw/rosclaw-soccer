"""Causal growth data on CURRENT motor memory, with all historical anchors protected.

New labels are actual executed successful recoveries, not advice on failed states.
The consumed neighbourhood development bank is explicitly reclassified as TRAIN.
Original DEV16 and every Fresh bank remain excluded from feature fitting.
"""

import copy
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.paid_canonical_behavior import digest, quality_score
from rosclaw_soccer.rsi.paid_neighborhood_sources import (
    load_current_student_prefix,
    load_neighborhood_plan,
)
from rosclaw_soccer.rsi.paid_neighborhood_study import current_prefix_config

DATA_STATUS = "PAID_GROWING_MOTOR_MEMORY_TRAIN_CAUSALLY_RECOMPUTED"
SCHEMA = "paid.growing_motor_memory.TRAIN_features.v1"
SOURCES = ("handoff80", "audit80", "handoff96", "audit96")


def witnesses(
    identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    sources: dict[str, Any] = {}
    pins: dict[str, str] = {}

    def merge(values: dict[str, str]) -> None:
        for path, h in values.items():
            if path in pins and pins[path] != h:
                raise ValueError("conflicting growing-memory provenance")
            pins[path] = h

    for name in SOURCES:
        root = Path(identity[name + "_root"])
        path = root / "complete.json"
        if (
            path.is_symlink()
            or digest(path) != identity[name + "_hash"]
            or (root / "failed.json").exists()
        ):
            raise ValueError("exact closed current-prefix recovery sources required")
        r = json.loads(path.read_text())
        if any(r.get(k) is not False for k in ("hardware_authorized", "promotion_authorized")):
            raise ValueError("SIM-only growing-memory sources required")
        sources[name] = r
        merge(r["input_file_hashes"])
        pins[str(path)] = digest(path)
    first = sources["handoff80"]
    _, plan, pp = load_neighborhood_plan(Path(first["plan_root"]), first["plan_hash"])
    merge(pp)
    prefix, pp = load_current_student_prefix(
        Path(first["prefix_root"]),
        first["prefix_hash"],
        Path(first["prefix_audit_root"]),
        first["prefix_audit_hash"],
        plan,
        first["plan_hash"],
    )
    merge(pp)
    for cut in (80, 96):
        r, a = sources["handoff" + str(cut)], sources["audit" + str(cut)]
        root = Path(identity["handoff" + str(cut) + "_root"])
        dp = root / "declaration.json"
        d = json.loads(dp.read_text())
        if (
            dp.is_symlink()
            or digest(dp) != r["declaration_hash"]
            or a["input_file_hashes"].get(str(dp)) != digest(dp)
            or d.get("base_stand") is not True
            or d.get("teacher_reoptimized") is not False
            or r["status"] != "PAID_CONTACT_NEIGHBORHOOD32_ARM_CLOSED_PENDING_AUDIT"
            or a["status"] != "PAID_CONTACT_NEIGHBORHOOD32_OWN_NN_IK_ALL_NATIVE_AUDIT_PASS"
            or a["source_hash"] != identity["handoff" + str(cut) + "_hash"]
            or any(
                v["config"] != current_prefix_config(cut)
                or v.get("current_student_prefix") is not True
                or v["handoff_control"] != cut
                or v["model_hash"] != plan["specifications"]["models"]["learned_memory"]["hash"]
                or v["body_scene_hash"] != plan["body_scene_hash"]
                for v in (r, d)
            )
            or any(
                r.get(k) != v
                for k, v in d.items()
                if k not in ("base_stand", "teacher_reoptimized")
            )
            or any(
                r[k] != first[k]
                for k in (
                    "plan_root",
                    "plan_hash",
                    "prefix_root",
                    "prefix_hash",
                    "prefix_audit_root",
                    "prefix_audit_hash",
                )
            )
            or any(v["arm"] != "fixed_successful_teacher" for v in (r, a, d))
            or r["episodes"] != 64
            or a["counts"]["episodes"] != 64
            or r["repeats_exact"] is not True
            or a["repeats_exact"] is not True
            or a["own_native_steps"] != r["actual_native_steps"]
            or a["own_trajectory_neural_rows"] != r["actual_neural_rows"]
            or any(v.get("optimizer_updates") != 0 or v.get("Fresh_rows_read") != 0 for v in (r, a))
            or [(v["case"], v["repeat"]) for v in r["records"]]
            != [(i, j) for i in range(32) for j in range(2)]
            or [(v["case"], v["repeat"]) for v in a["outcomes"]]
            != [(i, j) for i in range(32) for j in range(2)]
        ):
            raise ValueError(
                "complete executed and independently audited current-prefix recovery required"
            )
        for row, own in zip(r["records"], a["outcomes"], strict=True):
            if row["quality"] != own["quality"] or row["healthy_success"] != own["healthy_success"]:
                raise ValueError("current-prefix outcome differs from independent physical witness")
        pins[str(dp)] = digest(dp)
    return plan, prefix, sources, pins


def choose_behaviors(prefix: dict[str, Any], sources: dict[str, Any]) -> list[dict[str, Any]]:
    choices = []
    for i in range(32):
        previous = prefix["records"][2 * i : 2 * i + 2]
        if [r["healthy_success"] for r in previous] not in ([True, True], [False, False]):
            raise ValueError("both complete source repeats must agree")
        options = []
        for cut in (80, 96):
            rows = sources["handoff" + str(cut)]["records"][2 * i : 2 * i + 2]
            score = quality_score(rows)
            options.append(
                dict(
                    case=i,
                    cut=cut,
                    score=list(score) if score is not None else None,
                    record=rows[0],
                )
            )
        eligible = [v for v in options if v["score"] is not None]
        if previous[0]["healthy_success"]:
            selected = dict(case=i, kind="protected_current_success", cut=None, record=previous[0])
        elif eligible:
            best = max(eligible, key=lambda v: (*v["score"], -v["cut"]))
            selected = dict(**best, kind="executed_current_state_recovery")
        else:
            selected = None
        choices.append(
            dict(
                case=i,
                selected=selected,
                alternatives=options,
                original_current_success=previous[0]["healthy_success"],
            )
        )
    if not any(
        c["selected"] and c["selected"]["kind"] == "executed_current_state_recovery"
        for c in choices
    ):
        raise ValueError("no new physically supported recoveries, no empty growth fitting job")
    return choices


def protect_historical_anchors(parent: Any, old: dict[str, Any]) -> tuple[dict[str, Any], float]:
    import numpy as np
    import torch

    z = {k: v.copy() for k, v in old.items()}
    discrepancy = 0.0
    with torch.no_grad():
        for i, x in enumerate(z["features"]):
            p = parent.residual(torch.from_numpy(x.copy()).reshape(1, 288))[0].numpy().copy()
            discrepancy = max(
                discrepancy, float(np.abs(p - (old["parent_delta"][i] + old["target"][i])).max())
            )
            z["parent_delta"][i] = p
    if discrepancy > 2.1e-6:
        raise ValueError("current motor no longer preserves the historical learned anchor targets")
    # These are CURRENT policy retention anchors, not relabelled old expert actions.
    z["target"].fill(0)
    z["teacher"].fill(False)
    z["active"].fill(False)
    return z, discrepancy


def aggregate_behaviors(
    z: dict[str, Any], additions: list[tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    result = {k: [v] for k, v in z.items()}
    known = {
        x.tobytes(): (z["target"][i], z["parent_delta"][i]) for i, x in enumerate(z["features"])
    }
    groups = []
    cursor = len(z["features"])
    group = int(z["group"].max()) + 1
    for arrays, meta in additions:
        keep = []
        for i, x in enumerate(arrays["features"]):
            key = x.tobytes()
            y, p = arrays["target"][i], arrays["parent_delta"][i]
            if key in known:
                if not np.allclose(y, known[key][0], rtol=0, atol=2e-6) or not np.array_equal(
                    p, known[key][1]
                ):
                    raise ValueError(
                        "growing labels conflict with protected/selected identical context"
                    )
            else:
                known[key] = (y, p)
                keep.append(i)
        n = len(keep)
        if not n:
            raise ValueError("selected new behavior contributes no new observed contexts")
        protected = meta["kind"] == "protected_current_success"
        for k in ("features", "target", "parent_delta"):
            result[k].append(arrays[k][keep])
        result["teacher"].append(np.full(n, not protected, np.bool_))
        result["active"].append(np.abs(arrays["target"][keep]).max(axis=1) > 1e-6)
        result["group"].append(np.full(n, group, np.int32))
        groups.append(
            dict(
                **meta,
                start=cursor,
                stop=cursor + n,
                rows=n,
                source_frames=keep,
                duplicate_rows_skipped=len(arrays["features"]) - n,
                group=group,
            )
        )
        cursor += n
        group += 1
    if cursor > 50000:
        raise ValueError("bounded complete growth dataset required")
    return {k: np.concatenate(v) for k, v in result.items()}, groups


def reconstruct(
    identity: dict[str, Any], policy: Path
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np

    from rosclaw_soccer.rsi.paid_canonical_behavior import load_canonical_data
    from rosclaw_soccer.rsi.paid_compact_diagnosis import compact_own_trace
    from rosclaw_soccer.rsi.paid_interventional_data import successful_behavior_features
    from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student

    plan, prefix, sources, _ = witnesses(identity)
    spec = plan["specifications"]["models"]["learned_memory"]
    rp = Path(spec["root"]) / "complete.json"
    s = json.loads(rp.read_text())
    parent, _ = load_paid_student(
        policy,
        checkpoint=Path(s["checkpoint_path"]),
        checkpoint_hash=s["checkpoint_hash"],
        training_receipt=rp,
        training_receipt_hash=spec["hash"],
        scene_hash=plan["body_scene_hash"],
    )
    unused, base, old = load_canonical_data(Path(plan["data_root"]), plan["data_hash"], policy)
    del unused
    if (
        s["data_hash"] != plan["data_hash"]
        or s["config"].get("canonical_behavior") is not True
        or s["config"].get("control") is not False
    ):
        raise ValueError("exact learned current memory and its immutable old dataset required")
    z, discrepancy = protect_historical_anchors(parent, old)
    choices = choose_behaviors(prefix, sources)
    additions = []
    for item in choices:
        selected = item["selected"]
        if selected is None:
            continue
        cut = selected["cut"]
        root = Path(
            sources["handoff80"]["prefix_root"]
            if cut is None
            else identity["handoff" + str(cut) + "_root"]
        )
        link = selected["record"]["kick"]
        path = root / link["raw_path"]
        audit_root = Path(
            sources["handoff80"]["prefix_audit_root"]
            if cut is None
            else identity["audit" + str(cut) + "_root"]
        )
        audit = json.loads((audit_root / "complete.json").read_text())
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or digest(path) != link["raw_hash"]
            or audit["input_file_hashes"].get(str(path)) != link["raw_hash"]
        ):
            raise ValueError("exact independently audited successful own behavior required")
        with np.load(path, allow_pickle=False) as a:
            raw = {k: a[k].copy() for k in a.files}
        if raw["NN_obs"][0, 125:154].any() or not np.array_equal(
            raw["NN_obs"][1:, 125:154], raw["NN_action"][:-1]
        ):
            raise ValueError("whole previous ACTUAL action chain required")
        if cut is None:
            trace = compact_own_trace(parent, raw)
            if not np.array_equal(raw["NN_action"], raw["NN_nominal_action"]):
                raise ValueError("protected own policy must be actually executed")
            arrays = dict(
                features=trace["features"],
                parent_delta=trace["delta"],
                target=np.zeros_like(trace["delta"]),
            )
        else:
            arrays, frames = successful_behavior_features(parent, raw, cut)
            if frames != list(range(len(raw["NN_action"]))):
                raise ValueError("full successful own-history behavior required")
        additions.append(
            (
                arrays,
                dict(
                    case=item["case"],
                    kind=selected["kind"],
                    cut=cut,
                    source_path=str(path),
                    source_hash=link["raw_hash"],
                    source_rows=len(raw["NN_action"]),
                    prefix_is_expert=False,
                    student_prefix_rows=0 if cut is None else cut,
                ),
            )
        )
    z, groups = aggregate_behaviors(z, additions)
    manifest = dict(
        choices=choices,
        added_trajectories=groups,
        historical_anchor_rows=len(old["features"]),
        historical_ideal_deviation_rad=discrepancy,
        old_data_hash=plan["data_hash"],
        old_historical_trajectories=copy.deepcopy(base["trajectories"]),
        old_artifacts_unchanged=True,
        parent_root=spec["root"],
        parent_hash=spec["hash"],
        parent_checkpoint_hash=s["checkpoint_hash"],
        body_scene_hash=plan["body_scene_hash"],
        consumed_neighborhood_development_reclassified_as_TRAIN=True,
        original_DEV16_fit_rows=0,
        Fresh_fit_rows=0,
    )
    return parent, base, old, z, manifest


def load_growth_data(
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
        raise ValueError("exact closed growing motor memory dataset required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r["status"] != DATA_STATUS
        or any(v.get("schema") != SCHEMA or v.get("bank") != "train" for v in (r, d))
        or r["declaration_hash"] != digest(dp)
        or r["source_head"] != d["source_head"]
        or any(r[k] != d[k] for n in SOURCES for k in (n + "_root", n + "_hash"))
        or r["input_file_hashes"] != d["input_file_hashes"]
        or any(
            v.get("Fresh_fit_rows") != 0 or v.get("original_DEV16_fit_rows") != 0 for v in (r, d)
        )
        or any(
            v.get(k) is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
        or r["optimizer_updates"] != 0
        or r["own_native_steps"] != 0
    ):
        raise ValueError("explicit TRAIN consumption and current-parent growth contract required")
    parent, _, _, recomputed, manifest = reconstruct(d, policy)
    path, memory = Path(r["feature_path"]), root / "growth-behavior-memory.json"
    if (
        path.is_symlink()
        or path.parent != root
        or digest(path) != r["feature_hash"]
        or memory.is_symlink()
        or digest(memory) != r["behavior_memory_hash"]
    ):
        raise ValueError("confined exact growth features and behavior manifest required")
    with np.load(path, allow_pickle=False) as a:
        z = {k: a[k].copy() for k in a.files}
    if (
        set(z) != set(recomputed)
        or any(z[k].dtype != v.dtype or not np.array_equal(z[k], v) for k, v in recomputed.items())
        or json.loads(memory.read_text()) != manifest
        or r["manifest"] != manifest
        or any(
            r[k] != manifest[k]
            for k in ("parent_root", "parent_hash", "parent_checkpoint_hash", "body_scene_hash")
        )
        or r["rows"] != len(z["features"])
        or r["active_rows"] != int(z["active"].sum())
        or r["protected_rows"] != int((~z["active"]).sum())
    ):
        raise ValueError("growing data differ from complete causal source reconstruction")
    return parent, r, z


def growth_exam_candidate(
    root: Path, expected: str, plan: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any], dict[str, str]]:
    """Read-only exact candidate admission for the old consumed32 fixture."""
    from rosclaw_soccer.rsi.paid_neighborhood_study import neighborhood_config

    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact closed growing candidate required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    control = r["config"].get("control")
    if (
        type(control) is not bool
        or r["config"].get("growing_memory") is not True
        or r["status"]
        != (
            "PAID_GROWING_CARDINAL_MEMORY_ZERO_CONTROL_NOT_PHYSICALLY_QUALIFIED"
            if control
            else "PAID_GROWING_CARDINAL_MEMORY_FIT_NOT_PHYSICALLY_QUALIFIED"
        )
        or digest(dp) != r["declaration_hash"]
        or any(
            v["body_scene_hash"] != plan["body_scene_hash"]
            or v["parent_hash"] != plan["specifications"]["models"]["learned_memory"]["hash"]
            for v in (r, d)
        )
        or any(
            v.get(k) is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("explicit same-body current-parent growing candidate required")
    cp = Path(r["checkpoint_path"])
    if cp.is_symlink() or cp.parent != root or digest(cp) != r["checkpoint_hash"]:
        raise ValueError("confined growing-memory checkpoint required")
    pins = dict(r["input_file_hashes"])
    pins[str(rp)], pins[str(dp)], pins[str(cp)] = expected, digest(dp), r["checkpoint_hash"]
    config = neighborhood_config()
    config.update(
        protocol="paid.contact_neighborhood.growing_motor_candidate.v3",
        source_design_protocol=neighborhood_config()["protocol"],
        candidate_hash=expected,
        frozen_parent_hash=r["parent_hash"],
        candidate_zero_control=control,
        consumed_for_training=True,
        heldout_generalization_claim=False,
    )
    return dict(root=str(root), hash=expected), config, pins
