"""Own-causal TRAIN features for novel first actions plus all acquired router skills.

Only fixed teacher seed0's NEW successful cases4,5 are fitted. Retain canonical
R2 first-success trajectories, not conflicting stylistic teachers for the same
initial state. No consumed Fresh actions, observations or labels are read.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("qualification-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("qualification-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--teacher-roots", type=Path, nargs=3, required=True)
    p.add_argument("--teacher-audit-roots", type=Path, nargs=3, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused curation and frozen clean source required")
    rp = a.qualification_root / "complete.json"
    if digest(rp) != a.qualification_hash:
        raise ValueError("exact closed qualification required")
    qr = json.loads(rp.read_text())
    qd = json.loads((a.qualification_root / "declaration.json").read_text())
    if (
        qr["status"] != "ROUTER_FRONTIER_PASS_PENDING_NEW_FRESH"
        or qr["primary_seed"] != 0
        or qr["all_three_seeds_pass"] is not True
        or qr["Fresh_rows_read"] != 0
    ):
        raise ValueError("fixed primary router and TRAIN/DEV frontier required")
    pins = dict(qr["input_file_hashes"])
    pins[str(rp)] = a.qualification_hash
    reports = []
    for seed, (root, audit) in enumerate(zip(a.teacher_roots, a.teacher_audit_roots, strict=True)):
        rp, ap = root / "complete.json", audit / "complete.json"
        r, ar = [json.loads(f.read_text()) for f in (rp, ap)]
        if (
            any((f / "failed.json").exists() for f in (root, audit))
            or r["status"] != "PAID_TRAIN_FIRST_CONTACT_TEACHER_CLOSED_PENDING_OWN_AUDIT"
            or ar["status"] != "PAID_FIRST_CONTACT_ALL_TRIAL_OWN_NN_IK_AND_NATIVE_AUDIT_PASS"
            or ar["source_hash"] != digest(rp)
            or r["config"]["seed_index"] != seed
            or ar["config"] != r["config"]
            or ar["selected_whole_repeats_exact"] is not True
            or ar["counts"]["episodes"] != 108
            or r["episodes"] != 108
            or ar["own_native_steps"] != r["actual_native_steps"]
            or ar["own_trajectory_neural_rows"] != r["actual_neural_rows"]
            or any(o["Fresh_rows_read"] != 0 or o["optimizer_updates"] != 0 for o in (r, ar))
            or any(
                o[k] is not False
                for o in (r, ar)
                for k in ("hardware_authorized", "promotion_authorized")
            )
        ):
            raise ValueError("all three complete own-NN/IK/native teacher audits required")
        for path, h in {
            **ar["input_file_hashes"],
            str(rp): digest(rp),
            str(ap): digest(ap),
        }.items():
            if path in pins and pins[path] != h:
                raise ValueError("teacher/router lineage conflict")
            pins[path] = h
        reports.append(r)
    fixed = reports[0]["searches"][0]
    if (
        fixed["pair"] != [4, 5]
        or fixed["baseline_rank"][4] != 0
        or fixed["selected_rank"][:6] != [1, 1, 1, 1, 2, 2]
    ):
        raise ValueError("fixed seed0 must physically rescue both novel first-contact cases")
    parent_root = Path(qd["pipelines"][0]) / "trained"
    prp = parent_root / "complete.json"
    training = json.loads(prp.read_text())
    parent_hash = digest(prp)
    if pins.get(str(prp)) != parent_hash or training["config"]["seed_index"] != 0:
        raise ValueError("same frozen seed0 competence-router parent required")
    old_root = Path(qd["pipelines"][0]) / "train-exam"
    new_root = Path(qd["new_exams"][0])
    roots = [old_root, new_root]
    sources = []
    for root in roots:
        rp = root / "complete.json"
        if pins.get(str(rp)) != digest(rp):
            raise ValueError("qualified actual acquired-success bank changed")
        sources.append(json.loads(rp.read_text()))
    old, new = sources
    if old["summary"] != dict(stable=16, quality_by_kick=[16, 14, 15]) or new["summary"] != dict(
        stable=20, quality_by_kick=[12, 19, 16]
    ):
        raise ValueError("all current old45 and new35 successor successes required")
    first_cases = [w["case"] for w in new["worlds"] if w["quality_by_kick"][0]]
    if first_cases != [1, 2, 3, 7, 9, 11, 13, 15, 16, 17, 18, 19]:
        raise ValueError("canonical12 successful first trajectories required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            schema="paid.full_entry.TRAIN_features.v1",
            source_head=a.expected_head,
            parent_root=str(parent_root),
            parent_hash=parent_hash,
            qualification_hash=a.qualification_hash,
            teacher_seed_index=0,
            novel_first_teacher_cases=[4, 5],
            canonical_R2_first_cases=first_cases,
            old_protected_by_kick=[16, 14, 15],
            new_protected_by_kick=[0, 19, 16],
            expected_trajectories=94,
            first_state_has_one_canonical_teacher=True,
            bank="train",
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
            features="own_normalized160+own_R2_hidden128",
            target="radian_difference_from_frozen_router_parent",
            final_R2_relative_bound_rad=0.2,
            internal_difference_bound_rad=0.4,
            runtime_case_lookup=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    rows = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_progressive_features import progressive_features
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=prp,
            training_receipt_hash=parent_hash,
            scene_hash=training["body_scene_hash"],
        )
        actor.requires_grad_(False).eval()
        xs, ys, ps, ts, gs, trajectories = [], [], [], [], [], []

        def add(root, link, *, case, skill, role, recorded, protect):
            nonlocal rows
            f = root / link["raw_path"]
            if (
                f.is_symlink()
                or not f.resolve().is_relative_to(root.resolve())
                or pins.get(str(f)) != link["raw_hash"]
                or digest(f) != link["raw_hash"]
            ):
                raise ValueError("independently audited exact causal raw required")
            with np.load(f, allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            x, y, prior = progressive_features(
                actor.base,
                actor.residual,
                raw,
                actor.action_scale,
                recorded_policy=recorded,
                protect=protect,
            )
            if protect and y.any():
                raise ValueError("protected action must exactly equal own router parent")
            n = len(x)
            group = len(xs)
            xs.append(x)
            ys.append(y)
            ps.append(prior)
            ts.append(np.full(n, not protect, np.bool_))
            gs.append(np.full(n, group, np.int32))
            trajectories.append(
                dict(
                    case=case,
                    skill=skill,
                    role=role,
                    source_path=str(f),
                    source_hash=link["raw_hash"],
                    recorded_policy=recorded,
                    rows=n,
                    start=rows,
                    stop=rows + n,
                )
            )
            rows += n

        for record in fixed["selected"][0]:
            if not record["stand"] or not record["stand"]["stability_observed"]:
                raise ValueError("novel teacher must have actually stable post-kick recovery")
            add(
                a.teacher_roots[0],
                record["kick"],
                case=record["case"],
                skill=0,
                role="novel_first_cartesian_teacher",
                recorded="base",
                protect=False,
            )
        for i in first_cases:
            add(
                new_root,
                new["worlds"][i]["kicks"][0]["result"],
                case=i,
                skill=0,
                role="canonical_first_R2_teacher",
                recorded="base",
                protect=False,
            )
        old_counts, new_counts = [0, 0, 0], [0, 0, 0]
        for root, source, cohort, counts in (
            (old_root, old, "original_acquired_success", old_counts),
            (new_root, new, "new_train_acquired_success", new_counts),
        ):
            for w in source["worlds"]:
                if not w["stable"]:
                    raise ValueError("failed body cannot supply acquired-success protection")
                for k, good in enumerate(w["quality_by_kick"]):
                    if good and (root == old_root or k > 0):
                        add(
                            root,
                            w["kicks"][k]["result"],
                            case=w["case"],
                            skill=k,
                            role=cohort,
                            recorded="base" if k == 0 else "parent",
                            protect=True,
                        )
                        counts[k] += 1
        if len(xs) != 94 or old_counts != [16, 14, 15] or new_counts != [0, 19, 16]:
            raise ValueError("complete protection and canonical teacher coverage required")
        x, y, prior, teacher, group = [np.concatenate(v) for v in (xs, ys, ps, ts, gs)]
        active = np.abs(y).max(axis=1) > 1e-6
        if np.any(active & ~teacher) or not active.any() or not (~active).any():
            raise ValueError("nontrivial first learning with exact acquired retention required")
        f = a.output / "full-entry-features.npz"
        np.savez_compressed(
            f, features=x, target=y, parent_delta=prior, teacher=teacher, group=group, active=active
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_FULL_ENTRY_TRAIN_FEATURES_CAUSALLY_RECOMPUTED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                feature_path=str(f),
                feature_hash=digest(f),
                parent_root=str(parent_root),
                parent_hash=parent_hash,
                parent_checkpoint_hash=training["checkpoint_hash"],
                body_scene_hash=training["body_scene_hash"],
                trajectories=trajectories,
                rows=rows,
                teacher_rows=int(teacher.sum()),
                protected_rows=int((~teacher).sum()),
                active_rows=int(active.sum()),
                positive_core_rows=int((np.abs(y).max(axis=1) > 0.005).sum()),
                negative_support_rows=int((~active).sum()),
                own_base_feature_rows=rows,
                own_parent_head_rows=rows,
                old_protected_by_kick=old_counts,
                new_protected_by_kick=new_counts,
                bank="train",
                Fresh_fit_rows=0,
                DEV_fit_rows=0,
                optimizer_updates=0,
                own_native_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                own_base_feature_rows=rows,
                own_native_steps=0,
                optimizer_updates=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
