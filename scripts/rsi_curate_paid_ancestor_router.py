"""Causal own-body router data from audited NEW TRAIN counterfactual teachers.

Retain current45 old TRAIN successes and every successful NEW TRAIN successor.
No consumed Fresh actions or labels enter fitting. No optimizer/physics here.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("review-root", "parent-root", "retention-root", "retention-audit-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("review-hash", "parent-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--study-shards", type=Path, nargs=4, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or digest(a.review_root / "complete.json") != a.review_hash
        or digest(a.parent_root / "complete.json") != a.parent_hash
    ):
        raise ValueError("unused curation and exact frozen sources required")
    review = json.loads((a.review_root / "complete.json").read_text())
    training = json.loads((a.parent_root / "complete.json").read_text())
    if (
        (a.review_root / "failed.json").exists()
        or review["status"] != "PAID_TRAIN20_ANCESTOR_COUNTERFACTUAL_ALL_AUDITS_CLOSED"
        or review["bank"] != "train"
        or review["Fresh_fit_rows"] != 0
        or not review["comparison"]["reproducible_teacher_exists"]
        or review["comparison"]["same_entry_ancestor_rescues"] != [15, 16, 17, 18]
        or training["config"]["protocol"] != "acquired-parent-progressive-contact-v1"
        or training["config"]["seed_index"] != 0
    ):
        raise ValueError(
            "qualified current parent and complete disjoint TRAIN teacher study required"
        )
    from rosclaw_soccer.rsi.paid_student_evidence import qualify_student_worlds

    rp, dp, ap = (
        a.retention_root / "complete.json",
        a.retention_root / "declaration.json",
        a.retention_audit_root / "complete.json",
    )
    original, od, oa = [json.loads(f.read_text()) for f in (rp, dp, ap)]
    qualify_student_worlds(
        original, od, oa, report_path=str(rp), report_hash=digest(rp), declaration_hash=digest(dp)
    )
    if (
        od["bank"] != "train"
        or od["student_hash"] != a.parent_hash
        or original["summary"] != dict(stable=16, quality_by_kick=[16, 14, 15])
    ):
        raise ValueError("all current45 TRAIN successes,not just earlier43,required")
    pins = dict(review["input_file_hashes"])
    for path, h in oa["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("parent/study lineage conflict")
        pins[path] = h
    for f in (a.review_root / "complete.json", a.parent_root / "complete.json", rp, dp, ap):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            schema="paid.ancestor_router.TRAIN_features.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            review_hash=a.review_hash,
            parent_hash=a.parent_hash,
            parent_root=str(a.parent_root),
            teacher_cases=[15, 16, 17, 18],
            original_protected_by_kick=[16, 14, 15],
            new_train_protect_successor_slots_only=True,
            bank="train",
            Fresh_fit_rows=0,
            inference_features="own_normalized160+own_R2_hidden128",
            runtime_case_lookup=False,
            alternatives="exact_current_parent_delta_or_exact_R2_zero_delta",
            bound_radians=0.2,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    nrows = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_progressive_features import progressive_features
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(p) for p in pins if p.endswith("/ckp/policy_30000.onnx"))
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=a.parent_root / "complete.json",
            training_receipt_hash=a.parent_hash,
            scene_hash=training["body_scene_hash"],
        )
        actor.requires_grad_(False).eval()
        features = []
        differences = []
        priors = []
        roles = []
        groups = []
        trajectories = []

        def add(row, root, *, case, skill, role):
            nonlocal nrows
            f = root / row["raw_path"]
            if (
                f.is_symlink()
                or not f.resolve().is_relative_to(root.resolve())
                or pins.get(str(f)) != row["raw_hash"]
                or digest(f) != row["raw_hash"]
                or row["body_failed"]
            ):
                raise ValueError("actual audited successful raw required")
            with np.load(f, allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            teacher = role == "ancestor_teacher"
            x, y, prior = progressive_features(
                actor.base,
                actor.residual,
                raw,
                actor.action_scale,
                recorded_policy="base" if teacher or skill == 0 else "parent",
                protect=not teacher,
            )
            if teacher and not np.array_equal(raw["NN_action"], raw["NN_nominal_action"]):
                raise ValueError("exact ownR2 teacher,not approximate cancellation orIK,required")
            if not teacher and np.count_nonzero(y):
                raise ValueError("protected old actions not actually parent")
            n = len(x)
            group = len(features)
            features.append(x)
            differences.append(y)
            priors.append(prior)
            roles.append(np.full(n, teacher, np.bool_))
            groups.append(np.full(n, group, np.int32))
            trajectories.append(
                dict(
                    case=case,
                    skill=skill,
                    role=role,
                    source_path=str(f),
                    source_hash=row["raw_hash"],
                    rows=n,
                    start=nrows,
                    stop=nrows + n,
                )
            )
            nrows += n

        for item in review["selected_physical_teachers"]:
            add(
                item["kick"],
                Path(item["teacher_root"]),
                case=item["case"],
                skill=2,
                role="ancestor_teacher",
            )
        original_count = [0, 0, 0]
        for i, w in enumerate(original["worlds"]):
            for k, good in enumerate(w["quality_by_kick"]):
                if good:
                    add(
                        w["kicks"][k]["result"],
                        a.retention_root,
                        case=i,
                        skill=k,
                        role="original_acquired_success",
                    )
                    original_count[k] += 1
        new_count = [0, 0, 0]
        for shard, root in enumerate(a.study_shards):
            path = root / "complete.json"
            if pins.get(str(path)) != digest(path):
                raise ValueError("shard not bound to qualified review")
            source = json.loads(path.read_text())
            if source["shard"] != shard or source["plan_hash"] != review["plan_hash"]:
                raise ValueError("wrong TRAIN shard")
            for group in source["groups"]:
                w = group["arms"]["grown"]
                if not w["stable"]:
                    continue
                # NEWTRAIN firstslot usedR2,not hypothetical parent; no false
                # parent-zero relabel. Candidate also retains firstslotR2.
                for k in (1, 2):
                    if len(w["quality_by_kick"]) > k and w["quality_by_kick"][k]:
                        add(
                            w["kicks"][k]["result"],
                            root / "grown",
                            case=group["case"],
                            skill=k,
                            role="new_train_acquired_success",
                        )
                        new_count[k] += 1
        if original_count != [16, 14, 15] or new_count != [0, 19, 12] or len(features) != 80:
            raise ValueError("all45 old,31 NEWTRAIN successes and4 teachers required")
        x, y, prior, teacher, group = map(
            np.concatenate, (features, differences, priors, roles, groups)
        )
        differing = np.any(prior != 0, axis=1)
        core = teacher & (np.abs(prior).max(axis=1) > 0.005)
        protected = (~teacher) & differing
        if not core.any() or not protected.any() or np.count_nonzero(y[~teacher]):
            raise ValueError("material teacher and genuine functional retention required")
        path = a.output / "router-features.npz"
        np.savez_compressed(
            path,
            features=x,
            parent_delta=prior,
            parent_difference=y,
            teacher=teacher,
            group=group,
            positive_core=core,
            protected_nonzero=protected,
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_ANCESTRAL_ROUTER_TRAIN_FEATURES_CAUSALLY_RECOMPUTED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                feature_path=str(path),
                feature_hash=digest(path),
                parent_hash=a.parent_hash,
                parent_root=str(a.parent_root),
                parent_checkpoint_hash=training["checkpoint_hash"],
                body_scene_hash=training["body_scene_hash"],
                rows=nrows,
                trajectories=trajectories,
                teacher_rows=int(teacher.sum()),
                protected_rows=int((~teacher).sum()),
                positive_core_rows=int(core.sum()),
                protected_nonzero_rows=int(protected.sum()),
                original_protected_by_kick=original_count,
                new_train_protected_by_kick=new_count,
                own_base_feature_rows=nrows,
                own_parent_head_rows=nrows,
                bank="train",
                Fresh_fit_rows=0,
                optimizer_updates=0,
                own_native_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                completed_feature_rows=nrows,
                own_native_steps=0,
                hardware_authorized=False,
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
