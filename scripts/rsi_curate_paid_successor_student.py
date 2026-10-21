"""Build TRAIN-only neural-contact examples from complete independently audited physics."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("transfer-root", "audit-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("transfer-hash", "audit-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    tp, ap = a.transfer_root / "complete.json", a.audit_root / "complete.json"
    if (
        a.output.exists()
        or digest(tp) != a.transfer_hash
        or digest(ap) != a.audit_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("exact audited transfer and unused frozen curation output required")
    r, review, td = [
        json.loads(path.read_text()) for path in (tp, ap, a.transfer_root / "declaration.json")
    ]
    if (
        r["status"] != "PAID_FROZEN_SUCCESSOR_TEACHER_TRAIN_DEV_TRANSFER_PENDING_AUDIT"
        or review["status"] != "PAID_SUCCESSOR_TRANSFER_ALL_NATIVE_NEURAL_IK_AUDIT_PASS"
        or review["input_file_hashes"].get(str(tp)) != a.transfer_hash
        or review["summaries"]["train"]["quality"] != 10
        or review["summaries"]["train"]["regression"]
        or review["counts"]["own_neural"] != r["actual_neural_rows"]
        or review["own_native_steps"] != r["actual_native_steps"]
    ):
        raise ValueError("independently qualified clean stable teacher transfer required")
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate

    root, audit = Path(td["bank_roots"]["train"]), Path(td["bank_audit_roots"]["train"])
    standing_gate(root, audit, variant=1, bank="train")
    parent, pd = [
        json.loads((root / name).read_text()) for name in ("complete.json", "declaration.json")
    ]
    pins = dict(review["input_file_hashes"])
    pins.update({str(tp): a.transfer_hash, str(ap): a.audit_hash})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            teacher_cases=[2, 3, 10, 11],
            protect_successes_only=True,
            protect_first16_second14_third6=True,
            bank="train",
            DEV_episodes_fit=0,
            base_wrapper_checkpoint_hash=pd["actor_checkpoint_hash"],
            base_wrapper_training_receipt_hash=pd["actor_training_receipt_hash"],
            features="own_normalized160+own_causal128",
            features_exclude_case_or_source_ids=True,
            hidden_states_copied_as_inputs=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    neural_rows = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_recovery_proof import unique_proof_path
        from rosclaw_soccer.rsi.paid_successor_features import successor_features
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(path) for path in pins if path.endswith("/ckp/policy_30000.onnx"))
        scene = next(
            h for path, h in pins.items() if path.endswith("/cached-standard/declared-scene.xml")
        )
        base_wrapper, _ = load_contact_residual_actor(
            policy,
            checkpoint=unique_proof_path(pins, pd["actor_checkpoint_hash"]),
            checkpoint_hash=pd["actor_checkpoint_hash"],
            training_receipt=unique_proof_path(pins, pd["actor_training_receipt_hash"]),
            training_receipt_hash=pd["actor_training_receipt_hash"],
            scene_hash=scene,
        )
        scale = torch.from_numpy(
            PaidResearchPolicy(policy, expected_hash=PUBLISHED_HASH, horizon=1)
            .parameters["action_scale"]
            .copy()
        )
        base = base_wrapper.base.eval()
        base.requires_grad_(False)
        declaration_rows = []
        features, targets, roles, groups = [], [], [], []

        def add(row, source_root, *, case, skill, teacher):
            nonlocal neural_rows
            path = source_root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(source_root.resolve())
                or pins.get(str(path)) != row["raw_hash"]
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError(
                    "independently pinned TRAIN raw, no DEV/fabricated labels required"
                )
            if row["body_failed"]:
                raise ValueError("failed-body trajectory cannot become positive supervision")
            with np.load(path, allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            x, y = successor_features(base, raw, scale, teacher=teacher)
            n = len(x)
            group = len(features)
            features.append(x)
            targets.append(y)
            roles.append(np.full(n, teacher, np.bool_))
            groups.append(np.full(n, group, np.int32))
            declaration_rows.append(
                dict(
                    case=case,
                    skill=skill,
                    role="teacher" if teacher else "protected_success",
                    source_path=str(path),
                    source_hash=row["raw_hash"],
                    rows=n,
                    start=neural_rows,
                    stop=neural_rows + n,
                )
            )
            neural_rows += n

        for i in (2, 3, 10, 11):
            w = r["results"]["train"]["rows"][i]
            if not w["quality"] or not w["stand"]["stability_observed"] or not w["teacher_applied"]:
                raise ValueError("actual selected teacher full kick-and-stand success required")
            add(w["kick"], a.transfer_root, case=i, skill=2, teacher=True)
        protected = [0, 0, 0]
        for i, world in enumerate(parent["worlds"]):
            for k in range(3):
                if world["quality_by_kick"][k]:
                    add(world["kicks"][k]["result"], root, case=i, skill=k, teacher=False)
                    protected[k] += 1
        if protected != [16, 14, 6] or len(features) != 40:
            raise ValueError("exact complete success-retention curriculum required")
        x, y, role, group = [np.concatenate(items) for items in (features, targets, roles, groups)]
        active = np.abs(y).max(axis=1) > 1e-6
        if not active.any() or active[~role].any() or np.abs(y[~role]).max() != 0:
            raise ValueError("nontrivial actual teacher and true zero-retention labels required")
        path = a.output / "train-features.npz"
        np.savez_compressed(path, features=x, target=y, teacher=role, group=group, active=active)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_SUCCESSOR_TRAIN_FEATURES_CAUSALLY_RECOMPUTED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                feature_path=str(path),
                feature_hash=digest(path),
                trajectories=declaration_rows,
                rows=neural_rows,
                CPU_feature_recompute_rows=neural_rows,
                teacher_rows=int(role.sum()),
                protected_rows=int((~role).sum()),
                active_rows=int(active.sum()),
                negative_support_rows=int((~active).sum()),
                protected_successes_by_kick=protected,
                base_wrapper_checkpoint_hash=pd["actor_checkpoint_hash"],
                base_wrapper_training_receipt_hash=pd["actor_training_receipt_hash"],
                body_scene_hash=scene,
                bank="train",
                DEV_episodes_fit=0,
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
                CPU_feature_recompute_rows=neural_rows,
                own_native_steps=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
