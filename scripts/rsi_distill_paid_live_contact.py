"""Learn live-body contact examples while rehearsing the qualified v0 actor.

Positive supervision is filtered by the declared primitive quality rule, but
all sixteen outcomes stay in the evidence and subsequent whole-world exams.
CUDA computes offline gradients only; it is not qualified for motor control.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_search_paid_foot_contact import contact_rank

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--teacher-hash", required=True)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--exam-root", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--variant", type=int, choices=range(4), required=True)
    p.add_argument("--epochs", type=int, choices=(1, 256), default=256)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tp, ap, bp, xp = [
        root / "complete.json" for root in (a.teacher_root, a.audit_root, a.base_root, a.exam_root)
    ]
    if (
        a.output.exists()
        or digest(tp) != a.teacher_hash
        or digest(ap) != a.audit_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output, exact completed teacher/audit and frozen source required")
    teacher, audit, base, exam = [json.loads(path.read_text()) for path in (tp, ap, bp, xp)]
    if (
        teacher["status"] != "PAID_LIVE_BODY_CONTACT_TEACHER_RESEARCH_NOT_PROMOTED"
        or audit["status"] != "PAID_LIVE_CONTACT_PRIMITIVE_CEM_AUDIT_PASS"
        or audit["input_file_hashes"].get(str(tp)) != a.teacher_hash
        or audit["raw_episode_count"] != 672
        or not audit["all_selected_full_repeats_and_branches_exact"]
        or not audit["all_generation_artifacts_sealed"]
        or teacher["input_file_hashes"].get(str(bp)) != digest(bp)
        or teacher["input_file_hashes"].get(str(xp)) != digest(xp)
    ):
        raise ValueError("actual fully audited live-body teacher and original v0 exam required")
    cp = a.base_root / "distilled-epoch256.pt"
    pins = {
        **teacher["input_file_hashes"],
        str(tp): a.teacher_hash,
        str(ap): a.audit_hash,
        str(bp): digest(bp),
        str(xp): digest(xp),
        str(cp): base["final_checkpoint_hash"],
    }
    if (
        len(teacher["final"]) != 16
        or len(exam["candidates"][0]["results"]["old16"]["results"]) != 16
    ):
        raise ValueError("full unfiltered sixteen-case cohorts required")
    qualified = [i for i, row in enumerate(teacher["final"]) if contact_rank([row, row])[1] > 0]
    if len(qualified) != 12:
        raise ValueError(
            "this declared audited curriculum requires twelve positive labels and all16outcomes"
        )
    expert = []
    for i in qualified:
        row = teacher["final"][i]
        path = a.teacher_root / row["raw_path"]
        if not path.resolve().is_relative_to(a.teacher_root.resolve()):
            raise ValueError("teacher raw path escaped the data root")
        pins[str(path)] = row["raw_hash"]
        expert.append(
            dict(case=i, source=row["source"], path=str(path), neural_rows=row["neural_rows"])
        )
    rehearsal = []
    for i, row in enumerate(exam["candidates"][0]["results"]["old16"]["results"]):
        path = a.exam_root / f"candidate0/old16/case{i}-repeat0.npz"
        pins[str(path)] = row["raw_hash"]
        rehearsal.append(
            dict(case=i, source=row["source"], path=str(path), neural_rows=row["neural_rows"])
        )
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    config = dict(
        epochs=a.epochs,
        variant=a.variant,
        seed=20261022 + (a.variant if a.variant < 3 else 0),
        lr=0.0001,
        rehearsal=1.0 if a.variant < 3 else 3.0,
        freeze_rnn=False,
        teacher_episodes=12,
        rehearsal_episodes=16,
        exam_continuous_cases=16,
        DEV_episodes_used=0,
        active_weight=5.0,
        other_weight=1.0,
        max_grad_norm=1.0,
        full_episode_BPTT=True,
        teacher_hidden_states_not_inputs=True,
        normalizer_frozen=True,
        teacher_minibatches_per_epoch=3,
        rehearsal_minibatch_sizes=[6, 5, 5],
        loss_units="joint_target_radians_squared",
        unique_teacher_neural_rows=sum(x["neural_rows"] for x in expert),
        unique_rehearsal_neural_rows=sum(x["neural_rows"] for x in rehearsal),
    )
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            config=config,
            expert=expert,
            rehearsal=rehearsal,
            all16_teacher_outcomes=teacher["final"],
            excluded_positive_label_cases=[i for i in range(16) if i not in qualified],
            qualification_rule="original physical contact quality; no threshold changes",
            all16_continuous_cases_remain_in_exam=True,
            base_checkpoint_hash=pins[str(cp)],
            base_training_receipt_hash=digest(bp),
            teacher_hash=a.teacher_hash,
            teacher_audit_hash=a.audit_hash,
            CUDA_used_for="offline gradient computation only",
            CUDA_control_qualified=False,
            CPU_neural_only_full_sequence_exam_required=True,
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    counts = dict(teacher_forward_rows=0, rehearsal_forward_rows=0, optimizer_updates=0)
    try:
        import numpy as np
        import onnx
        import torch

        from rosclaw_soccer.rsi.paid_distilled_actor import load_distilled_actor
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if (
            not visible.startswith("GPU-")
            or "," in visible
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or torch.cuda.device_count() != 1
        ):
            raise ValueError("one UUID-isolated GPU and deterministic CUDA workspace required")
        props = torch.cuda.get_device_properties(0)
        if "GPU-" + str(props.uuid) != visible:
            raise ValueError("actual GPU UUID differs")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        rng = np.random.default_rng(config["seed"])
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        scenes = [Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml")]
        if not scenes or len({digest(f) for f in scenes}) != 1:
            raise ValueError("identical qualified standard-ball body recipe required")
        model, _ = load_distilled_actor(
            policy,
            checkpoint=cp,
            checkpoint_hash=pins[str(cp)],
            training_receipt=bp,
            training_receipt_hash=digest(bp),
            scene_hash=digest(scenes[0]),
        )
        model = model.to("cuda:0").train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        normalization = {k: model.state_dict()[k].detach().clone() for k in ("mean", "denominator")}
        metadata = {
            x.key: x.value for x in onnx.load(policy, load_external_data=False).metadata_props
        }
        scale = torch.tensor(
            [float(x) for x in metadata["action_scale"].split(",")],
            device="cuda:0",
            dtype=torch.float32,
        )

        def load(entries, teacher_labels):
            arrays = []
            for entry in entries:
                with np.load(entry["path"], allow_pickle=False) as z:
                    obs, action = z["NN_obs"].copy(), z["NN_action"].copy()
                    active = (
                        np.any(z["PLAN_pose_offset"] != 0, axis=1)
                        if teacher_labels
                        else np.zeros(len(obs), bool)
                    )
                    if not teacher_labels and not np.array_equal(action, z["NN_nominal_action"]):
                        raise ValueError("rehearsal must be unassisted v0 neural actions")
                if (
                    obs.shape != (entry["neural_rows"], 160)
                    or action.shape != (len(obs), 29)
                    or not np.isfinite(obs).all()
                    or not np.isfinite(action).all()
                ):
                    raise ValueError("complete finite160/29 real sequences required")
                arrays.append(
                    dict(
                        obs=torch.from_numpy(obs).to("cuda:0"),
                        action=torch.from_numpy(action).to("cuda:0"),
                        active=torch.from_numpy(active).to("cuda:0"),
                    )
                )
            return arrays

        teachers, anchors = load(expert, True), load(rehearsal, False)
        compute = dict(
            UUID=visible,
            name=props.name,
            torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,
            TF32=False,
            distributed_allreduce=False,
            physics_steps=0,
            CUDA_control_qualified=False,
        )
        write_once(a.output / "compute.json", compute)

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu(v) for v in value)
            return value

        def checkpoint(epoch):
            path = a.output / f"live-distilled-epoch{epoch}.pt"
            if path.exists():
                raise ValueError("checkpoint overwrite forbidden")
            state = dict(
                schema="soccer.paid_live_distilled.v1",
                epoch=epoch,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                config=config,
                counts=counts.copy(),
                source_actor_hash=PUBLISHED_HASH,
                body_scene_hash=digest(scenes[0]),
                base_checkpoint_hash=pins[str(cp)],
                base_training_receipt_hash=digest(bp),
                teacher_hash=a.teacher_hash,
                teacher_audit_hash=a.audit_hash,
                network=cpu(model.state_dict()),
                optimizer=cpu(optimizer.state_dict()),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
                commercial_publicity_authorized=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            torch.save(state, path)
            if torch.load(path, map_location="cpu", weights_only=True)["schema"] != state["schema"]:
                raise ValueError("safe typed checkpoint reload failed")
            return digest(path)

        initial_hash = checkpoint(0)
        history = []

        def loss_for(episode, teacher_labels):
            zero = (
                torch.zeros(2, 1, 128, device="cuda:0"),
                torch.zeros(2, 1, 128, device="cuda:0"),
            )
            predicted, _ = model(episode["obs"][:, None], zero)
            key = "teacher_forward_rows" if teacher_labels else "rehearsal_forward_rows"
            counts[key] += len(episode["obs"])
            weight = torch.where(episode["active"], config["active_weight"], config["other_weight"])
            return (
                ((predicted[:, 0] - episode["action"]) * scale).square() * weight[:, None]
            ).sum() / (29 * weight.sum())

        for epoch in range(1, a.epochs + 1):
            teacher_order, anchor_order = rng.permutation(12).tolist(), rng.permutation(16).tolist()
            batch_losses = []
            for group in range(3):
                optimizer.zero_grad(set_to_none=True)
                t_ids = teacher_order[group * 4 : (group + 1) * 4]
                lo, hi = ((0, 6), (6, 11), (11, 16))[group]
                r_ids = anchor_order[lo:hi]
                total = 0.0
                for i in t_ids:
                    loss = loss_for(teachers[i], True) / len(t_ids)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite teacher loss")
                    loss.backward()
                    total += float(loss.detach())
                for i in r_ids:
                    loss = config["rehearsal"] * loss_for(anchors[i], False) / len(r_ids)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite rehearsal loss")
                    loss.backward()
                    total += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                counts["optimizer_updates"] += 1
                batch_losses.append(total)
            if any(
                not torch.equal(model.state_dict()[k], v) for k, v in normalization.items()
            ) or any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("normalizer mutation or nonfinite network")
            row = dict(
                epoch=epoch,
                teacher_order=teacher_order,
                rehearsal_order=anchor_order,
                mean_minibatch_loss=float(np.mean(batch_losses)),
                counts=counts.copy(),
            )
            if epoch % 64 == 0 or epoch == a.epochs:
                row["checkpoint_hash"] = checkpoint(epoch)
            write_once(a.output / f"epoch{epoch}.json", row)
            history.append(row)
            if epoch % 16 == 0 or epoch == a.epochs:
                print(json.dumps(row), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_LIVE_DISTILLATION_TRAINED_NOT_PHYSICALLY_EXAMINED",
                input_file_hashes=pins,
                source_head=a.expected_head,
                config=config,
                history=history,
                counts=counts,
                initial_checkpoint_hash=initial_hash,
                final_checkpoint_hash=digest(a.output / f"live-distilled-epoch{a.epochs}.pt"),
                compute=compute,
                native_steps=0,
                CPU_neural_only_full_sequence_exam_required=True,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
                commercial_publicity_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), counts=counts, native_steps=0, hardware_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
