"""Train one recurrent actor on audited complete successful two-skill worlds.

This offline experiment has no simulation imports or hardware authority. The
case IDs choose TRAIN data only; inference receives only the original160 obs.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_search_paid_live_contact import qualified_successor_bootstrap

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-root", type=Path, required=True)
    p.add_argument("--bank-hash", required=True)
    p.add_argument("--bootstrap-audit-root", type=Path, required=True)
    p.add_argument("--bootstrap-audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--variant", type=int, choices=range(4), required=True)
    p.add_argument("--epochs", type=int, choices=(1, 256), default=256)
    p.add_argument(
        "--curriculum", choices=("paired-world", "protected-successor"), default="paired-world"
    )
    p.add_argument("--zero-root", type=Path)
    p.add_argument("--zero-hash")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    bank_path = a.bank_root / "complete.json"
    audit_path = a.bootstrap_audit_root / "complete.json"
    if (
        a.output.exists()
        or digest(bank_path) != a.bank_hash
        or digest(audit_path) != a.bootstrap_audit_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output, frozen source and exact audited paired bank required")
    bank, audit = [json.loads(f.read_text()) for f in (bank_path, audit_path)]
    if (
        bank["schema"] != "soccer.audited_successor_sequence_bank.v1"
        or bank["complete_successful_course_coverage"] != 16
        or bank["first_neural_rows"] != 3810
        or bank["second_neural_rows"] != 3810
        or [c["case"] for c in bank["courses"]] != list(range(16))
        or bank["body_reset_between_skills"] is not False
        or bank["runtime_case_lookup_forbidden"] is not True
        or bank["oracle_training_data_only"] is not True
        or bank["single_policy_16of16_proven"] is not False
    ):
        raise ValueError("all sixteen complete training worlds, not a runtime oracle required")
    base_root = Path(bank["base_root"])
    cp, bp = base_root / "live-distilled-epoch256.pt", base_root / "complete.json"
    xp = Path(bank["bootstrap_exam_root"]) / "complete.json"
    base, exam = [json.loads(f.read_text()) for f in (bp, xp)]
    qualified = qualified_successor_bootstrap(
        exam, audit, report_path=xp, report_hash=digest(xp), training_hash=digest(bp)
    )
    if digest(cp) != bank["base_checkpoint_hash"] or qualified["checkpoint_hash"] != digest(cp):
        raise ValueError("all stable complete-world learned initialization required")
    pins = dict(bank["input_file_hashes"])
    for f, h in (
        (bank_path, a.bank_hash),
        (audit_path, a.bootstrap_audit_hash),
        (bp, digest(bp)),
        (cp, digest(cp)),
        (xp, digest(xp)),
    ):
        if str(f) in pins and pins[str(f)] != h:
            raise ValueError("inconsistent source commitment")
        pins[str(f)] = h
    specialist = a.curriculum == "protected-successor"
    courses = bank["courses"]
    positive_cases = list(range(16))
    if specialist:
        if a.zero_root is None or a.zero_hash is None:
            raise ValueError("specialist needs actual learned-prefix zero-bank identity")
        zp = a.zero_root / "complete.json"
        if digest(zp) != a.zero_hash or pins.get(str(zp)) != a.zero_hash:
            raise ValueError("zero-bank must already belong to the independently audited bank")
        zero = json.loads(zp.read_text())
        from scripts.rsi_search_paid_foot_contact import contact_rank

        if zero["first_kick_reference"] != "learned_safe_successor" or len(zero["baseline"]) != 16:
            raise ValueError("actual matching learned-entry live baseline required")
        positive_cases = [i for i in range(16) if i not in (10, 14)]
        courses = []
        for i, original in enumerate(bank["courses"]):
            w = qualified["continuous"]["results"][i]
            first = dict(w["first"], path=str(xp.parent / w["first"]["raw_path"]))
            second = original["second"]
            role = "positive_contact_demonstration"
            if i not in positive_cases:
                row = zero["baseline"][i]
                if row["body_failed"] or contact_rank([row, row])[1] > 0:
                    raise ValueError("negative anchors must be healthy but TASK_FAILED")
                second = dict(row, path=str(a.zero_root / row["raw_path"]))
                role = "body_stability_only_TASK_STILL_FAILED"
            elif original["origin"].startswith("historical"):
                raise ValueError("no historical prefix mismatches in specialist positive data")
            for row in (first, second):
                if str(row["path"]) in pins and pins[row["path"]] != row["raw_hash"]:
                    raise ValueError("conflicting original trajectory identity")
                pins[row["path"]] = row["raw_hash"]
            courses.append(dict(original, first=first, second=second, role=role))
    elif a.zero_root is not None or a.zero_hash is not None:
        raise ValueError("unrelated zero bank may not modify original paired-world protocol")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    config = dict(
        protocol="complete-successor-sequences-v1",
        variant=a.variant,
        seed=20261028 + (a.variant if a.variant < 3 else 0),
        epochs=a.epochs,
        lr=0.0001,
        first_weight=3.0,
        active_weight=5.0,
        other_weight=1.0,
        max_grad_norm=1.0,
        worlds=16,
        first_unique_rows=3810,
        second_unique_rows=3810,
        minibatches_per_epoch=4,
        worlds_per_minibatch=4,
        memory_handoff="carry" if a.variant < 3 else "reset",
        full_world_BPTT=True,
        teacher_hidden_states_not_inputs=True,
        normalizer_frozen=True,
        DEV_episodes_used=0,
        observation_gap="no fabricated neural rows; memory constant during hold/drop",
        loss_units="joint_target_radians_squared",
    )
    if specialist:
        config.update(
            protocol="protected-successor-specialist-v1",
            seed=20261031 + (a.variant if a.variant < 3 else 0),
            first_weight=3.0 if a.variant < 3 else 1.0,
            memory_handoff="reset",
            full_world_BPTT=False,
            independent_skill_rehearsal=True,
            fixed_qualified_entry_actor=True,
            positive_successor_cases=positive_cases,
            negative_body_only_cases=[10, 14],
            negative_body_weight=3.0,
        )
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            config=config,
            input_file_hashes=pins,
            bank_hash=a.bank_hash,
            base_checkpoint_hash=digest(cp),
            base_training_receipt_hash=digest(bp),
            initialization_exam_hash=digest(xp),
            initialization_audit_hash=a.bootstrap_audit_hash,
            original160_observation_contract_unchanged=True,
            worlds=courses,
            **(dict(zero_bank_hash=a.zero_hash) if specialist else {}),
            no_case_id_policy_inputs=True,
            no_runtime_teacher=True,
            CPU_whole_world_exam_required=True,
            hardware_authorized=False,
            promotion_authorized=False,
            commercial_publicity_authorized=False,
        ),
    )
    counts = dict(first_forward_rows=0, second_forward_rows=0, optimizer_updates=0)
    try:
        import numpy as np
        import onnx
        import torch

        from rosclaw_soccer.rsi.paid_live_distilled_actor import load_live_distilled_actor
        from rosclaw_soccer.rsi.paid_successor_sequences import paired_arrays, paired_forward
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if (
            not visible.startswith("GPU-")
            or "," in visible
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or torch.cuda.device_count() != 1
        ):
            raise ValueError("one UUID-isolated offline CUDA device required")
        props = torch.cuda.get_device_properties(0)
        if "GPU-" + str(props.uuid) != visible:
            raise ValueError("actual GPU identity differs")
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
            raise ValueError("exact single body recipe required")
        model, _ = load_live_distilled_actor(
            policy,
            checkpoint=cp,
            checkpoint_hash=digest(cp),
            training_receipt=bp,
            training_receipt_hash=digest(bp),
            scene_hash=digest(scenes[0]),
        )
        model = model.to("cuda:0").train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        normalization = {k: model.state_dict()[k].detach().clone() for k in ("mean", "denominator")}
        metadata = {x.key: x.value for x in onnx.load(policy).metadata_props}
        scale = torch.tensor(
            [float(x) for x in metadata["action_scale"].split(",")],
            dtype=torch.float32,
            device="cuda:0",
        )
        worlds = []
        for c in courses:
            if any(pins.get(row["path"]) != row["raw_hash"] for row in (c["first"], c["second"])):
                raise ValueError("actual paired raw trajectories must already be sealed")
            with (
                np.load(c["first"]["path"], allow_pickle=False) as first,
                np.load(c["second"]["path"], allow_pickle=False) as second,
            ):
                data = paired_arrays(first, second)
            if specialist:
                # Every primary entry is EXACTLY the fixed actor's actual
                # prefix, unlike the previous heterogeneous whole-world bank.
                original_first = bank["courses"][c["case"]]["first"]
                if c["case"] in positive_cases:
                    with (
                        np.load(original_first["path"], allow_pickle=False) as teacher_first,
                        np.load(c["first"]["path"], allow_pickle=False) as entry_first,
                    ):
                        if set(teacher_first.files) != set(entry_first.files) or any(
                            not np.array_equal(teacher_first[k], entry_first[k])
                            for k in entry_first.files
                        ):
                            raise ValueError(
                                "positive teacher must use EXACT fixed entry trajectory"
                            )
            boundary = int(data.pop("boundary"))
            worlds.append(
                dict(
                    boundary=boundary,
                    second_weight=3.0 if specialist and c["case"] not in positive_cases else 1.0,
                    **{k: torch.from_numpy(v).to("cuda:0") for k, v in data.items()},
                )
            )
        if (
            sum(w["boundary"] for w in worlds) != 3810
            or sum(len(w["obs"]) - w["boundary"] for w in worlds) != 3810
        ):
            raise ValueError("all actual complete first and second rows required")

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu(v) for v in value)
            return value

        def checkpoint(epoch):
            path = a.output / f"successor-epoch{epoch}.pt"
            if path.exists():
                raise ValueError("checkpoint overwrite forbidden")
            state = dict(
                schema="soccer.paid_successor_specialist.v1"
                if specialist
                else "soccer.paid_successor_sequences.v1",
                epoch=epoch,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                config=config,
                counts=counts.copy(),
                source_actor_hash=PUBLISHED_HASH,
                body_scene_hash=digest(scenes[0]),
                base_checkpoint_hash=digest(cp),
                base_training_receipt_hash=digest(bp),
                bank_hash=a.bank_hash,
                network=cpu(model.state_dict()),
                optimizer=cpu(optimizer.state_dict()),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            )
            torch.save(state, path)
            if torch.load(path, map_location="cpu", weights_only=True)["schema"] != state["schema"]:
                raise ValueError("safe typed checkpoint reload failed")
            return digest(path)

        initial_hash = checkpoint(0)
        compute = dict(
            UUID=visible,
            name=props.name,
            torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,
            TF32=False,
            CUDA_control_qualified=False,
            physics_steps=0,
        )
        write_once(a.output / "compute.json", compute)
        for epoch in range(1, a.epochs + 1):
            order = rng.permutation(16).tolist()
            losses = []
            for group in range(4):
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for i in order[group * 4 : (group + 1) * 4]:
                    w = worlds[i]
                    n = w["boundary"]
                    pred, _ = paired_forward(
                        model, w["obs"], n, carry=config["memory_handoff"] == "carry"
                    )
                    counts["first_forward_rows"] += n
                    counts["second_forward_rows"] += len(w["obs"]) - n
                    weight = torch.where(
                        w["active"], config["active_weight"], config["other_weight"]
                    )
                    errors = ((pred - w["action"]) * scale).square().mean(dim=1)
                    loss = (
                        config["first_weight"] * (errors[:n] * weight[:n]).sum() / weight[:n].sum()
                        + w["second_weight"] * (errors[n:] * weight[n:]).sum() / weight[n:].sum()
                    ) / 4
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite full-world loss")
                    loss.backward()
                    total += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                counts["optimizer_updates"] += 1
                losses.append(total)
            if any(
                not torch.equal(model.state_dict()[k], v) for k, v in normalization.items()
            ) or any(not torch.isfinite(x).all() for x in model.parameters()):
                raise ValueError("normalizer mutation or nonfinite network")
            row = dict(
                epoch=epoch, order=order, mean_loss=float(np.mean(losses)), counts=counts.copy()
            )
            if epoch % 64 == 0 or epoch == a.epochs:
                row["checkpoint_hash"] = checkpoint(epoch)
            write_once(a.output / f"epoch{epoch}.json", row)
            if epoch % 16 == 0 or epoch == a.epochs:
                print(json.dumps(row), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_SUCCESSOR_SPECIALIST_TRAINED_NOT_PHYSICALLY_EXAMINED"
                if specialist
                else "PAID_SUCCESSOR_SEQUENCE_TRAINED_NOT_PHYSICALLY_EXAMINED",
                input_file_hashes=pins,
                source_head=a.expected_head,
                config=config,
                counts=counts,
                compute=compute,
                initial_checkpoint_hash=initial_hash,
                final_checkpoint_hash=digest(a.output / f"successor-epoch{a.epochs}.pt"),
                native_steps=0,
                CPU_whole_world_exam_required=True,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            ),
        )
    except Exception as error:
        write_once(a.output / "failed.json", dict(error=repr(error), counts=counts, native_steps=0))
        raise


if __name__ == "__main__":
    main()
