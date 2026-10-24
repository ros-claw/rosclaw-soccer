"""Fit ONLY a new neural refinement head on qualified actual histories, NOT PPO."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("feature-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("feature-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seed", type=int, choices=(115023, 115024, 115025), required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    from rosclaw_soccer.rsi.refinement_motor_learning import (
        FEATURE_STATUS,
        KIND,
        SCHEMA,
        STATUS,
        refinement_config,
        refinement_schedule,
        validate_refinement_checkpoint,
    )
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_output_capacity import require_output_capacity

    fp = a.feature_root / "complete.json"
    if (
        a.output.exists()
        or a.feature_root.is_symlink()
        or fp.is_symlink()
        or digest(fp) != a.feature_hash
        or (a.feature_root / "failed.json").exists()
    ):
        raise ValueError("exact whole own-feature proof and unused training output required")
    require_output_capacity(a.output, 2 * 1024**3)
    f = json.loads(fp.read_text())
    if (
        f["status"] != FEATURE_STATUS
        or f["zero_refinement_bit_identity"] is not True
        or f["all_shared_parent_traces_match"] is not True
        or f["parent_weights_unchanged"] is not True
        or f["teacher_hidden_state_executed"] is not False
        or any(
            f[k] != 0
            for k in (
                "physical_steps",
                "student_optimizer_updates",
                "DEV_fit_rows",
                "Fresh_fit_rows",
            )
        )
        or any(
            f[k] is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError(
            "actual own private-state replay and zero identity required before fitting"
        )
    cfg = refinement_config(a.seed, a.device)
    schedule = refinement_schedule(f["records"], a.seed)
    pins = dict(f["input_file_hashes"])
    pins[str(fp)] = a.feature_hash
    _, bank = receipt(pins, f["bank_hash"])
    _, parent_receipt = receipt(pins, f["parent_training_hash"])
    if (
        bank["fit_plan"] != f["fit_plan"]
        or len(bank["records"]) != len(f["records"])
        or parent_receipt["checkpoint_hash"] != f["parent_checkpoint_hash"]
        or parent_receipt["head_optimizer_steps"] != 2048
        or parent_receipt["config"]["seed"] != 105025
    ):
        raise ValueError("same qualified whole bank and acquired105025 parent required")
    for r, raw in zip(f["records"], bank["records"], strict=True):
        path = a.feature_root / r["feature_path"]
        if (
            path.is_symlink()
            or path.resolve().parent != a.feature_root.resolve()
            or r["index"] != raw["index"]
            or r["role"] != raw["role"]
            or r["rows"] != raw["rows"]
            or r["source_raw_hash"] != raw["raw_hash"]
            or r["borrowed_hidden_state"] is not False
            or r["zero_refinement_bit_identity"] is not True
        ):
            raise ValueError("every whole unchanged-history feature archive required")
        pins[str(path)] = r["feature_hash"]
    commitments = dict(
        feature_hash=a.feature_hash,
        bank_hash=f["bank_hash"],
        parent_training_hash=f["parent_training_hash"],
        parent_checkpoint_hash=f["parent_checkpoint_hash"],
        parent_lineage_hash=f["parent_lineage_hash"],
        schedule_hash=str(hash_json(schedule)),
    )
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            config=cfg,
            preflight_only=a.preflight_only,
            fit_plan=f["fit_plan"],
            **commitments,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    steps, history = 0, []
    try:
        import copy
        import os

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        _, parent = open_motor(
            pins,
            f["parent_training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="consolidated-adapter-v1",
        )
        torch.manual_seed(a.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = RefinementMotorAdapter(parent)
        initial = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        frozen = {k: v.detach().cpu().clone() for k, v in model.parent.state_dict().items()}
        model.to(a.device).train()
        parameters = model.plastic_parameters()
        opt = torch.optim.Adam(parameters, lr=cfg["lr"])
        if opt.state_dict()["state"] or len(parameters) != 4:
            raise ValueError("four genuinely new head parameters and fresh Adam required")
        episodes = {}
        scale = parent.memory.parent.get_buffer("action_scale").cpu().numpy()
        for r in f["records"]:
            with np.load(a.feature_root / r["feature_path"], allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            shapes = dict(
                features=(r["rows"], 288),
                parent_delta=(r["rows"], 29),
                target_delta=(r["rows"], 29),
                loss_weights=(r["rows"],),
                action_scale=(29,),
            )
            if (
                set(raw) != set(shapes)
                or any(
                    raw[k].shape != v or raw[k].dtype != np.float32 or not np.isfinite(raw[k]).all()
                    for k, v in shapes.items()
                )
                or not np.array_equal(raw["action_scale"], scale)
                or (raw["loss_weights"] <= 0).any()
                or abs(raw["parent_delta"]).max() > 0.2
                or abs(raw["target_delta"]).max() > 0.20002
            ):
                raise ValueError(
                    "exact finite whole features and unchanged total target bounds required"
                )
            episodes[r["index"]] = {
                k: torch.from_numpy(raw[k]).to(a.device)
                for k in ("features", "parent_delta", "target_delta", "loss_weights")
            }
        with torch.no_grad():
            for ep in episodes.values():
                _, total = model.adapt_features(ep["features"], ep["parent_delta"])
                if not torch.equal(total, ep["parent_delta"]):
                    raise ValueError("new head is not initially exact acquired-parent identity")
        device = dict(
            logical=a.device,
            actual_head_device=str(next(model.head.parameters()).device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu_name=torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
        )
        for indices in schedule:
            tensors = {
                k: torch.cat([episodes[i][k] for i in indices])
                for k in ("features", "parent_delta", "target_delta")
            }
            weights = torch.cat(
                [
                    episodes[i]["loss_weights"]
                    / episodes[i]["loss_weights"].sum()
                    / (4 if col < 8 else 2)
                    for col, i in enumerate(indices)
                ]
            )
            end_old = sum(len(episodes[i]["features"]) for i in indices[:4])
            end_current = end_old + sum(len(episodes[i]["features"]) for i in indices[4:8])
            opt.zero_grad(set_to_none=True)
            _, total = model.adapt_features(tensors["features"], tensors["parent_delta"])
            error = ((total - tensors["target_delta"]) / cfg["loss_unit_rad"]).square().mean(
                -1
            ) * weights
            old_loss, current_loss, teacher_loss = (
                error[:end_old].sum(),
                error[end_old:end_current].sum(),
                error[end_current:].sum(),
            )
            loss = (
                cfg["old_weight"] * old_loss
                + cfg["current_weight"] * current_loss
                + cfg["teacher_weight"] * teacher_loss
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite physical-teacher imitation loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                parameters, cfg["grad_norm_limit"], error_if_nonfinite=True
            )
            if any(p.grad is not None for p in model.parent.parameters()):
                raise ValueError("frozen whole motor received training gradients")
            if a.preflight_only:
                if opt.state_dict()["state"] or not (grad > 0):
                    raise ValueError(
                        "actual nonzero first gradient without optimizer update required"
                    )
                check_pins(pins)
                clean(a.expected_head)
                write_once(
                    a.output / "complete.json",
                    dict(
                        status="WHOLE_MOTOR_REFINEMENT_PREFLIGHT_LOSS_BACKWARD_NOT_TRAINED",
                        source_head=a.expected_head,
                        input_file_hashes=pins,
                        config=cfg,
                        device=device,
                        loss=float(loss.detach()),
                        gradient_norm=float(grad),
                        new_head_steps=0,
                        fresh_adam_state_empty=True,
                        real_qualified_histories_used=True,
                        physical_steps=0,
                        hardware_authorized=False,
                        promotion_authorized=False,
                        goal_complete=False,
                    ),
                )
                return
            opt.step()
            steps += 1
            report = dict(
                step=steps,
                indices=indices,
                loss=float(loss.detach()),
                old_loss=float(old_loss.detach()),
                current_loss=float(current_loss.detach()),
                teacher_loss=float(teacher_loss.detach()),
                gradient_norm=float(grad),
            )
            history.append(report)
            if steps == 1 or steps % 64 == 0:
                write_once(a.output / f"update{steps}.json", report)
                print(json.dumps(report), flush=True)
        model.cpu().eval().requires_grad_(False)
        if any(not torch.equal(v, frozen[k]) for k, v in model.parent.state_dict().items()):
            raise ValueError("acquired head or old recurrent memory changed")

        def cpu_tree(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(cpu_tree(v) for v in value)
            return copy.deepcopy(value)

        checkpoint = dict(
            schema=SCHEMA,
            config=cfg,
            head=dict(model.head.state_dict()),
            initial_head=initial,
            head_optimizer=cpu_tree(opt.state_dict()),
            new_head_steps=steps,
            **commitments,
        )
        validate_refinement_checkpoint(
            checkpoint, dict(model.head.state_dict()), config=cfg, commitments=commitments
        )
        cp = a.output / "refinement-final.pt"
        torch.save(checkpoint, cp)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=STATUS,
                checkpoint_kind=KIND,
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                config=cfg,
                fit_plan=f["fit_plan"],
                **commitments,
                checkpoint_path=str(cp),
                checkpoint_hash=digest(cp),
                new_head_steps=steps,
                acquired_head_steps_unchanged=2048,
                memory_steps_unchanged=dict(actor=112, critic=48),
                parent_weights_unchanged=True,
                history=history,
                device=device,
                physical_steps_executed=0,
                physical_retention_proven=False,
                online_RL_claim=False,
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
                actual_new_head_steps=steps,
                physical_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
