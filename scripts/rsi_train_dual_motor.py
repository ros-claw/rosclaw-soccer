"""Actual full-history plastic-RNN adaptation over a frozen skill-memory branch.

Supervised replay, NOT online PPO. Actual failed football outcomes keep their
labels. All learned-gain/retention claims wait for independent native exams.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("data-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("data-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--seed-index", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--warm-start-root", type=Path)
    p.add_argument("--warm-start-hash")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen real plastic-RNN training source required")

    clean()
    dp = a.data_root / "complete.json"
    if (
        a.output.exists()
        or dp.is_symlink()
        or digest(dp) != a.data_hash
        or (a.data_root / "failed.json").exists()
    ):
        raise ValueError("unused output and exact closed whole-history data required")
    d = json.loads(dp.read_text())
    dagger = d["status"] == "DUAL_MOTOR_ACTUAL_DAGGER_SEQUENCES_READY_NOT_TRAINED"
    if bool(a.warm_start_root) != bool(a.warm_start_hash) or dagger != bool(a.warm_start_root):
        raise ValueError(
            "new actual-state DAgger protocol requires exact previous learned warm start"
        )
    if (
        d["status"]
        not in (
            "DUAL_MOTOR_ACTUAL_HISTORY_SEQUENCES_READY_NOT_TRAINED",
            "DUAL_MOTOR_ACTUAL_DAGGER_SEQUENCES_READY_NOT_TRAINED",
        )
        or d["observation_width"] != 160
        or d["learner_hidden_width"] != 256
        or d["teacher_hidden_state_supplied_to_learner"] is not False
        or not d["whole_actual_sequences"]
        or d["DEV_fit_rows"] != 0
        or d["Fresh_fit_rows"] != 0
        or d["hardware_authorized"] is not False
        or d["promotion_authorized"] is not False
        or len(d["records"]) != (24 if dagger else 22)
    ):
        raise ValueError("actual complete sequences and own learner memory required")
    pins = dict(d["input_file_hashes"])
    pins[str(dp)] = a.data_hash
    if dagger:
        wp = a.warm_start_root / "complete.json"
        if pins.get(str(wp)) != a.warm_start_hash or digest(wp) != a.warm_start_hash:
            raise ValueError(
                "warm start must be the actual learner that generated admitted corrections"
            )
        wr = json.loads(wp.read_text())
        recovery_paths = [
            Path(x)
            for x, h in pins.items()
            if h == d["recovery_hash"] and x.endswith("/complete.json")
        ]
        if (
            len(recovery_paths) != 1
            or json.loads(recovery_paths[0].read_text())["training_hash"] != a.warm_start_hash
        ):
            raise ValueError("actual failed learner and new warm start must be identical")
        if wr["parent_lineage_hash"] != d["parent_lineage_hash"]:
            raise ValueError("same original frozen reference lineage required")
    paths = [
        Path(x)
        for x, h in pins.items()
        if h == d["parent_lineage_hash"] and x.endswith("/complete.json")
    ]
    if len(paths) != 1:
        raise ValueError("unique frozen canonical parent lineage required")
    parent_record = json.loads(paths[0].read_text())
    for record in d["records"]:
        path = a.data_root / record["raw_path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(a.data_root.resolve())
            or digest(path) != record["raw_hash"]
        ):
            raise ValueError("all exact complete training episodes must bind before optimization")
        pins[str(path)] = record["raw_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    config = dict(
        protocol="dual256.full_history_dagger_adaptation.v2"
        if dagger
        else "dual256.full_history_supervised_adaptation.v1",
        seed_index=a.seed_index,
        seed=51022 + a.seed_index,
        updates=256,
        lr=0.0001,
        old_positive_weight=4.0,
        temporal_error_weight=0.1,
        gradient_norm_limit=1.0,
        old_negative_bc_weight=0.0,
        full_unpadded_BPTT=True,
        plastic_rnn_and_action_network=True,
        freeze_normalizer=True,
        device="cuda:0",
        runtime_inference_device="cpu",
        selected_checkpoint="final_only",
        online_PPO=False,
    )
    if dagger:
        config.update(
            warm_start_receipt_hash=a.warm_start_hash,
            new_sequence_pool=4,
            corrections_supervised_only=True,
        )
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            data_hash=a.data_hash,
            parent_lineage_hash=d["parent_lineage_hash"],
            config=config,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    updates = 0
    offline_queries = 0
    try:
        import copy

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.dual_motor_checkpoint import load_dual_for_sim_audit
        from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        if not torch.cuda.is_available():
            raise ValueError("declared GPU training required, no fake CUDA fallback")
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        device = torch.device("cuda:0")
        teacher_path = next(
            Path(x)
            for x, h in pins.items()
            if h == d["teacher_hash"] and x.endswith("/complete.json")
        )
        teacher_decl = json.loads((teacher_path.parent / "declaration.json").read_text())
        _, policy, scene = open_paid_bench(pins, [teacher_decl["source_case"]])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(parent_record["checkpoint_path"]),
            checkpoint_hash=parent_record["checkpoint_hash"],
            training_receipt=paths[0],
            training_receipt_hash=d["parent_lineage_hash"],
            scene_hash=scene,
        )
        model = DualRecurrentMotor(parent).eval()
        if dagger:
            model, _ = load_dual_for_sim_audit(
                parent,
                parent_lineage_hash=d["parent_lineage_hash"],
                root=a.warm_start_root,
                expected_receipt_hash=a.warm_start_hash,
            )
            model.plastic.requires_grad_(True)
        frozen = {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()}
        initial = {k: v.detach().cpu().clone() for k, v in model.plastic.state_dict().items()}
        episodes = []
        for record in d["records"]:
            with np.load(a.data_root / record["raw_path"], allow_pickle=False) as z:
                episode = {k: torch.from_numpy(z[k].copy()).to(device) for k in z.files}
            n = len(episode["observations"])
            if (
                n != record["rows"]
                or episode["observations"].shape != (n, 160)
                or episode["targets"].shape != (n, 29)
                or not 1 <= n <= 400
            ):
                raise ValueError("one complete unpadded actual episode required")
            episodes.append(episode)
        positive = [i for i, r in enumerate(d["records"]) if r["role"] == "acquired_parent_skill"]
        novel = [
            i
            for i, r in enumerate(d["records"])
            if r["source_kind"] in ("relative_improvement", "actual_state_dagger")
        ]
        negative = [i for i, r in enumerate(d["records"]) if r["role"] == "negative_parent_task"]
        if len(positive) != 14 or len(novel) != (4 if dagger else 2) or len(negative) != 6:
            raise ValueError("fixed audited positive/negative/partial-skill cohort required")
        model.plastic.to(device).train()
        params = list(model.plastic_parameters())
        if len(params) != 16:
            raise ValueError("all eight RNN and eight action-network tensors must train")
        optimizer = torch.optim.Adam(params, lr=config["lr"])
        initial_path = a.output / "plastic-initial.pt"
        torch.save(
            dict(
                schema="soccer.dual_motor.plastic.v1",
                plastic_network=initial,
                optimizer_updates=0,
                parent_lineage_hash=d["parent_lineage_hash"],
                data_hash=a.data_hash,
            ),
            initial_path,
        )

        def losses(index):
            ep = episodes[index]
            state = (torch.zeros(2, 1, 128, device=device), torch.zeros(2, 1, 128, device=device))
            learned, _, _ = model.plastic.forward_with_features(ep["observations"][:, None], state)
            nominal, prior, scale = (
                ep["reference_nominal"],
                ep["reference_delta"],
                ep["action_scale"],
            )
            final = (prior + (learned[:, 0] - nominal) * scale).clamp(-0.2, 0.2)
            predicted = nominal + final / scale
            error = (predicted - ep["targets"]) * scale
            mse = error.square().mean()
            temporal = (error[1:] - error[:-1]).square().mean() if len(error) > 1 else mse * 0
            return mse + config["temporal_error_weight"] * temporal, mse

        history = []
        for step in range(256):
            optimizer.zero_grad(set_to_none=True)
            new_index = novel[step % len(novel)]
            old_index = positive[(step // 2) % 14]
            new_loss, new_mse = losses(new_index)
            old_loss, old_mse = losses(old_index)
            loss = new_loss + 4.0 * old_loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite recurrent adaptation objective")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(grad):
                raise ValueError("nonfinite recurrent gradient")
            optimizer.step()
            updates += 1
            row = dict(
                update=updates,
                new_sequence=new_index,
                old_sequence=old_index,
                new_MSE=float(new_mse.detach().cpu()),
                old_MSE=float(old_mse.detach().cpu()),
                loss=float(loss.detach().cpu()),
                gradient_norm=float(grad.detach().cpu()),
            )
            history.append(row)
            if updates % 16 == 0:
                print(json.dumps(row), flush=True)
                write_once(a.output / f"update{updates}.json", row)

        def cpu_tree(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu_tree(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu_tree(v) for v in value)
            return copy.deepcopy(value)

        final = cpu_tree(model.plastic.state_dict())
        if any(not torch.equal(final[k], initial[k]) for k in ("mean", "denominator")):
            raise ValueError("original normalization must remain immutable")
        changed = [k for k, v in final.items() if not torch.equal(v, initial[k])]
        if len(changed) != 16:
            raise ValueError("actual end-to-end recurrent and action updates required")
        if any(
            not torch.equal(v.detach().cpu(), frozen[k]) for k, v in parent.state_dict().items()
        ):
            raise ValueError("frozen competence memory changed")
        final_path = a.output / "plastic-final.pt"
        state = dict(
            schema="soccer.dual_motor.plastic.v1",
            plastic_network=final,
            optimizer=cpu_tree(optimizer.state_dict()),
            optimizer_updates=updates,
            parent_lineage_hash=d["parent_lineage_hash"],
            data_hash=a.data_hash,
            config=config,
            hidden_width=256,
            cpu_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state().cpu(),
        )
        torch.save(state, final_path)
        checked = torch.load(final_path, map_location="cpu", weights_only=True)
        if (
            checked["optimizer_updates"] != 256
            or len(checked["optimizer"]["state"]) != 16
            or any(float(v["step"]) != 256 for v in checked["optimizer"]["state"].values())
        ):
            raise ValueError("actual safe Adam checkpoint slots/steps required")
        model.plastic.to("cpu")
        model.eval().requires_grad_(False)
        replay = []
        for index, record in enumerate(d["records"]):
            with np.load(a.data_root / record["raw_path"], allow_pickle=False) as z:
                obs = z["observations"].copy()
                target = z["targets"].copy()
                scale = z["action_scale"].copy()
                reference = z["reference_nominal"] + z["reference_delta"] / scale
            private = model.initial_state()
            means = []
            for i, row in enumerate(obs):
                with torch.no_grad():
                    mean, private, e = model.forward_with_evidence(
                        torch.from_numpy(row).reshape(1, 1, 160), private
                    )
                offline_queries += 1
                if not np.array_equal(e["reference_mean"][0, 0].numpy(), reference[i]):
                    raise ValueError("plastic state contaminated frozen reference history")
                means.append(mean[0, 0].numpy().copy())
            predicted = np.stack(means)
            path = a.output / f"cpu-sequence{index}.npz"
            np.savez_compressed(path, means=predicted)
            replay.append(
                dict(
                    index=index,
                    role=record["role"],
                    rows=len(obs),
                    motor_MSE=float(np.mean(((predicted - target) * scale) ** 2)),
                    prediction_path=path.name,
                    prediction_hash=digest(path),
                )
            )
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_RECURRENT_MOTOR_ACTUALLY_TRAINED_NOT_PHYSICALLY_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                data_hash=a.data_hash,
                parent_lineage_hash=d["parent_lineage_hash"],
                declaration_hash=digest(a.output / "declaration.json"),
                initial_checkpoint_path=str(initial_path),
                initial_checkpoint_hash=digest(initial_path),
                checkpoint_path=str(final_path),
                checkpoint_hash=digest(final_path),
                config=config,
                optimizer_updates=updates,
                changed_parameter_tensors=changed,
                frozen_parent_unchanged=True,
                cpu_offline_decisions=offline_queries,
                cpu_replay=replay,
                physical_steps_executed=0,
                online_RL_claim=False,
                old_retention_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                actual_optimizer_updates=updates,
                cpu_offline_decisions=offline_queries,
                physical_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
