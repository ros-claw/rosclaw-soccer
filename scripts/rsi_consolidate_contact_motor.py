"""Distill qualified physical strokes into the actual recurrent neural motor.

One final-only seed of a preregistered three-seed offline experiment. No
physics here; CPU whole-history predictions are not physical qualification.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("data-producer-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--seed", type=int, choices=(65023, 65024, 65025), required=True)
    p.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    p.add_argument("--failed-predecessor-root", type=Path)
    p.add_argument("--failed-predecessor-hash")
    p.add_argument("--stage", choices=("initial", "own-state-correction"), default="initial")
    a = p.parse_args()
    from rosclaw_soccer.rsi.contact_consolidation import (
        CORRECTION_STATUS,
        STATUS,
        consolidation_config,
    )
    from rosclaw_soccer.rsi.contact_consolidation_data import (
        BANK_STATUS,
        CORRECTION_BANK_STATUS,
        consolidation_pools,
    )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen consolidation source required")

    clean()
    correction = a.stage == "own-state-correction"
    config = consolidation_config(a.seed, a.device, stage=a.stage)
    initial_actor, final_actor = config["initial_actor_steps"], config["final_actor_steps"]
    dp = a.data_root / "complete.json"
    data = json.loads(dp.read_text())
    if (
        a.output.exists()
        or dp.is_symlink()
        or (a.data_root / "failed.json").exists()
        or data["status"] != (CORRECTION_BANK_STATUS if correction else BANK_STATUS)
        or data["source_head"] != a.data_producer_head
        or len(data["records"]) != (110 if correction else 65)
        or data["teacher_hidden_state_supplied_to_learner"] is not False
        or any(
            data[k] != 0
            for k in ("physical_steps", "optimizer_updates", "Fresh_fit_rows", "DEV_fit_rows")
        )
        or any(
            data[k] is not False
            for k in ("hardware_authorized", "promotion_authorized", "student_success_claim")
        )
    ):
        raise ValueError("complete audited bank65 and unused output required")
    pins = dict(data["input_file_hashes"])
    pins[str(dp)] = digest(dp)
    declaration = a.data_root / "declaration.json"
    if digest(declaration) != data["declaration_hash"]:
        raise ValueError("exact data declaration required")
    pins[str(declaration)] = data["declaration_hash"]

    def receipt(h):
        paths = [
            Path(x) for x, value in pins.items() if value == h and x.endswith("/complete.json")
        ]
        if len(paths) != 1:
            raise ValueError("unique exact warm lineage required")
        return paths[0], json.loads(paths[0].read_text())

    warm_hash = data["warm_student_hashes"][str(a.seed)] if correction else data["warm_start_hash"]
    wp, warm = receipt(warm_hash)
    if (
        warm["cumulative_optimizer_steps"] != initial_actor
        or warm["cumulative_critic_optimizer_steps"] != 48
        or (correction and warm["config"]["seed"] != a.seed)
    ):
        raise ValueError("exact independently learned predecessor and persistent critic48 required")
    witness_path, _ = receipt(data["context_witness_hash"])
    witness = json.loads((witness_path.parent / "declaration.json").read_text())
    for row in data["records"]:
        path = a.data_root / row["raw_path"]
        if (
            path.is_symlink()
            or path.parent.resolve() != a.data_root.resolve()
            or digest(path) != row["raw_hash"]
        ):
            raise ValueError("confined complete sequence required")
        pins[str(path)] = row["raw_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    anchors, teachers, repairs = consolidation_pools(data["records"], correction=correction)
    if bool(a.failed_predecessor_root) != bool(a.failed_predecessor_hash):
        raise ValueError("exact preserved failure binding required")
    if a.failed_predecessor_root is not None:
        if correction:
            raise ValueError(
                "initial container-check recovery cannot be reused as a new learning phase retry"
            )
        failed = a.failed_predecessor_root / "failed.json"
        prior_decl = a.failed_predecessor_root / "declaration.json"
        if failed.is_symlink() or digest(failed) != a.failed_predecessor_hash:
            raise ValueError("exact preserved zero-update failure required")
        failure, prior = [json.loads(x.read_text()) for x in (failed, prior_decl)]
        if (
            failure["offline_supervised_steps"] != 0
            or failure["cpu_offline_decisions"] != 0
            or failure["error"]
            != "ValueError('exact warm network/moments/counters/rates before schedule required')"
            or prior["config"] != config
            or prior["data_hash"] != digest(dp)
            or prior["warm_start_hash"] != warm_hash
        ):
            raise ValueError(
                "recovery fixes only typed network-container comparison before any update"
            )
        pins[str(failed)] = a.failed_predecessor_hash
        pins[str(prior_decl)] = digest(prior_decl)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            config=config,
            source_head=a.expected_head,
            input_file_hashes=pins,
            data_hash=digest(dp),
            warm_start_hash=warm_hash,
            parent_lineage_hash=data["parent_lineage_hash"],
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    steps, queries, history = 0, 0, []
    try:
        import copy
        import random

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.contact_consolidation import sequence_loss
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import (
            same_training_state,
            validate_adam_slots,
        )
        from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
        from scripts.rsi_stroke_search_common import open_motor

        torch.manual_seed(a.seed)
        torch.use_deterministic_algorithms(True)
        rng = random.Random(a.seed)
        bench, model = open_motor(
            pins,
            warm_hash,
            {0: dict(case=witness["source_case"])},
            checkpoint_kind="contact-consolidated" if correction else "ar1-actor-critic",
        )
        warm_path = Path(warm["checkpoint_path"])
        if digest(warm_path) != warm["checkpoint_hash"]:
            raise ValueError("exact validated warm checkpoint required")
        warm_state = torch.load(warm_path, map_location="cpu", weights_only=True)
        critic = MotorCritic().eval()
        critic.load_state_dict(warm_state["critic"], strict=True)
        model.plastic.requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                dict(params=list(model.plastic_parameters()), lr=0.0001),
                dict(params=list(critic.parameters()), lr=0.001),
            ]
        )
        optimizer.load_state_dict(copy.deepcopy(warm_state["optimizer"]))
        if not same_training_state(
            optimizer.state_dict(), warm_state["optimizer"]
        ) or not same_training_state(
            dict(model.plastic.state_dict()), warm_state["plastic_network"]
        ):
            raise ValueError("exact warm network/moments/counters/rates before schedule required")
        first = a.output / "consolidation-initial.pt"
        torch.save(copy.deepcopy(warm_state), first)
        frozen = copy.deepcopy(model.parent.state_dict())
        model.train()
        model.to(a.device)
        # Reload after device migration so owned actor Adam slots follow actual parameters.
        optimizer.load_state_dict(copy.deepcopy(warm_state["optimizer"]))
        optimizer.param_groups[0]["lr"] = config["actor_lr"]
        critic.requires_grad_(False)
        episodes = []
        for row in data["records"]:
            with np.load(a.data_root / row["raw_path"], allow_pickle=False) as z:
                episodes.append({k: torch.from_numpy(z[k].copy()).to(a.device) for k in z.files})
        for _step in range(config["steps"]):
            teacher = rng.choice(teachers)
            repair = rng.choice(repairs) if correction else None
            old = rng.sample(anchors, 4)
            optimizer.zero_grad(set_to_none=True)
            imitation = sequence_loss(model.plastic, episodes[teacher])
            retention = sum(sequence_loss(model.plastic, episodes[i]) for i in old) / 4
            loss = imitation + 2 * retention
            if correction:
                repair_loss = sequence_loss(model.plastic, episodes[repair])
                loss = loss + config["own_state_repair_weight"] * repair_loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite consolidation loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.plastic_parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            steps += 1
            entry = dict(
                step=steps,
                teacher_index=teacher,
                anchor_indices=old,
                loss=float(loss.detach()),
                imitation=float(imitation.detach()),
                retention=float(retention.detach()),
                gradient_norm=float(grad),
            )
            if correction:
                entry.update(repair_index=repair, repair_loss=float(repair_loss.detach()))
            history.append(entry)
            write_once(a.output / f"update{steps}.json", entry)
            print(json.dumps(entry), flush=True)
        model.cpu().eval().requires_grad_(False)

        def cpu_tree(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().clone()
            if isinstance(x, dict):
                return {k: cpu_tree(v) for k, v in x.items()}
            if isinstance(x, list):
                return [cpu_tree(v) for v in x]
            if isinstance(x, tuple):
                return tuple(cpu_tree(v) for v in x)
            return copy.deepcopy(x)

        final_opt = cpu_tree(optimizer.state_dict())
        validate_adam_slots(
            final_opt, list(model.plastic.parameters()) + list(critic.parameters()), final_actor, 48
        )
        critic_ids = final_opt["param_groups"][1]["params"]
        if (
            any(
                not same_training_state(final_opt["state"][i], warm_state["optimizer"]["state"][i])
                for i in critic_ids
            )
            or not same_training_state(dict(critic.state_dict()), warm_state["critic"])
            or not same_training_state(model.parent.state_dict(), frozen)
        ):
            raise ValueError("critic weights/Adam or frozen reference changed during consolidation")
        final = cpu_tree(model.plastic.state_dict())
        changed = [
            k for k, v in final.items() if not torch.equal(v, warm_state["plastic_network"][k])
        ]
        if len(changed) != 16:
            raise ValueError("all16 actual end-to-end recurrent parameter tensors must learn")
        last = a.output / "consolidation-final.pt"
        torch.save(
            dict(
                schema="soccer.dual_motor.contact_correction.v2"
                if correction
                else "soccer.dual_motor.contact_consolidation.v1",
                plastic_network=final,
                critic=cpu_tree(critic.state_dict()),
                optimizer=final_opt,
                optimizer_updates=dict(actor=final_actor, critic=48),
                config=config,
                warm_start_hash=warm_hash,
                parent_lineage_hash=data["parent_lineage_hash"],
            ),
            last,
        )
        replay = []
        for row in data["records"]:
            with np.load(a.data_root / row["raw_path"], allow_pickle=False) as z:
                obs, targets, scale = (
                    z[k].copy() for k in ("observations", "targets", "action_scale")
                )
            state, means = model.initial_state(), []
            with torch.no_grad():
                for value in obs:
                    mean, state = model(torch.from_numpy(value).reshape(1, 1, 160), state)
                    queries += 1
                    means.append(mean[0, 0].numpy().copy())
            means = np.stack(means)
            path = a.output / f"cpu-sequence{row['index']}.npz"
            np.savez_compressed(path, means=means)
            replay.append(
                dict(
                    index=row["index"],
                    role=row["role"],
                    rows=len(obs),
                    motor_MSE=float(np.mean(((means - targets) * scale) ** 2)),
                    prediction_path=path.name,
                    prediction_hash=digest(path),
                )
            )
        if bench.native_steps != 0:
            raise ValueError("offline training must not integrate physics")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status=CORRECTION_STATUS if correction else STATUS,
                source_head=a.expected_head,
                config=config,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                data_hash=digest(dp),
                warm_start_hash=warm_hash,
                parent_lineage_hash=data["parent_lineage_hash"],
                initial_checkpoint_path=str(first),
                initial_checkpoint_hash=digest(first),
                checkpoint_path=str(last),
                checkpoint_hash=digest(last),
                changed_parameter_tensors=changed,
                offline_supervised_steps=steps,
                cumulative_optimizer_steps=final_actor,
                cumulative_critic_optimizer_steps=48,
                initial_optimizer_steps=initial_actor,
                initial_critic_optimizer_steps=48,
                PPO_optimizer_steps_this_stage=0,
                cumulative_PPO_actor_steps=16,
                history=history,
                cpu_replay=replay,
                cpu_offline_decisions=queries,
                physical_steps_executed=0,
                new_rollout_generations=0,
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
                offline_supervised_steps=steps,
                cpu_offline_decisions=queries,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
