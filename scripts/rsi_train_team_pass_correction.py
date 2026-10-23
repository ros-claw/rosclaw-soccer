"""Complete-coverage actual-student-state offline pass correction, v2.

Carries the exact research student128/critic48 state and every Adam slot;
64 actual whole-history actor updates, no critic/PPO/ref update. All positive
admitted teacher and retention fit histories have a committed seeded schedule.
No physics here, no online-RL or growth claim before independent native exams.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("data-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument(
        "--seed", type=int, choices=(85023, 85024, 85025, 95023, 95024, 95025), required=True
    )
    p.add_argument(
        "--curriculum", choices=("all-positive-v2", "single-mode-v3"), default="all-positive-v2"
    )
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    a = p.parse_args()
    from rosclaw_soccer.rsi.team_pass_correction import (
        CORRECTION_BANK_STATUS as BANK_STATUS,
    )
    from rosclaw_soccer.rsi.team_pass_correction import (
        CORRECTION_KIND,
        correction_schedule,
    )
    from rosclaw_soccer.rsi.team_pass_correction import (
        CORRECTION_STATUS as STATUS,
    )
    from rosclaw_soccer.rsi.team_pass_correction import (
        correction_config as pass_config,
    )
    from rosclaw_soccer.rsi.team_pass_correction import (
        correction_pools as pass_pools,
    )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    dp = a.data_root / "complete.json"
    if (
        a.output.exists()
        or dp.is_symlink()
        or (a.data_root / "failed.json").exists()
        or digest(dp) != a.data_hash
    ):
        raise ValueError("exact completed audited274 bank and unused output required")
    data = json.loads(dp.read_text())
    if (
        data["status"] != BANK_STATUS
        or data["observation_width"] != 160
        or data["learner_hidden_width"] != 256
        or data["teacher_hidden_state_supplied_to_learner"] is not False
        or data["whole_actual_sequences"] is not True
        or any(
            data[k] != 0
            for k in ("physical_steps", "optimizer_updates", "DEV_fit_rows", "Fresh_fit_rows")
        )
        or any(
            data[k] is not False
            for k in (
                "goal_complete",
                "student_success_claim",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("whole actual unpromoted TRAIN-only bank required")
    pass_pools(data["records"])
    config_fn, schedule_fn, kind, status = pass_config, correction_schedule, CORRECTION_KIND, STATUS
    checkpoint_schema = "soccer.dual_motor.team_pass_correction.v2"
    fit_metadata = {}
    if a.curriculum == "single-mode-v3":
        from rosclaw_soccer.rsi.single_mode_pass_curriculum import single_mode_pass_curriculum
        from rosclaw_soccer.rsi.team_pass_single_mode import KIND as SELECTED_KIND
        from rosclaw_soccer.rsi.team_pass_single_mode import STATUS as SELECTED_STATUS
        from rosclaw_soccer.rsi.team_pass_single_mode import selected_config, selected_schedule

        config_fn, schedule_fn, kind, status = (
            selected_config,
            selected_schedule,
            SELECTED_KIND,
            SELECTED_STATUS,
        )
        checkpoint_schema = "soccer.dual_motor.team_pass_correction.v3"
        fit_metadata = {"fit_selection": single_mode_pass_curriculum(data["records"])}
    pins = dict(data["input_file_hashes"])
    pins[str(dp)] = a.data_hash
    dd = a.data_root / "declaration.json"
    if digest(dd) != data["declaration_hash"]:
        raise ValueError("exact whole bank declaration required")
    pins[str(dd)] = data["declaration_hash"]
    wp, warm = receipt(pins, data["warm_start_hash"])
    if (
        warm["cumulative_optimizer_steps"] != 128
        or warm["cumulative_critic_optimizer_steps"] != 48
        or warm["config"]["seed"] != 75024
    ):
        raise ValueError("same retained75024 warm actor128/critic48 required")
    for row in data["records"]:
        path = a.data_root / row["raw_path"]
        if (
            path.is_symlink()
            or path.parent.resolve() != a.data_root.resolve()
            or digest(path) != row["raw_hash"]
        ):
            raise ValueError("all274 confined complete actual sequences required")
        pins[str(path)] = row["raw_hash"]
    config = config_fn(a.seed, a.device)
    schedule = schedule_fn(data["records"], a.seed)
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            config=config,
            input_file_hashes=pins,
            data_hash=a.data_hash,
            warm_start_hash=data["warm_start_hash"],
            parent_lineage_hash=data["parent_lineage_hash"],
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
            **fit_metadata,
        ),
    )
    steps, queries = 0, 0
    history = []
    try:
        import copy

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
        contexts = (
            {int(k): v for k, v in data["contexts"].items()}
            if isinstance(data["contexts"], dict)
            else {i: v for i, v in enumerate(data["contexts"])}
        )
        bench, model = open_motor(
            pins,
            data["warm_start_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="team-pass-consolidated-v1",
        )
        path = Path(warm["checkpoint_path"])
        if digest(path) != warm["checkpoint_hash"]:
            raise ValueError("validated exact inherited actor/critic/optimizer checkpoint required")
        initial = torch.load(path, map_location="cpu", weights_only=True)
        critic = MotorCritic().eval().requires_grad_(False)
        critic.load_state_dict(initial["critic"], strict=True)
        model.plastic.requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                dict(params=list(model.plastic_parameters()), lr=1e-5),
                dict(params=list(critic.parameters()), lr=0.001),
            ]
        )
        optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
        if not same_training_state(
            optimizer.state_dict(), initial["optimizer"]
        ) or not same_training_state(dict(model.plastic.state_dict()), initial["plastic_network"]):
            raise ValueError("exact inherited network and every Adam slot before schedule required")
        first = a.output / "pass-initial.pt"
        torch.save(copy.deepcopy(initial), first)
        frozen = copy.deepcopy(model.parent.state_dict())
        model.train().to(a.device)
        optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
        optimizer.param_groups[0]["lr"] = config["actor_lr"]
        episodes = []
        for row in data["records"]:
            with np.load(a.data_root / row["raw_path"], allow_pickle=False) as z:
                episodes.append({k: torch.from_numpy(z[k].copy()).to(a.device) for k in z.files})
        for plan in schedule:
            teachers, new_anchor, anchors = (
                plan["teacher_indices"],
                plan["new_anchor_index"],
                plan["anchor_indices"],
            )
            optimizer.zero_grad(set_to_none=True)
            imitation = sum(sequence_loss(model.plastic, episodes[i]) for i in teachers) / 2
            retention = sum(sequence_loss(model.plastic, episodes[i]) for i in anchors) / 4
            new_retention = sequence_loss(model.plastic, episodes[new_anchor])
            loss = imitation + 4 * retention + new_retention
            if not torch.isfinite(loss):
                raise ValueError("finite whole-history motor loss required")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.plastic_parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            steps += 1
            entry = dict(
                step=steps,
                teacher_indices=teachers,
                new_anchor_index=new_anchor,
                anchor_indices=anchors,
                loss=float(loss.detach()),
                imitation=float(imitation.detach()),
                retention=float(retention.detach()),
                new_retention=float(new_retention.detach()),
                gradient_norm=float(grad),
            )
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

        opt = cpu_tree(optimizer.state_dict())
        validate_adam_slots(
            opt,
            list(model.plastic.parameters()) + list(critic.parameters()),
            192,
            48,
            actor_phase=kind,
        )
        if (
            any(
                not same_training_state(opt["state"][i], initial["optimizer"]["state"][i])
                for i in opt["param_groups"][1]["params"]
            )
            or not same_training_state(dict(critic.state_dict()), initial["critic"])
            or not same_training_state(model.parent.state_dict(), frozen)
        ):
            raise ValueError("critic weights/moments or frozen reference changed")
        final = cpu_tree(model.plastic.state_dict())
        changed = [k for k, v in final.items() if not torch.equal(v, initial["plastic_network"][k])]
        if set(changed) != set(dict(model.plastic.named_parameters())):
            raise ValueError("all16 plastic parameters and no normalizer buffers must change")
        last = a.output / "pass-final.pt"
        torch.save(
            dict(
                schema=checkpoint_schema,
                plastic_network=final,
                critic=cpu_tree(critic.state_dict()),
                optimizer=opt,
                optimizer_updates=dict(actor=192, critic=48),
                config=config,
                warm_start_hash=data["warm_start_hash"],
                parent_lineage_hash=data["parent_lineage_hash"],
                **(
                    {"fit_selection_hash": fit_metadata["fit_selection"]["selection_hash"]}
                    if fit_metadata
                    else {}
                ),
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
                    means.append(mean[0, 0].numpy().copy())
                    queries += 1
            means = np.stack(means)
            out = a.output / f"cpu-sequence{row['index']}.npz"
            np.savez_compressed(out, means=means)
            replay.append(
                dict(
                    index=row["index"],
                    role=row["role"],
                    rows=len(obs),
                    motor_MSE=float(np.mean(((means - targets) * scale) ** 2)),
                    prediction_path=out.name,
                    prediction_hash=digest(out),
                )
            )
        if bench.native_steps:
            raise ValueError("offline learning must not integrate physics")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=status,
                source_head=a.expected_head,
                config=config,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                data_hash=a.data_hash,
                warm_start_hash=data["warm_start_hash"],
                parent_lineage_hash=data["parent_lineage_hash"],
                initial_checkpoint_path=str(first),
                initial_checkpoint_hash=digest(first),
                checkpoint_path=str(last),
                checkpoint_hash=digest(last),
                changed_parameter_tensors=changed,
                offline_supervised_steps=steps,
                initial_optimizer_steps=128,
                initial_critic_optimizer_steps=48,
                cumulative_optimizer_steps=192,
                cumulative_critic_optimizer_steps=48,
                PPO_optimizer_steps_this_stage=0,
                cumulative_PPO_actor_steps=16,
                history=history,
                cpu_replay=replay,
                cpu_offline_decisions=queries,
                physical_steps_executed=0,
                online_RL_claim=False,
                old_retention_proven=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
                **fit_metadata,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="TEAM_PASS_CORRECTION_TRAINING_FAILED_PRESERVE_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                offline_supervised_steps=steps,
                cpu_offline_decisions=queries,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
