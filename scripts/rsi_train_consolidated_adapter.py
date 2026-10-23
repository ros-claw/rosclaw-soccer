"""Train a new neural head without rewriting any of the learned112/48 memory.

Whole causal histories supply own frozen features; minibatches carry no teacher
h/c and no case identifier into the network. Offline BC, never online RL.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("feature-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("feature-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seed", type=int, choices=(105023, 105024, 105025), required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    from rosclaw_soccer.rsi.consolidated_adapter_learning import (
        FEATURE_STATUS,
        KIND,
        SCHEMA,
        STATUS,
        adapter_config,
        adapter_fit_plan,
        adapter_schedule,
        validate_head_checkpoint,
    )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import receipt, source_pins

    clean(a.expected_head)
    fp = a.feature_root / "complete.json"
    if (
        a.output.exists()
        or fp.is_symlink()
        or digest(fp) != a.feature_hash
        or (a.feature_root / "failed.json").exists()
    ):
        raise ValueError("exact closed own-feature qualification and unused output required")
    f = json.loads(fp.read_text())
    if (
        f["status"] != FEATURE_STATUS
        or len(f["records"]) != 274
        or f["zero_head_bit_identity"] is not True
        or f["teacher_hidden_state_supplied"] is not False
        or any(
            f[k] != 0
            for k in ("optimizer_updates", "physical_steps", "DEV_fit_rows", "Fresh_fit_rows")
        )
        or any(
            f[k] is not False
            for k in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("whole274 actual TRAIN features and unchanged memory required")
    pins = dict(f["input_file_hashes"])
    pins[str(fp)] = a.feature_hash
    _, bank = receipt(pins, f["data_hash"])
    _, memory_receipt = receipt(pins, f["memory_training_hash"])
    if (
        memory_receipt["checkpoint_hash"] != f["memory_checkpoint_hash"]
        or memory_receipt["cumulative_optimizer_steps"] != 112
        or memory_receipt["cumulative_critic_optimizer_steps"] != 48
    ):
        raise ValueError("exact whole learned112/48 slow memory required")
    for row, source in zip(f["records"], bank["records"], strict=True):
        path = a.feature_root / row["feature_path"]
        if (
            path.is_symlink()
            or path.parent.resolve() != a.feature_root.resolve()
            or row["index"] != source["index"]
            or row["source_raw_hash"] != source["raw_hash"]
            or row["raw_task_success"] is not source["task_success"]
            or row["zero_head_bit_identity"] is not True
            or row["rows"] != source["rows"]
        ):
            raise ValueError("all actual matched whole histories and unmodified labels required")
        pins[str(path)] = row["feature_hash"]
    config = adapter_config(a.seed, a.device)
    plan = adapter_fit_plan(bank["records"])
    schedule = adapter_schedule(plan, a.seed)
    commitments = dict(
        feature_hash=a.feature_hash,
        data_hash=f["data_hash"],
        fit_plan_hash=plan["hash"],
        memory_training_hash=f["memory_training_hash"],
        memory_checkpoint_hash=f["memory_checkpoint_hash"],
        parent_lineage_hash=bank["parent_lineage_hash"],
    )
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            config=config,
            fit_plan=plan,
            **commitments,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    steps = 0
    try:
        import copy

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
        from scripts.rsi_stroke_search_common import open_motor

        torch.manual_seed(a.seed)
        torch.use_deterministic_algorithms(True)
        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        _, memory = open_motor(
            pins,
            f["memory_training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="contact-consolidated-v2",
        )
        model = ConsolidatedMotorAdapter(memory, head_bound_rad=config["head_bound_rad"])
        initial = copy.deepcopy(model.head.state_dict())
        frozen = copy.deepcopy(model.memory.state_dict())
        model.train().to(a.device)
        optimizer = torch.optim.Adam(model.plastic_parameters(), lr=config["head_lr"])
        if optimizer.state_dict()["state"]:
            raise ValueError("new head has new parameters, not reset inherited Adam")
        fit_indices = sorted(
            set(
                plan["teacher_indices"] + plan["old_anchor_indices"] + plan["baseline_pass_indices"]
            )
        )
        episodes = {}
        scale = model.memory.parent.get_buffer("action_scale")
        for i in fit_indices:
            row = f["records"][i]
            with np.load(a.feature_root / row["feature_path"], allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            shapes = dict(
                features=(row["rows"], 288),
                memory_means=(row["rows"], 29),
                memory_delta=(row["rows"], 29),
                target_delta=(row["rows"], 29),
                loss_weights=(row["rows"],),
                action_scale=(29,),
            )
            if (
                set(raw) != set(shapes)
                or any(
                    raw[k].shape != shape
                    or raw[k].dtype != np.float32
                    or not np.isfinite(raw[k]).all()
                    for k, shape in shapes.items()
                )
                or (raw["loss_weights"] <= 0).any()
            ):
                raise ValueError(
                    "exact finite own-feature archives and positive loss weights required"
                )
            if not np.array_equal(raw["action_scale"], scale.detach().cpu().numpy()):
                raise ValueError("unchanged original action scale required")
            target = (
                raw["target_delta"]
                if i in plan["teacher_indices"]
                else np.zeros_like(raw["target_delta"])
            )
            if np.abs(target).max() >= config["head_bound_rad"]:
                raise ValueError(
                    "teacher action outside new-head authority; no silent target clipping"
                )
            episodes[i] = {
                k: torch.from_numpy(v).to(a.device)
                for k, v in dict(
                    features=raw["features"],
                    memory_delta=raw["memory_delta"],
                    target=target,
                    weights=raw["loss_weights"],
                ).items()
            }
        history = []
        for entry in schedule:
            indices = (
                entry["teacher_indices"]
                + entry["old_anchor_indices"]
                + [entry["baseline_pass_index"]]
            )
            n = max(len(episodes[i]["features"]) for i in indices)
            x = torch.zeros(n, 7, 288, device=a.device)
            delta = torch.zeros(n, 7, 29, device=a.device)
            target = torch.zeros_like(delta)
            w = torch.zeros(n, 7, device=a.device)
            for col, i in enumerate(indices):
                ep, t = episodes[i], len(episodes[i]["features"])
                x[:t, col], delta[:t, col] = ep["features"], ep["memory_delta"]
                target[:t, col], w[:t, col] = ep["target"], ep["weights"]
            optimizer.zero_grad(set_to_none=True)
            added, _ = model.adapt_features(x, delta, scale)
            per_history = (
                ((added - target) / config["motor_loss_unit_rad"]).square().mean(-1) * w
            ).sum(0) / w.sum(0)
            imitation, retention, baseline = (
                per_history[:2].mean(),
                per_history[2:6].mean(),
                per_history[6],
            )
            loss = (
                imitation
                + config["old_weight"] * retention
                + config["baseline_pass_weight"] * baseline
            )
            if not torch.isfinite(loss):
                raise ValueError("finite actual new-head loss required")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.plastic_parameters(), config["grad_norm_limit"], error_if_nonfinite=True
            )
            optimizer.step()
            steps += 1
            report = dict(
                **entry,
                loss=float(loss.detach()),
                imitation=float(imitation.detach()),
                retention=float(retention.detach()),
                baseline=float(baseline.detach()),
                gradient_norm=float(grad),
            )
            history.append(report)
            if steps % 64 == 0:
                write_once(a.output / f"update{steps}.json", report)
                print(json.dumps(report), flush=True)
        model.cpu().eval().requires_grad_(False)
        if any(not torch.equal(v, frozen[k]) for k, v in model.memory.state_dict().items()):
            raise ValueError("whole learned memory changed while fitting new head")

        def cpu_tree(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu_tree(v) for v in value)
            return copy.deepcopy(value)

        checkpoint = dict(
            schema=SCHEMA,
            config=config,
            head=dict(model.head.state_dict()),
            initial_head=dict(initial),
            head_optimizer=cpu_tree(optimizer.state_dict()),
            head_optimizer_steps=steps,
            **commitments,
        )
        validate_head_checkpoint(
            checkpoint, dict(model.head.state_dict()), config=config, commitments=commitments
        )
        cp = a.output / "adapter-final.pt"
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
                config=config,
                fit_plan=plan,
                **commitments,
                checkpoint_path=str(cp),
                checkpoint_hash=digest(cp),
                head_optimizer_steps=steps,
                memory_optimizer_steps_unchanged=dict(actor=112, critic=48),
                history=history,
                physical_steps_executed=0,
                old_retention_proven=False,
                online_RL_claim=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                actual_head_optimizer_steps=steps,
                physical_steps=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
