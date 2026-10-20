"""Three fixed-seed supervised motor/auxiliary-value students; not online RL."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--expected-bank-hash", required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != args.expected_head
        or args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("clean pinned training source and unused output required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if digest(args.bank / "complete.json") != args.expected_bank_hash:
        raise ValueError("curated motor bank identity differs")
    proof = json.loads((args.bank / "complete.json").read_text())
    bank = json.loads((args.bank / "declaration.json").read_text())
    if (
        proof["total_courses"] != 40
        or proof["train_courses"] != 32
        or proof["validation_courses"] != 8
        or proof["declaration_hash"] != digest(args.bank / "declaration.json")
        or digest(args.bank / proof["artifact"]) != proof["artifact_hash"]
    ):
        raise ValueError("complete40 curated bank required")
    pins = dict(bank["input_file_hashes"])
    pins.update({str(p): digest(p) for p in args.bank.iterdir() if p.is_file()})
    check_pins(pins)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    protocol = Path.cwd() / "docs/rsi/recurrent-motor-bc-preregister-20261020.zh-CN.md"
    pins[str(protocol)] = digest(protocol)
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.motor_student_bank import training_observation_statistics
    from rosclaw_soccer.rsi.neural_motor_student import create_motor_student
    from scripts.rsi_atomic_artifacts import write_once

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda"):
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError("deterministic CUDA workspace must be declared before training")
        torch.cuda.set_per_process_memory_fraction(0.10)
    device = torch.device(args.device)
    with np.load(args.bank / proof["artifact"], allow_pickle=False) as z:
        x = z["features"].copy()
        y = z["targets"].copy()
        train = z["train"].copy()
        actor = z["actor_eligible"].copy()
    if (
        x.shape != (40, 300, 108)
        or y.shape != (40, 300, 29)
        or x.dtype != np.float32
        or y.dtype != np.float32
        or train.shape != (40,)
        or actor.shape != (40,)
        or train.dtype != np.bool_
        or actor.dtype != np.bool_
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.any(train & actor)
    ):
        raise ValueError("finite complete causal motor bank required")
    model = create_motor_student().to(device)
    observation_mean, observation_std = training_observation_statistics(x, train)
    with torch.no_grad():
        model.observation_mean.copy_(torch.from_numpy(observation_mean).to(device))
        model.observation_std.copy_(torch.from_numpy(observation_std).to(device))
    initial_parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.recurrent_motor_bc.v1",
        source_head=head,
        input_file_hashes=pins,
        bank_complete_hash=args.expected_bank_hash,
        seed=args.seed,
        device=str(device),
        torch_version=str(torch.__version__),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        training_steps=3000,
        hidden_size=128,
        action_limit=8,
        learning_rate=0.0003,
        auxiliary_value_weight=0.1,
        critic_is_online_rl=False,
        publicity_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)

    def checkpoint(name, step):
        payload = dict(
            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            seed=args.seed,
            optimizer_steps=step,
            hidden_size=128,
            action_limit=8.0,
            source_head=head,
            bank_hash=args.expected_bank_hash,
        )
        with tempfile.NamedTemporaryFile(dir=args.output, suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            try:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
                os.link(tmp, args.output / name)
            finally:
                tmp.unlink(missing_ok=True)

    checkpoint("initial.pt", 0)
    xt = torch.from_numpy(x[train]).to(device)
    yt = torch.from_numpy(y[train]).to(device)
    actor_mask = torch.from_numpy(actor[train]).to(device)
    discount = torch.pow(
        torch.tensor(0.99, device=device), torch.arange(299, -1, -1, device=device)
    )
    values = (
        torch.from_numpy(actor[train].astype(np.float32)).to(device)[:, None] * discount[None, :]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    final = {}
    for step in range(1, 3001):
        optimizer.zero_grad(set_to_none=True)
        actions, value, _ = model(xt)
        action_loss = (actions[actor_mask] - yt[actor_mask]).square().mean()
        value_loss = (value - values).square().mean()
        loss = action_loss + 0.1 * value_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("missing or nonfinite actor/memory/value gradients")
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            with torch.no_grad():
                vx = torch.from_numpy(x[~train]).to(device)
                vy = torch.from_numpy(y[~train]).to(device)
                predicted, _, _ = model(vx)
                valid_actor = torch.from_numpy(actor[~train]).to(device)
                validation = (
                    float((predicted[valid_actor] - vy[valid_actor]).square().mean().cpu())
                    if valid_actor.any()
                    else None
                )
            final = dict(
                step=step,
                train_actor_mse=float(action_loss.detach().cpu()),
                train_value_mse=float(value_loss.detach().cpu()),
                validation_actor_mse=validation,
                gradient_norm_before_clip=float(gradient.detach().cpu()),
            )
            write_once(args.output / f"step{step:04d}.json", final)
            print(
                "MOTOR_STUDENT_OPTIMIZER", args.seed, json.dumps(final, sort_keys=True), flush=True
            )
        if step % 1000 == 0:
            checkpoint(f"step{step:04d}.pt", step)
    change = sum(
        float((p.detach().cpu() - initial_parameters[name]).square().sum())
        for name, p in model.named_parameters()
    )
    if change <= 0:
        raise ValueError("optimizer made no parameter change")
    check_pins(pins)
    write_once(
        args.output / "complete.json",
        dict(
            final,
            optimizer_steps=3000,
            parameter_change_l2_squared=change,
            checkpoint="step3000.pt",
            checkpoint_hash=digest(args.output / "step3000.pt"),
            initial_checkpoint_hash=digest(args.output / "initial.pt"),
            source_head=head,
            bank_complete_hash=args.expected_bank_hash,
            physical_rollouts=0,
            closed_loop_skill_verified=False,
            online_reinforcement_learning=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
