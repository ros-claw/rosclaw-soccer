"""Bounded supervised/self-imitation memory consolidation, NOT PPO or promotion."""

import argparse
import json
import math
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "baseline-training", "core-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-bank-hash", "expected-baseline-hash"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or digest(a.bank / "complete.json") != a.expected_bank_hash
        or digest(a.baseline_training / "complete.json") != a.expected_baseline_hash
    ):
        raise ValueError("clean frozen source and closed curated/base data required")
    bc = json.loads((a.bank / "complete.json").read_text())
    bd = json.loads((a.bank / "declaration.json").read_text())
    tc = json.loads((a.baseline_training / "complete.json").read_text())
    td = json.loads((a.baseline_training / "declaration.json").read_text())
    if (
        bc["status"] != "MOTION_CONSOLIDATION_DATA_CLOSED"
        or bc["sequences"] != 42
        or bc["validation_sequences"] != 0
        or bc["base_training_hash"] != a.expected_baseline_hash
        or bc["base_weights_hash"] != tc["checkpoint_hash"]
        or td["seed"] != 29
    ):
        raise ValueError(
            "fixed42 training-only consolidation and common seed29 initialization required"
        )
    path = a.baseline_training / tc["checkpoint"]
    if (
        digest(path) != tc["checkpoint_hash"]
        or digest(a.bank / bc["artifact"]) != bc["artifact_hash"]
    ):
        raise ValueError("model/data bytes differ")
    pins = {**bd["input_file_hashes"], **td["input_file_hashes"]}
    for root in (a.bank, a.baseline_training):
        pins.update({str(f): digest(f) for f in root.iterdir() if f.is_file()})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.motion_memory_consolidation_training.v1",
            mode="consolidate",
            source_head=a.expected_head,
            input_file_hashes=pins,
            seed=a.seed,
            base_weights_hash=tc["checkpoint_hash"],
            base_training_hash=a.expected_baseline_hash,
            bank_hash=bc["original_bank_hash"],
            experience_bank_hash=a.expected_bank_hash,
            common_initialization_seed=29,
            inherited_base_optimizer_steps=tc["optimizer_steps"],
            minibatch_sequences=8,
            optimizer_steps=3000,
            learning_rate=1e-4,
            gradient_clip=1.0,
            critic_frozen_and_not_calibrated_as_rl_value=True,
            frozen_muscle=True,
            normalizer_frozen=True,
            encoder_and_memory_trainable=True,
            online_reinforcement_learning=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.motion_consolidation_loss import weighted_motion_token_loss
    from rosclaw_soccer.rsi.neural_motion_intention import create_motion_intention_student

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(a.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("declared real CUDA learner required, no silent CPU substitution")
    torch.cuda.set_per_process_memory_fraction(0.1, device=device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(a.bank / bc["artifact"], allow_pickle=False) as z:
        x, y, mask, weight = (z[k].copy() for k in ("features", "motion_tokens", "mask", "weights"))
    if (
        x.shape != (42, 300, 108)
        or y.shape != (42, 300, 64)
        or mask.shape != (42, 300)
        or weight.shape != (42,)
        or x.dtype != np.float32
        or y.dtype != np.float32
        or mask.dtype != np.bool_
        or weight.dtype != np.float32
        or mask[:, :20].any()
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(weight).all()
        or np.any(weight <= 0)
    ):
        raise ValueError("complete finite fixed42 masked memory data required")
    active = np.flatnonzero(mask.any(axis=1))
    if len(active) < 8:
        raise ValueError("at least eight genuinely supervised sequences required")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = create_motion_intention_student()
    model.load_state_dict(payload["state_dict"], strict=True)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.critic.requires_grad_(False)
    model.to(device)
    xt, yt, mt, wt = [torch.from_numpy(v).to(device) for v in (x, y, mask, weight)]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(a.seed)

    def loss_at(indices):
        tokens, _, _ = model(xt[indices])
        return weighted_motion_token_loss(tokens, yt[indices], mt[indices], wt[indices]), tokens

    def export(step):
        state = {"base." + k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        state["log_std"] = torch.full((64,), math.log(0.02), dtype=torch.float32)
        output = a.output / ("initial.pt" if step == 0 else "candidate.pt")
        torch.save(
            dict(
                state_dict=state,
                optimizer=optimizer.state_dict(),
                optimizer_steps=step,
                inherited_base_optimizer_steps=tc["optimizer_steps"],
                source_head=a.expected_head,
                base_weights_hash=tc["checkpoint_hash"],
                experience_bank_hash=a.expected_bank_hash,
                seed=a.seed,
                model_kind="motion_intention_consolidation64",
                cpu_rng_state=torch.get_rng_state(),
                minibatch_rng_state=generator.get_state(),
                cuda_rng_state=torch.cuda.get_rng_state(device).cpu(),
                hardware_authorized=False,
                promotion_authorized=False,
            ),
            output,
        )
        return digest(output)

    initial_hash = export(0)
    history = []
    model.train()
    for step in range(1, 3001):
        order = torch.randperm(len(active), generator=generator)[:8].numpy()
        indices = torch.from_numpy(active[order]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_at(indices)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite supervised consolidation loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            model.eval()
            with torch.no_grad():
                full_loss, tokens = loss_at(torch.arange(42, device=device))
                mismatch = ((tokens != yt) & mt[:, :, None]).sum() / (mt.sum() * 64)
            row = dict(
                step=step,
                weighted_full_training_token_mse=float(full_loss.cpu()),
                supervised_symbol_mismatch_fraction=float(mismatch.cpu()),
                gradient_norm=float(norm.cpu()),
            )
            write_once(a.output / f"update{step:04d}.json", row)
            history.append(row)
            print("MOTION_MEMORY_CONSOLIDATION", a.seed, json.dumps(row), flush=True)
            model.train()
    changes = {
        k: float((v.detach().cpu() - before[k]).double().square().sum())
        for k, v in model.state_dict().items()
    }
    if any(
        changes[k] != 0 for k in changes if k.startswith(("critic.", "observation_"))
    ) or not all(
        any(changes[k] > 0 for k in changes if k.startswith(prefix))
        for prefix in ("actor.", "encoder.", "memory.")
    ):
        raise ValueError("actual actor/trunk changes and exact frozen critic/normalizer required")
    checkpoint_hash = export(3000)
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="MOTION_MEMORY_CONSOLIDATED_REQUIRES_EXAM",
            source_head=a.expected_head,
            seed=a.seed,
            optimizer_steps=3000,
            checkpoint_hash=checkpoint_hash,
            checkpoint="candidate.pt",
            initial_checkpoint_hash=initial_hash,
            experience_bank_hash=a.expected_bank_hash,
            base_weights_hash=tc["checkpoint_hash"],
            parameter_change_l2_squared=changes,
            device=torch.cuda.get_device_name(device),
            history=history,
            new_native_steps=0,
            supervised_consolidation=True,
            online_reinforcement_learning=False,
            critic_not_calibrated_as_rl_value=True,
            closed_loop_skill_verified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
