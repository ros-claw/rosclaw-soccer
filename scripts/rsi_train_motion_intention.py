"""Fixed3000 supervised motion-intention updates; frozen muscle weights never change."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "smoke", "prior-cycle", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "expected-bank-hash",
        "expected-smoke-hash",
        "expected-prior-cycle-hash",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    a = p.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != a.expected_head
        or a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("clean frozen intention source and unused training output required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    for path, value in (
        (a.bank / "complete.json", a.expected_bank_hash),
        (a.smoke / "complete.json", a.expected_smoke_hash),
        (a.prior_cycle / "complete.json", a.expected_prior_cycle_hash),
    ):
        if digest(path) != value:
            raise ValueError("fixed input evidence identity differs")
    bd = json.loads((a.bank / "declaration.json").read_text())
    bc = json.loads((a.bank / "complete.json").read_text())
    smoke = json.loads((a.smoke / "complete.json").read_text())
    prior = json.loads((a.prior_cycle / "complete.json").read_text())
    if (
        bc["actions_byte_exact"] != 12000
        or bc["status"] != "ALL40_TEACHER_NN_EXACT_MOTION_LABELS_NOT_TRAINED_POLICY"
        or bc["total_courses"] != 40
        or bc["train_courses"] != 32
        or digest(a.bank / bc["artifact"]) != bc["artifact_hash"]
        or smoke["bank_hash"] != a.expected_bank_hash
        or smoke["actual_native_steps"] != 3000
        or smoke["replay"]["additional_native_steps"] != 3000
        or smoke["replay"]["student_neural_and_muscle_native_bit_exact"] is not True
        or smoke["replay"]["teacher_encoder_calls"] != 0
        or smoke["replay"]["teacher_planner_calls"] != 0
        or prior["status"] != "DAGGER_GENERATION_PHYSICALLY_VERIFIED_NOT_PROMOTED"
        or prior["trained_qualified"] != 0
    ):
        raise ValueError(
            "full motion data, real own-state smoke and concluded failed direct29 cycle required"
        )
    pins = dict(bd["input_file_hashes"])
    pins.update(smoke["runtime_file_hashes"])
    for relative in (
        "src/rosclaw_soccer/rsi/neural_motion_intention.py",
        "src/rosclaw_soccer/rsi/foundation_motor_codec.py",
        "scripts/rsi_motion_student_rollout.py",
    ):
        if {
            h for name, h in smoke["runtime_file_hashes"].items() if name.endswith("/" + relative)
        } != {digest(Path.cwd() / relative)}:
            raise ValueError("intention/muscle implementation differs from actual native smoke")
    for root in (a.bank, a.smoke):
        pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
    for name in ("complete.json", "declaration.json"):
        pins[str(a.prior_cycle / name)] = digest(a.prior_cycle / name)
    check_pins(pins)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    protocol = Path.cwd() / "docs/rsi/motion-intention-training-preregister-20261020.zh-CN.md"
    pins[str(protocol)] = digest(protocol)
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.motor_student_bank import training_observation_statistics
    from rosclaw_soccer.rsi.neural_motion_intention import create_motion_intention_student
    from scripts.rsi_atomic_artifacts import write_once

    with np.load(a.bank / bc["artifact"], allow_pickle=False) as z:
        x, y, train, actor = (
            z[k].copy() for k in ("features", "motion_tokens", "train", "actor_eligible")
        )
    if (
        x.shape != (40, 300, 108)
        or y.shape != (40, 300, 64)
        or x.dtype != np.float32
        or y.dtype != np.float32
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.array_equal(y * 16, np.round(y * 16))
        or y.min() < -1
        or y.max() > 0.9375
        or train.shape != (40,)
        or actor.shape != (40,)
        or train.dtype != np.bool_
        or actor.dtype != np.bool_
        or int(train.sum()) != 32
        or int((train & actor).sum()) != 10
        or not np.array_equal(train, np.asarray([r["split"] == "train" for r in bd["rows"]]))
    ):
        raise ValueError("original whole40 causal108/FSQ64 bank and split required")
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    if a.device.startswith("cuda"):
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError("deterministic CUDA workspace required")
        torch.cuda.set_per_process_memory_fraction(0.10)
    device = torch.device(a.device)
    model = create_motion_intention_student().to(device)
    mean, std = training_observation_statistics(x, train)
    with torch.no_grad():
        model.observation_mean.copy_(torch.from_numpy(mean).to(device))
        model.observation_std.copy_(torch.from_numpy(std).to(device))
    model.initialize_training_mean(torch.from_numpy(y[train & actor].mean(axis=(0, 1))).to(device))
    initial = {n: v.detach().cpu().clone() for n, v in model.named_parameters()}
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.motion_intention_fsq64_training.v1",
            source_head=head,
            input_file_hashes=pins,
            bank_complete_hash=a.expected_bank_hash,
            smoke_hash=a.expected_smoke_hash,
            prior_cycle_hash=a.expected_prior_cycle_hash,
            seed=a.seed,
            hidden_size=128,
            motion_width=64,
            model_kind="motion_intention_fsq64",
            training_steps=3000,
            device=str(device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            torch_version=str(torch.__version__),
            learning_rate=0.0003,
            auxiliary_value_weight=0.1,
            initialization="train_positive_mean_intention_with_frozen_pretrained_muscle_decoder",
            online_reinforcement_learning=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )

    def checkpoint(name, step):
        saved = dict(
            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            seed=a.seed,
            optimizer_steps=step,
            model_kind="motion_intention_fsq64",
            hidden_size=128,
            motion_width=64,
            source_head=head,
            bank_hash=a.expected_bank_hash,
        )
        with tempfile.NamedTemporaryFile(dir=a.output, suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            try:
                torch.save(saved, stream)
                stream.flush()
                os.fsync(stream.fileno())
                os.link(tmp, a.output / name)
            finally:
                tmp.unlink(missing_ok=True)

    checkpoint("initial.pt", 0)

    def tensor(array):
        return torch.from_numpy(array).to(device)

    xt, yt, mask = tensor(x[train]), tensor(y[train]), tensor(actor[train])
    discount = torch.pow(
        torch.tensor(0.99, device=device), torch.arange(299, -1, -1, device=device)
    )
    values = tensor(actor[train].astype(np.float32))[:, None] * discount[None]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    final = {}
    for step in range(1, 3001):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tokens, value, _ = model(xt)
        actor_loss = (tokens[mask] - yt[mask]).square().mean()
        value_loss = (value - values).square().mean()
        loss = actor_loss + 0.1 * value_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite motion loss")
        loss.backward()
        if any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters()):
            raise ValueError("missing/nonfinite motion gradient")
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            model.eval()
            with torch.no_grad():
                predicted, _, _ = model(tensor(x[~train]))
                valid_mask = tensor(actor[~train])
                validation = (predicted[valid_mask] - tensor(y[~train])[valid_mask]).square().mean()
            final = dict(
                step=step,
                train_motion_token_mse=float(actor_loss.detach().cpu()),
                train_value_mse=float(value_loss.detach().cpu()),
                validation_motion_token_mse=float(validation.cpu()),
                gradient_norm=float(gradient.detach().cpu()),
            )
            write_once(a.output / f"step{step:04d}.json", final)
            print(
                "MOTION_INTENTION_OPTIMIZER", a.seed, json.dumps(final, sort_keys=True), flush=True
            )
        if step % 1000 == 0:
            checkpoint(f"step{step:04d}.pt", step)
    change = sum(
        float((v.detach().cpu() - initial[n]).square().sum()) for n, v in model.named_parameters()
    )
    if change <= 0:
        raise ValueError("no actual intention parameter change")
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            **final,
            optimizer_steps=3000,
            parameter_change_l2_squared=change,
            checkpoint="step3000.pt",
            checkpoint_hash=digest(a.output / "step3000.pt"),
            initial_checkpoint_hash=digest(a.output / "initial.pt"),
            source_head=head,
            bank_complete_hash=a.expected_bank_hash,
            model_kind="motion_intention_fsq64",
            muscle_decoder_trained_or_modified=False,
            physical_rollouts=0,
            closed_loop_skill_verified=False,
            online_reinforcement_learning=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
