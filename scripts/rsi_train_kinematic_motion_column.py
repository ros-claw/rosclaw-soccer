"""Train only an isolated geometry adapter on the frozen42 train-only bank.

Supervised/self-imitation, NOT online RL. Zero-column native identity must close
first. The old motion parent, critic, normalizers and muscle remain untouched.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "core-source", "bank", "identity", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-inputs-hash",
        "expected-bank-hash",
        "identity-unit",
        "expected-identity-source",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit, load_frozen_experts
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, _, pins, _ = admit(a)
    wait_for_terminal_collection(a.identity_unit, wait=True)
    identity = json.loads((a.identity / "complete.json").read_text())
    if (
        identity["status"] != "ZERO_KINEMATIC_COLUMN_NATIVE_PARENT_BIT_EXACT"
        or identity["source_head"] != a.expected_identity_source
        or identity["new_native_steps"] != 24000
        or identity["optimizer_steps"] != 0
        or digest(a.bank / "complete.json") != a.expected_bank_hash
    ):
        raise ValueError("closed native zero-identity and fixed bank required before training")
    idd = json.loads((a.identity / "declaration.json").read_text())
    for key, value in idd["input_file_hashes"].items():
        if key in pins and pins[key] != value:
            raise ValueError("identity inputs conflict with learner")
        pins[key] = value
    for name in ("measured_foot_kinematics.py", "kinematic_motion_residual.py"):
        old = [Path(k) for k in idd["input_file_hashes"] if k.endswith("/rsi/" + name)]
        if len(old) != 1 or digest(old[0]) != digest(Path("src/rosclaw_soccer/rsi") / name):
            raise ValueError("native identity must bind EXACT learner geometry/actor code")
    bc = json.loads((a.bank / "complete.json").read_text())
    bd = json.loads((a.bank / "declaration.json").read_text())
    if (
        bc["status"] != "MOTION_CONSOLIDATION_DATA_CLOSED"
        or bc["sequences"] != 42
        or bc["validation_sequences"] != 0
        or bc["base_weights_hash"] != spec["experts"][0]["checkpoint_hash"]
        or digest(a.bank / bc["artifact"]) != bc["artifact_hash"]
    ):
        raise ValueError("same original42 supervised/self-imitation bank required")
    pins.update(bd["input_file_hashes"])
    for root in (a.bank, a.identity):
        pins.update({str(f): digest(f) for f in root.glob("*.json")})
    pins[str(a.bank / bc["artifact"])] = bc["artifact_hash"]
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.kinematic_column_training.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            seed=a.seed,
            experience_bank_hash=a.expected_bank_hash,
            parent_checkpoint_hash=spec["experts"][0]["checkpoint_hash"],
            world_hash=digest(Path(spec["world"])),
            optimizer_steps=2000,
            raw_residual_cap=0.125,
            frozen_parent_all_parameters=True,
            value_column_frozen=True,
            log_std_frozen=True,
            geometry_is_reconstructed_only_from_existing108=True,
            online_reinforcement_learning=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.kinematic_motion_residual import create_kinematic_motion_residual
    from rosclaw_soccer.rsi.measured_foot_kinematics import MeasuredFootKinematics
    from rosclaw_soccer.rsi.motion_consolidation_loss import weighted_motion_token_loss

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise ValueError("declared CUDA training required")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    world = mujoco.MjModel.from_binary_path(spec["world"])
    extractor = MeasuredFootKinematics(world, expected_world_hash=digest(Path(spec["world"])))
    parent = load_frozen_experts(spec)[0]
    model = create_kinematic_motion_residual(parent, extractor)
    model.value_adapter.requires_grad_(False)
    model.log_std.requires_grad_(False)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with np.load(a.bank / bc["artifact"], allow_pickle=False) as z:
        arrays = [z[k].copy() for k in ("features", "motion_tokens", "mask", "weights")]
    x, target, mask, weights = [torch.from_numpy(v) for v in arrays]
    # Original loss checks full shapes, finite targets and prefix masking.
    weighted_motion_token_loss(target, target, mask, weights)
    if x.shape != (42, 300, 108) or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError("fixed42 current-state sequences required")
    geometry = model.geometry(x)
    model.geometry_mean.copy_(geometry[mask].mean(0))
    model.geometry_std.copy_(geometry[mask].std(0, unbiased=False).clamp_min(0.01))
    # Only the new branch's normalization changes; parent normalizer is immutable.
    model.to(device)
    x, target, mask, weights, geometry = [
        v.to(device) for v in (x, target, mask, weights, geometry)
    ]
    with torch.no_grad():
        norm = ((x - parent.base.observation_mean) / parent.base.observation_std).clamp(-10, 10)
        memory, _ = parent.base.memory(parent.base.encoder(norm))
        raw_parent = parent.base.actor(memory)
        context = torch.cat(
            (norm, memory, ((geometry - model.geometry_mean) / model.geometry_std).clamp(-10, 10)),
            -1,
        )
    generator = torch.Generator().manual_seed(a.seed)
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=1e-4)
    with torch.no_grad():
        lower = (
            torch.round(
                15.515501022338867 * torch.tanh(raw_parent - 0.125 + 0.03223699703812599) - 0.5
            )
            / 16
        )
        upper = (
            torch.round(
                15.515501022338867 * torch.tanh(raw_parent + 0.125 + 0.03223699703812599) - 0.5
            )
            / 16
        )
        unreachable = (((target < lower) | (target > upper)) & mask[:, :, None]).sum() / (
            mask.sum() * 64
        )
    write_once(
        a.output / "bounded-target-reachability.json",
        dict(
            supervised_symbols_outside_raw_cap_fraction=float(unreachable.cpu()),
            cap_will_not_be_changed_after_observing_results=True,
            reachability_is_not_task_success=True,
            optimizer_steps=0,
            new_native_steps=0,
        ),
    )

    def tokens(indices):
        raw = raw_parent[indices] + 0.125 * torch.tanh(model.adapter(context[indices]))
        bounded = 15.515501022338867 * torch.tanh(raw + 0.03223699703812599) - 0.5
        continuous, discrete = bounded / 16.0, torch.round(bounded) / 16.0
        return continuous + (discrete - continuous).detach()

    history = []
    for step in range(1, 2001):
        indices = torch.randperm(42, generator=generator)[:8].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_motion_token_loss(
            tokens(indices), target[indices], mask[indices], weights[indices]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            with torch.no_grad():
                full = weighted_motion_token_loss(
                    tokens(torch.arange(42, device=device)), target, mask, weights
                )
            row = dict(step=step, weighted_full_training_token_mse=float(full.cpu()))
            history.append(row)
            write_once(a.output / f"update{step:04d}.json", row)
            print("KINEMATIC_COLUMN_LEARNED", a.seed, row, flush=True)
    changes = {
        k: float((v.detach().cpu() - before[k]).double().square().sum())
        for k, v in model.state_dict().items()
    }
    if any(
        value != 0
        for k, value in changes.items()
        if k.startswith(("parent.", "value_adapter.", "log_std"))
    ) or not any(value > 0 for k, value in changes.items() if k.startswith("adapter.")):
        raise ValueError("new column must learn while entire predecessor remains bit-exact")
    torch.save(
        dict(
            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            source_head=a.expected_head,
            model_kind="kinematic_motion_residual64",
            optimizer=optimizer.state_dict(),
            seed=a.seed,
            optimizer_steps=2000,
            experience_bank_hash=a.expected_bank_hash,
            parent_checkpoint_hash=spec["experts"][0]["checkpoint_hash"],
            world_hash=digest(Path(spec["world"])),
            minibatch_rng_state=generator.get_state(),
            cpu_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state().cpu(),
            hardware_authorized=False,
            promotion_authorized=False,
        ),
        a.output / "candidate.pt",
    )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="KINEMATIC_COLUMN_TRAINED_REQUIRES_NATIVE_EXAM",
            source_head=a.expected_head,
            seed=a.seed,
            optimizer_steps=2000,
            checkpoint_hash=digest(a.output / "candidate.pt"),
            parameter_changes=changes,
            history=history,
            experience_bank_hash=a.expected_bank_hash,
            device=torch.cuda.get_device_name(),
            new_native_steps=0,
            online_reinforcement_learning=False,
            closed_loop_skill_verified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
