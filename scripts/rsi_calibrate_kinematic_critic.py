"""Fixed-actor value calibration on a sealed SAME-behavior32 episode bank.

Internal24/8 trajectory split; NOT original8 validation, Fresh or new skill.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "core-source", "pipeline", "policy", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "expected-inputs-hash",
        "expected-policy-hash",
        "expected-round-hash",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    root = a.pipeline / "round1"
    if (
        digest(root / "complete.json") != a.expected_round_hash
        or digest(a.policy) != a.expected_policy_hash
    ):
        raise ValueError("sealed second-batch and EXACT behavior policy required")
    d = json.loads((a.pipeline / "declaration.json").read_text())
    producer = "37174508238b36e08983f0002f2ef61c79b82b89"
    if d["source_head"] != producer or d["seed"] != 29:
        raise ValueError("explicit selected development actor29; no posthoc concealment")
    pins.update(d["input_file_hashes"])
    pins[str(root / "complete.json")] = a.expected_round_hash
    pins[str(a.policy)] = a.expected_policy_hash
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            actor_checkpoint_hash=a.expected_policy_hash,
            actor_selected_on_consumed_development=True,
            critic_minibatch_seed=a.seed,
            optimizer_steps=1000,
            critic_learning_rate=0.001,
            actor_and_std_and_parent_and_geometry_normalizer_frozen=True,
            internal_fit_contexts=24,
            internal_holdout_contexts=8,
            original_validation_in_gradients=0,
            no_new_physics=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.motion_value_calibration import (
        calibration_split,
        task_returns,
        value_metrics,
    )
    from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree
    from scripts.rsi_audit_online_kinematic_motion import read_phase
    from scripts.rsi_online_kinematic_motion import load_model

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    world = mujoco.MjModel.from_binary_path(spec["world"])
    phase_args = SimpleNamespace(seed=29, expected_pipeline_source=producer)
    collection = read_phase(phase_args, bd, pins, world, root, "collect", 1, a.expected_policy_hash)
    check_pins(pins)
    rows = {}
    for folder, c in collection:
        for row in c["result"]["results"]:
            i = row["course_index"]
            part = folder / f"course{i:02d}"
            with np.load(part / "native.npz", allow_pickle=False) as z:
                x = z["student_features"].copy()
            with np.load(part / "sampling.npz", allow_pickle=False) as z:
                r = z["rewards"].copy()
            rows[i] = (x, r)
    indices = sorted(rows)
    fit, hold = calibration_split(indices)
    x = torch.from_numpy(np.stack([rows[i][0] for i in indices]))
    target = task_returns(np.stack([rows[i][1] for i in indices]))
    model = load_model(spec, a.policy)
    model.requires_grad_(False)
    model.value_adapter.requires_grad_(True)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        original_mean, original_std, initial_value, original_hidden = model.parameters_for_state(x)
        geometry = model.geometry(x)
    if not torch.cuda.is_available():
        raise ValueError("declared CUDA critic-only learner required")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model.to(device)
    xt = x.to(device)
    with torch.no_grad():
        base = model.parent.base
        norm = ((xt - base.observation_mean) / base.observation_std).clamp(-10, 10)
        memory, _ = base.memory(base.encoder(norm))
        context = torch.cat(
            (
                norm,
                memory,
                ((geometry.to(device) - model.geometry_mean) / model.geometry_std).clamp(-10, 10),
            ),
            -1,
        )
        base_value = base.critic(memory).squeeze(-1)
    targets = torch.from_numpy(target).to(device)
    optimizer = torch.optim.Adam(model.value_adapter.parameters(), lr=0.001)
    # Retain existing critic Adam moments/clock (4 updates); actor Adam untouched.
    old = torch.load(a.policy, map_location="cpu", weights_only=True)
    oldopt = old["optimizer"]
    state = optimizer.state_dict()
    if oldopt["param_groups"][0]["params"] != list(range(9)):
        raise ValueError("exact original actor/logstd/critic optimizer order required")
    for i, parameter in enumerate(model.value_adapter.parameters()):
        if any(
            oldopt["state"][i + 5][name].shape != parameter.shape
            for name in ("exp_avg", "exp_avg_sq")
        ):
            raise ValueError("original critic moment tensor mapping differs")
    state["state"] = {i: oldopt["state"][i + 5] for i in range(4)}
    optimizer.load_state_dict(state)
    generator = torch.Generator().manual_seed(a.seed)
    history = []
    for step in range(1, 1001):
        sampled = torch.randperm(24, generator=generator)[:8].tolist()
        batch = [fit[i] for i in sampled]
        optimizer.zero_grad(set_to_none=True)
        pred = base_value[batch] + model.value_adapter(context[batch]).squeeze(-1)
        loss = (pred[:, 20:] - targets[batch, 20:]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite value calibration")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.value_adapter.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        if step == 1 or step % 250 == 0:
            with torch.no_grad():
                values = (base_value + model.value_adapter(context).squeeze(-1)).cpu().numpy()
            row = dict(
                step=step,
                fit=value_metrics(values[fit], target[fit]),
                hold=value_metrics(values[hold], target[hold]),
            )
            history.append(row)
            write_once(a.output / f"update{step:04d}.json", row)
            print("CRITIC_ONLY_CALIBRATION", a.seed, row, flush=True)
    model.cpu().eval()
    with torch.no_grad():
        mean, std, value, hidden = model.parameters_for_state(x)
    for name, left, right in (
        ("mean", mean, original_mean),
        ("std", std, original_std),
        ("hidden", hidden, original_hidden),
    ):
        exact_tensor_tree(left, right, path="actor." + name)
    changes = {
        k: float((v - before[k]).double().square().sum()) for k, v in model.state_dict().items()
    }
    if any(v != 0 for k, v in changes.items() if not k.startswith("value_adapter.")) or not any(
        v > 0 for v in changes.values()
    ):
        raise ValueError("only critic column may change")
    initial = value_metrics(initial_value[hold].numpy(), target[hold])
    final = value_metrics(value[hold].numpy(), target[hold])
    moments = optimizer.state_dict()
    # Store optimizer by PARAMETER NAME; no ambiguous implicit remapping downstream.
    names = [n for n, _ in model.value_adapter.named_parameters()]
    named_moments = {
        name: {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in moments["state"][i].items()
        }
        for i, name in enumerate(names)
    }
    if any(float(v["step"]) != 1004 for v in named_moments.values()):
        raise ValueError("critic Adam lineage did not continue from original4")
    torch.save(
        dict(
            state_dict=model.state_dict(),
            source_head=a.expected_head,
            model_kind="critic_calibrated_kinematic_motion_residual64",
            actor_checkpoint_hash=a.expected_policy_hash,
            critic_optimizer_named_state=named_moments,
            frozen_actor_optimizer_state=oldopt,
            critic_new_optimizer_steps=1000,
            actor_new_optimizer_steps=0,
            world_hash=digest(Path(spec["world"])),
            cpu_rng_state=torch.get_rng_state(),
            minibatch_rng_state=generator.get_state(),
            hardware_authorized=False,
            promotion_authorized=False,
        ),
        a.output / "critic-candidate.pt",
    )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="FIXED_ACTOR_CRITIC_CALIBRATION_CLOSED",
            source_head=a.expected_head,
            seed=a.seed,
            history=history,
            initial_internal_holdout=initial,
            final_internal_holdout=final,
            fit_course_indices=[indices[i] for i in fit],
            internal_holdout_course_indices=[indices[i] for i in hold],
            internal_calibration_gate=bool(
                final["explained_variance"] is not None
                and final["explained_variance"] >= 0.25
                and final["mse"] <= 0.8 * initial["mse"]
            ),
            actor_mean_std_hidden_and_all_noncritic_parameters_bit_exact=True,
            checkpoint_hash=digest(a.output / "critic-candidate.pt"),
            new_native_steps=0,
            actor_optimizer_steps=0,
            critic_optimizer_steps=1000,
            critic_adam_step=1004,
            new_football_skill_claimed=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
