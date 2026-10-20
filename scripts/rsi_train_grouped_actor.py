"""Critic-free grouped actor update with finite KL-constrained Adam proposals."""

import argparse
import copy
import json
import math
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "core-source", "bank", "policy", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-inputs-hash",
        "expected-bank-hash",
        "expected-policy-hash",
        "bank-unit",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--retention-exam", type=Path)
    p.add_argument("--retention-audit", type=Path)
    p.add_argument("--expected-retention-audit-hash")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, _, pins, _ = admit(a)
    wait_for_terminal_collection(a.bank_unit, wait=True)
    if (
        digest(a.bank / "complete.json") != a.expected_bank_hash
        or digest(a.policy) != a.expected_policy_hash
    ):
        raise ValueError("closed whole32x4 bank and exact behavior anchor required")
    bc = json.loads((a.bank / "complete.json").read_text())
    if (
        bc["status"] != "FOUR_GROUP_CRITIC_FREE_MOTION_BANK_CLOSED"
        or bc["behavior_checkpoint_hash"] != a.expected_policy_hash
        or bc["training_contexts"] != 32
        or bc["validation_contexts"] != 0
        or bc["episode_executions"] != 128
        or digest(a.bank / bc["artifact"]) != bc["artifact_hash"]
    ):
        raise ValueError("matched full training-only bank identity differs")
    for k, h in bc["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting evidence identity")
        pins[k] = h
    pins.update(
        {
            str(a.bank / "complete.json"): a.expected_bank_hash,
            str(a.bank / bc["artifact"]): bc["artifact_hash"],
            str(a.policy): a.expected_policy_hash,
        }
    )
    # No control helper can change between collected behavior and learning.
    source = [
        Path(k).parent.parent
        for k in bc["input_file_hashes"]
        if k.endswith("/scripts/rsi_online_kinematic_motion.py")
        and Path(k).parent.parent.name == "rosclaw-soccer-online-geometry-source-20261020"
    ]
    if (
        len(source) != 1
        or subprocess.check_output(
            ["git", "-C", str(source[0]), "rev-parse", "HEAD"], text=True
        ).strip()
        != "37174508238b36e08983f0002f2ef61c79b82b89"
    ):
        raise ValueError("one original immutable shared actor source required")
    for folder in ("src", "scripts"):
        for old in (source[0] / folder).rglob("*.py"):
            if digest(old) != digest(Path.cwd() / old.relative_to(source[0])):
                raise ValueError("original control helpers changed")
    check_pins(pins)
    retention = None
    if any((a.retention_exam, a.retention_audit, a.expected_retention_audit_hash)):
        if not all((a.retention_exam, a.retention_audit, a.expected_retention_audit_hash)):
            raise ValueError("complete explicit retention evidence identity required")
        retention = load_retention(a, spec, pins)
        check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.grouped_actor_learning.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            bank_hash=a.expected_bank_hash,
            behavior_checkpoint_hash=a.expected_policy_hash,
            max_adam_proposals=4,
            maximum_mean_gaussian_kl_old_to_new=0.005,
            raw_residual_cap=0.125,
            critic_frozen_and_not_used_in_advantage=True,
            protected_deterministic_train_memory=[0, 10, 15, 16] if retention else [],
            maximum_protected_token_changes=0 if retention else None,
            guarded_direction_feasibility_probe=retention is not None,
            parent_and_geometry_normalizer_frozen=True,
            global_advantage_scale_only=True,
            per_context_std_normalization=False,
            control_step_surrogate_not_unbiased_full_path_importance_sampling=True,
            original_far_or_private_fresh=False,
            new_native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        result = train(a, spec, bc, pins, retention=retention)
        check_pins(pins)
        write_once(a.output / "complete.json", dict(source_head=a.expected_head, **result))
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                new_native_steps=0,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


def load_retention(a, spec, pins):
    import numpy as np

    from scripts.rsi_native_context_recovery_common import digest

    receipt = a.retention_audit / "complete.json"
    if digest(receipt) != a.expected_retention_audit_hash:
        raise ValueError("unchanged closed anchor audit required")
    audit = json.loads(receipt.read_text())
    if (
        audit["status"] != "ONLINE_THREE_ROUNDS_TENSOR_AND_OUTCOME_REPRODUCED"
        or audit["seed"] != 29
    ):
        raise ValueError("original seed29 whole pipeline audit required")
    for k, h in audit["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting retention input commitment")
        pins[k] = h
    pins[str(receipt)] = a.expected_retention_audit_hash
    features, tokens = [], []
    for i in (0, 10, 15, 16):
        root = a.retention_exam / f"evaluate{i % 4}" / f"course{i:02d}"
        raw, record = root / "native.npz", root / "complete.json"
        if str(raw) not in pins or str(record) not in pins:
            raise ValueError("retention memory not bound to authenticated raw evidence")
        row = json.loads(record.read_text())
        if (
            row["course_index"] != i
            or row["split"] != "train"
            or row["score"]["qualified"] is not True
            or row["prefix_diagnostics"]["post_boundary_control_can_cause_first_touch"] is not True
        ):
            raise ValueError("only physically successful training memory allowed")
        with np.load(raw, allow_pickle=False) as z:
            features.append(z["student_features"].copy())
            tokens.append(z["student_motion_token"].copy())
    return dict(features=np.stack(features), tokens=np.stack(tokens))


def train(a, spec, bc, pins, *, retention=None):
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.actor_trust_region import bounded_adam_proposal
    from rosclaw_soccer.rsi.grouped_motion_returns import leave_one_out_returns
    from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motor_student_rollout import exact_array
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_online_kinematic_motion import load_model

    torch.set_num_threads(1)
    torch.manual_seed(29)
    torch.use_deterministic_algorithms(True)
    with np.load(a.bank / bc["artifact"], allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    if data["features"].shape != (32, 4, 300, 108):
        raise ValueError("exact grouped current-observation tensor required")
    advantage, _ = leave_one_out_returns(data["rewards"])
    exact_array(advantage, data["advantages"])
    x = torch.from_numpy(data["features"].reshape(128, 300, 108))
    model = load_model(spec, a.policy)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    payload = torch.load(a.policy, map_location="cpu", weights_only=True)
    original_opt = copy.deepcopy(payload["optimizer"])
    if original_opt["param_groups"][0]["params"] != list(range(9)):
        raise ValueError("original complete9-parameter Adam lineage required")
    model.eval()
    with torch.no_grad():
        geometry = model.geometry(x)
        cpu_mean, cpu_std, _, _ = model.parameters_for_state(x, geometry=geometry)
    stored_mean = torch.from_numpy(data["mean"].reshape(128, 300, 64))
    stored_std = torch.from_numpy(data["std"].reshape(128, 300, 64))
    torch.testing.assert_close(cpu_mean, stored_mean, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cpu_std, stored_std, atol=1e-7, rtol=1e-6)
    if not torch.cuda.is_available():
        raise ValueError("declared CUDA actor learner required")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model.to(device)
    xt = x.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.load_state_dict(copy.deepcopy(original_opt))
    for group in optimizer.param_groups:
        group["lr"] = 1e-4
    model.value_adapter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    with torch.no_grad():
        base = model.parent.base
        norm = ((xt - base.observation_mean) / base.observation_std).clamp(-10, 10)
        memory, _ = base.memory(base.encoder(norm))
        raw_parent = base.actor(memory)
        context = torch.cat(
            (
                norm,
                memory,
                ((geometry.to(device) - model.geometry_mean) / model.geometry_std).clamp(-10, 10),
            ),
            -1,
        )
        ref_mean = raw_parent + 0.125 * torch.tanh(model.adapter(context))
        ref_std = (
            model.log_std.clamp(math.log(0.005), math.log(0.05)).exp().expand_as(ref_mean).clone()
        )
    torch.testing.assert_close(ref_mean.cpu(), stored_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ref_std.cpu(), stored_std, atol=1e-7, rtol=1e-6)
    raw = torch.from_numpy(data["raw"].reshape(128, 300, 64)).to(device)
    protected_context = protected_parent = protected_tokens = None
    if retention:
        rx = torch.from_numpy(retention["features"])
        rg = model.geometry(rx).to(device)
        with torch.no_grad():
            rx = rx.to(device)
            rn = ((rx - base.observation_mean) / base.observation_std).clamp(-10, 10)
            rm, _ = base.memory(base.encoder(rn))
            protected_parent = base.actor(rm)
            protected_context = torch.cat(
                (rn, rm, ((rg - model.geometry_mean) / model.geometry_std).clamp(-10, 10)), -1
            )
            protected_tokens = torch.from_numpy(retention["tokens"]).to(device)
            initial_mu = protected_parent + 0.125 * torch.tanh(model.adapter(protected_context))
            initial_tokens = (
                torch.round(15.515501022338867 * torch.tanh(initial_mu + 0.03223699703812599) - 0.5)
                / 16
            )
            if not torch.equal(initial_tokens[:, 20:], protected_tokens[:, 20:]):
                raise ValueError("initial CPU/native vs CUDA memory symbols differ")
    adv = torch.from_numpy(advantage.reshape(128, 300)).to(device)
    scale = adv[:, 20:].std(unbiased=False)
    if not torch.isfinite(scale) or float(scale) < 1e-8:
        raise ValueError("no group advantage variation")
    adv = adv / scale

    def log_prob(mean, std):
        return (-0.5 * ((raw - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)).sum(
            -1
        )

    with torch.no_grad():
        old_lp = log_prob(ref_mean, ref_std)
    recorded_lp = torch.from_numpy(data["log_prob"].reshape(128, 300)).to(device)
    torch.testing.assert_close(old_lp, recorded_lp, atol=0.005, rtol=0)
    error = float((old_lp - recorded_lp).abs().max())

    def closure():
        mean = raw_parent + 0.125 * torch.tanh(model.adapter(context))
        std = model.log_std.clamp(math.log(0.005), math.log(0.05)).exp().expand_as(mean)
        log_ratio = (log_prob(mean, std) - old_lp)[:, 20:]
        if not torch.isfinite(log_ratio).all() or log_ratio.abs().max() > 40:
            raise ValueError("stale or nonfinite grouped likelihood")
        ratio = log_ratio.exp()
        actor = -torch.minimum(ratio * adv[:, 20:], ratio.clamp(0.8, 1.2) * adv[:, 20:]).mean()
        forward = (
            torch.log(std / ref_std)
            + (ref_std.square() + (ref_mean - mean).square()) / (2 * std.square())
            - 0.5
        ).sum(-1)[:, 20:]
        reverse = (
            (
                torch.log(ref_std / std)
                + (std.square() + (mean - ref_mean).square()) / (2 * ref_std.square())
                - 0.5
            )
            .sum(-1)[:, 20:]
            .mean()
        )
        entropy = (std.log() + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)[:, 20:].mean()
        loss = actor + 0.01 * reverse - 0.0002 * entropy
        changes = torch.zeros((), device=device)
        if retention:
            with torch.no_grad():
                protected_mu = protected_parent + 0.125 * torch.tanh(
                    model.adapter(protected_context)
                )
                token = (
                    torch.round(
                        15.515501022338867 * torch.tanh(protected_mu + 0.03223699703812599) - 0.5
                    )
                    / 16
                )
                changes = (token[:, 20:] != protected_tokens[:, 20:]).sum()
        return loss, dict(
            protected_token_changes=changes,
            mean_kl=forward.mean(),
            maximum_frame_kl=forward.max(),
            reverse_kl=reverse,
            actor_loss=actor,
            entropy=entropy,
        )

    model.train()
    history = []
    for step in range(4):
        record = bounded_adam_proposal(trainable, optimizer, closure)
        record["proposal"] = step + 1
        write_once(a.output / f"proposal{step}.json", record)
        history.append(record)
        print("GROUPED_ACTOR_PROPOSAL", record, flush=True)
        if not record["accepted"]:
            break
    accepted = sum(r["accepted"] for r in history)
    model.cpu().eval()
    for k, v in model.state_dict().items():
        if not (k.startswith("adapter.") or k == "log_std"):
            exact_tensor_tree(v, before[k], path=k)
    states = optimizer.state_dict()
    for i in range(9):
        expected = 4 + accepted if i < 5 else 4
        if float(states["state"][i]["step"]) != expected:
            raise ValueError("actor/critic Adam clock violated")
        if i >= 5:
            exact_tensor_tree(
                states["state"][i], original_opt["state"][i], path="frozen_critic_moments"
            )
    torch.save(
        dict(
            state_dict=model.state_dict(),
            optimizer=states,
            source_head=a.expected_head,
            model_kind="online_kinematic_motion_residual64",
            learner_algorithm="group_leave_one_out_trust_region",
            protected_train_token_memory=retention is not None,
            actor_new_optimizer_steps=accepted,
            actor_optimizer_steps=4 + accepted,
            critic_optimizer_steps=4,
            parent_checkpoint_hash=spec["experts"][0]["checkpoint_hash"],
            behavior_checkpoint_hash=a.expected_policy_hash,
            world_hash=digest(Path(spec["world"])),
            bank_hash=a.expected_bank_hash,
            cpu_rng_state=torch.get_rng_state(),
            hardware_authorized=False,
            promotion_authorized=False,
        ),
        a.output / "candidate.pt",
    )
    return dict(
        status="GROUPED_ACTOR_UPDATED_REQUIRES_NATIVE_EXAM"
        if accepted
        else "GROUPED_ACTOR_UNCHANGED",
        checkpoint_hash=digest(a.output / "candidate.pt"),
        bank_hash=a.expected_bank_hash,
        behavior_checkpoint_hash=a.expected_policy_hash,
        new_native_steps=0,
        accepted_actor_optimizer_steps=accepted,
        protected_train_token_memory=retention is not None,
        optimizer_proposals=len(history),
        critic_optimizer_steps=0,
        history=history,
        global_advantage_scale=float(scale.cpu()),
        batched_vs_single_step_log_probability_max_error=error,
        critic_all_weights_and_moments_unchanged=True,
        whole_parent_unchanged=True,
        device=torch.cuda.get_device_name(),
        hardware_authorized=False,
        promotion_authorized=False,
    )


if __name__ == "__main__":
    main()
