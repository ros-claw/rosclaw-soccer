"""Fixed categorical-symbol learning pilot, with immutable body/muscle parent."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inputs",
        "core-source",
        "bank",
        "policy",
        "output",
        "retention-exam",
        "retention-audit",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-inputs-hash",
        "expected-bank-hash",
        "expected-policy-hash",
        "expected-retention-audit-hash",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_train_grouped_actor import load_retention

    spec, bd, pins, _ = admit(a)
    if (
        digest(a.bank / "complete.json") != a.expected_bank_hash
        or digest(a.policy) != a.expected_policy_hash
    ):
        raise ValueError("exact motion teacher bank and frozen research anchor required")
    bc = json.loads((a.bank / "complete.json").read_text())
    td = json.loads((a.bank / "declaration.json").read_text())
    if (
        bc["status"] != "ALL40_TEACHER_NN_EXACT_MOTION_LABELS_NOT_TRAINED_POLICY"
        or bc["actions_byte_exact"] != 12000
        or bc["qualified_train_courses"] != 10
        or digest(a.bank / bc["artifact"]) != bc["artifact_hash"]
    ):
        raise ValueError("authenticated complete teacher motion data required")
    for k, h in td["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting teacher data commitment")
        pins[k] = h
    for name in ("complete.json", "declaration.json", bc["artifact"]):
        pins[str(a.bank / name)] = digest(a.bank / name)
    pins[str(a.policy)] = a.expected_policy_hash
    retention = load_retention(a, spec, pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            seed=a.seed,
            bank_hash=a.expected_bank_hash,
            anchor_hash=a.expected_policy_hash,
            actor_inputs="current108 plus frozen current raw64 and measured geometry26",
            optimizer_steps=1000,
            learning_rate=0.0003,
            classes_per_dimension=32,
            motion_dimensions=64,
            protected_train_memories=[0, 10, 15, 16],
            own_memory_sequence_weight=2,
            teacher_sequence_weight=1,
            validation_used_in_gradient=False,
            same_state_action_classification_not_physical_success=True,
            new_native_steps=0,
            online_reinforcement_learning=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    try:
        result = train(a, spec, bd, bc, retention)
        check_pins(pins)
        write_once(a.output / "complete.json", dict(source_head=a.expected_head, **result))
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(error=repr(error), automatic_retry_authorized=False, new_native_steps=0),
        )
        raise


def train(a, spec, bd, bc, retention):
    import numpy as np
    import torch
    from torch.nn import functional as F

    from rosclaw_soccer.rsi.categorical_motion import create_categorical_motion
    from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_online_kinematic_motion import load_model

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise ValueError("declared actual CUDA learner required")
    torch.cuda.set_per_process_memory_fraction(0.1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    with np.load(a.bank / bc["artifact"], allow_pickle=False) as z:
        x, y, split, actor = (
            z[k].copy() for k in ("features", "motion_tokens", "train", "actor_eligible")
        )
    if (
        x.shape != (40, 300, 108)
        or y.shape != (40, 300, 64)
        or x.dtype != np.float32
        or y.dtype != np.float32
        or split.dtype != np.bool_
        or actor.dtype != np.bool_
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.array_equal(y * 16, np.round(y * 16))
        or y.min() < -1
        or y.max() > 0.9375
        or not np.array_equal(split, np.array([r["split"] == "train" for r in bd["rows"]]))
    ):
        raise ValueError("exact original32/8 finite causal observation/symbol data required")
    selected = [i for i in range(40) if split[i] and actor[i] and i not in (0, 10, 15, 16)]
    if selected != [2, 6, 12, 22, 24, 32, 34, 36]:
        raise ValueError("declared8 non-conflicting teacher train contexts required")
    x = np.concatenate((x[selected], retention["features"]))
    y = np.concatenate((y[selected], retention["tokens"]))
    if x.shape != (12, 300, 108) or y.shape != (12, 300, 64):
        raise ValueError("twelve declared training memories required")
    parent = load_model(spec, a.policy).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    model = create_categorical_motion(parent)
    with torch.no_grad():
        context, prior, _, _ = model.context(torch.from_numpy(x))
    if not torch.equal((prior[8:, 20:].float() - 16) / 16, torch.from_numpy(y[8:, 20:])):
        raise ValueError("zero-head own-state token identity failed")
    device = torch.device("cuda:0")
    model.to(device)
    context, prior = context.to(device), prior.to(device)
    labels = torch.from_numpy((y * 16 + 16).astype(np.int64)).to(device)
    weight = torch.tensor([1.0] * 8 + [2.0] * 4, device=device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=0.0003)

    def save(name, steps):
        torch.save(
            dict(
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),
                model_kind="categorical_fsq32_motion64",
                source_head=a.expected_head,
                optimizer_steps=steps,
                seed=a.seed,
                anchor_hash=a.expected_policy_hash,
                world_hash=digest(Path(spec["world"])),
                hardware_authorized=False,
                promotion_authorized=False,
            ),
            a.output / name,
        )

    save("initial.pt", 0)
    initial = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
    final = {}
    for step in range(1, 1001):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(context, prior)[:, 20:]
        ce = (
            F.cross_entropy(logits.reshape(-1, 32), labels[:, 20:].reshape(-1), reduction="none")
            .reshape(12, 280, 64)
            .mean((1, 2))
        )
        loss = (ce * weight).sum() / weight.sum()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite categorical objective")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            with torch.no_grad():
                predicted = model.logits(context, prior)[:, 20:].argmax(-1)
                match = predicted == labels[:, 20:]
            final = dict(
                step=step,
                cross_entropy=float(loss.detach()),
                gradient_norm=float(grad),
                per_sequence_symbol_accuracy=match.float().mean((1, 2)).tolist(),
                per_sequence_exact_frame_accuracy=match.all(-1).float().mean(-1).tolist(),
                protected_memory_changed_symbols=(~match[8:]).sum((1, 2)).tolist(),
                same_state_accuracy_is_not_closed_loop_success=True,
            )
            write_once(a.output / f"step{step:04d}.json", final)
            print("CATEGORICAL_SYMBOL_LEARNING", a.seed, final, flush=True)
    model.cpu().eval()
    for k, v in model.parent.state_dict().items():
        exact_tensor_tree(v, before[k], path=k)
    change = sum(
        float((v.detach().cpu() - initial[k].cpu()).square().sum())
        for k, v in model.head.state_dict().items()
    )
    if not change > 0:
        raise ValueError("no actual classifier update")
    save("candidate.pt", 1000)
    return dict(
        status="CATEGORICAL_MOTION_TRAINED_REQUIRES_NATIVE_EXAM",
        checkpoint_hash=digest(a.output / "candidate.pt"),
        initial_checkpoint_hash=digest(a.output / "initial.pt"),
        anchor_hash=a.expected_policy_hash,
        optimizer_steps=1000,
        parameter_change_l2_squared=change,
        entire_parent_unchanged=True,
        teacher_train_indices=selected,
        own_train_indices=[0, 10, 15, 16],
        validation_sequences=0,
        new_native_steps=0,
        online_reinforcement_learning=False,
        hardware_authorized=False,
        promotion_authorized=False,
        **final,
    )


if __name__ == "__main__":
    main()
