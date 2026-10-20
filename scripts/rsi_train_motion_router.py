"""Train measured-body applicability, never overwrite frozen motion memories."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "core-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-inputs-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_portfolio_common import admit, training_table
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, cases = admit(a)
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.frozen_motion_router_training.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            inputs_hash=a.expected_inputs_hash,
            expert_ids=[r["id"] for r in spec["experts"]],
            seed=17,
            optimizer_steps=2000,
            learning_rate=0.001,
            success_loss_weight=0.25,
            route_confidence_threshold=0.75,
            expert_parameters_updated=False,
            validation_in_training=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np
    import torch
    import torch.nn.functional as F

    from rosclaw_soccer.rsi.motion_capability_router import create_motion_capability_router

    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    x, success, targets, indices = training_table(bd, cases)
    np.savez_compressed(
        a.output / "training-table.npz",
        features=x,
        success=success,
        targets=targets,
        training_course_indices=np.asarray(indices, dtype=np.int64),
    )
    model = create_motion_capability_router(experts=5)
    with torch.no_grad():
        model.observation_mean.copy_(torch.from_numpy(x.mean(axis=0)))
        model.observation_std.copy_(torch.from_numpy(np.maximum(x.std(axis=0), 0.01)))
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    xt, yt, st = torch.from_numpy(x), torch.from_numpy(targets), torch.from_numpy(success)
    counts = torch.bincount(yt, minlength=5).float()
    if (counts == 0).any():
        raise ValueError("every declared memory must have training-only routing support")
    weights = torch.sqrt(counts.max() / counts)
    positive = st.sum(0)
    pos_weight = (32 - positive) / positive.clamp_min(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = []
    for step in range(1, 2001):
        optimizer.zero_grad(set_to_none=True)
        logits, scores = model(xt)
        ce = F.cross_entropy(logits, yt, weight=weights)
        bce = F.binary_cross_entropy_with_logits(scores, st, pos_weight=pos_weight)
        loss = ce + 0.25 * bce
        if not torch.isfinite(loss):
            raise ValueError("nonfinite measured applicability loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            with torch.no_grad():
                owner_logits, _ = model(xt)
                probs = owner_logits.softmax(-1)
                confidence, owners = probs.max(-1)
                owners = torch.where(
                    (owners != 0) & (confidence < 0.75), torch.zeros_like(owners), owners
                )
                table_success = st[torch.arange(32), owners]
            row = dict(
                step=step,
                loss=float(loss.detach()),
                training_owner_accuracy=float((owners == yt).float().mean()),
                training_table_selected_successes=int(table_success.sum()),
                table_is_not_new_physical_execution=True,
            )
            metrics.append(row)
            write_once(a.output / f"update{step:04d}.json", row)
            print("MOTION_ROUTER_TRAINING", json.dumps(row), flush=True)
    change = sum(
        float((v - before[k]).double().square().sum()) for k, v in model.state_dict().items()
    )
    if change <= 0:
        raise ValueError("router parameters must actually change")
    model.eval().requires_grad_(False)
    torch.save(
        dict(
            state_dict=model.state_dict(),
            source_head=a.expected_head,
            model_kind="measured_motion_router5",
            inputs_hash=a.expected_inputs_hash,
            optimizer_steps=2000,
            optimizer=optimizer.state_dict(),
            rng_state=torch.get_rng_state(),
            expert_checkpoint_hashes=[r["checkpoint_hash"] for r in spec["experts"]],
            hardware_authorized=False,
            promotion_authorized=False,
        ),
        a.output / "router.pt",
    )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="MOTION_ROUTER_TRAINED_REQUIRES_EXAM",
            source_head=a.expected_head,
            inputs_hash=a.expected_inputs_hash,
            optimizer_steps=2000,
            checkpoint_hash=digest(a.output / "router.pt"),
            training_table_hash=digest(a.output / "training-table.npz"),
            parameter_change_l2_squared=change,
            training_courses=32,
            validation_courses=0,
            oracle_training_successes=7,
            metrics=metrics,
            new_native_steps=0,
            low_level_experts_updated=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
