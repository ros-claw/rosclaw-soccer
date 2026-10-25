"""Verify differentiable prior against all actual entries; no optimizer or physics."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-hash", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    bp = a.bank / "complete.json"
    if a.output.exists() or bp.is_symlink() or digest(bp) != a.bank_hash:
        raise ValueError("new evidence output and exact causal bank required")
    bank = json.loads(bp.read_text())
    if bank["status"] != "CAUSAL_KICK_ENTRY_BANK_RECONSTRUCTED_FIRST_ACTIONS_EXACT_NOT_TRAINED":
        raise ValueError("audited original causal entries required")
    path = a.bank / bank["artifact"]["path"]
    if (
        path.parent.resolve() != a.bank.resolve()
        or path.is_symlink()
        or (
            digest(path) != bank["artifact"]["hash"] or digest(a.model) != bank["policy_model_hash"]
        )
    ):
        raise ValueError("exact confined data and original prior required")
    pins = dict(bank["input_file_hashes"])
    pins.update(
        {
            str(bp): a.bank_hash,
            str(path): bank["artifact"]["hash"],
            str(a.model): bank["policy_model_hash"],
        }
    )
    source_pins(pins)
    check_pins(pins)
    import numpy as np
    import onnxruntime as ort
    import torch

    from rosclaw_soccer.rsi.kick_prior_actor import KICK_PRIOR_HASH, KickPriorActor

    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            bank_hash=a.bank_hash,
            input_file_hashes=pins,
            max_raw_action_absolute_error=1e-4,
            raw_action_rms_error=1e-5,
            required_entries=1052,
            actor_optimizer_updates=0,
            physical_steps_added=0,
            forward_order="one actual observation at a time, identical ONNX batch1 contract",
            gradient_check="all eight trainable tensors finite and nonzero; no optimizer.step",
            initialization_is_not_training=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    torch.set_num_threads(1)
    with np.load(path) as saved:
        inputs = saved["features"][:, :547].copy()
    if inputs.shape != (1052, 547) or not np.isfinite(inputs).all():
        raise ValueError("all unique actual finite547 policy inputs required")
    actor = KickPriorActor.from_onnx(a.model, expected_hash=KICK_PRIOR_HASH).eval()
    before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    original = ort.InferenceSession(
        str(a.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    expected, actual = [], []
    with torch.no_grad():
        for row in inputs:
            expected.append(original.run(["actions"], {"obs": row[None]})[0][0])
            actual.append(actor(torch.from_numpy(row[None])).numpy()[0])
    expected, actual = np.stack(expected), np.stack(actual)
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite prior copy output")
    error = actual.astype(np.float64) - expected.astype(np.float64)
    maximum, rms = float(np.abs(error).max()), float(np.sqrt(np.mean(error**2)))
    output = actor(torch.from_numpy(inputs[:8]))
    output.square().mean().backward()
    gradients = [
        dict(name=k, norm=float(v.grad.norm()), finite=bool(torch.isfinite(v.grad).all()))
        for k, v in actor.named_parameters()
        if v.grad is not None
    ]
    if len(gradients) != 8 or any(not g["finite"] or g["norm"] <= 0 for g in gradients):
        raise ValueError("all eight prior parameter tensors need finite nonzero gradients")
    if any(not torch.equal(before[k], v) for k, v in actor.state_dict().items()):
        raise ValueError("gradient-only verification unexpectedly changed prior weights")
    checkpoint = a.output / "kick-prior-initialization.pt"
    torch.save(
        dict(
            schema="soccer.g1.kick_prior_initialization.v1",
            state_dict=actor.state_dict(),
            source_model_hash=KICK_PRIOR_HASH,
            bank_hash=a.bank_hash,
            actor_optimizer_updates=0,
            simulation_only=True,
            hardware_authorized=False,
        ),
        checkpoint,
    )
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(not torch.equal(v, loaded["state_dict"][k]) for k, v in before.items()):
        raise ValueError("safe-loaded initialization differs from original tensors")
    arrays = a.output / "paired-raw-actions.npz"
    np.savez_compressed(arrays, onnx=expected, torch=actual)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="DIFFERENTIABLE_KICK_PRIOR_NUMERIC_FIDELITY_CHECKED_NOT_TRAINED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            max_raw_action_absolute_error=maximum,
            raw_action_rms_error=rms,
            numeric_fidelity_pass=maximum <= 1e-4 and rms <= 1e-5,
            entries=len(inputs),
            parameter_elements=sum(p.numel() for p in actor.parameters()),
            gradients=gradients,
            original_weights_unchanged=True,
            checkpoint=dict(path=checkpoint.name, hash=digest(checkpoint)),
            arrays=dict(path=arrays.name, hash=digest(arrays)),
            actor_optimizer_updates=0,
            physical_steps_added=0,
            reinforcement_learning=False,
            simulator_trajectory_equivalence_proven=False,
            physical_football_gain_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
